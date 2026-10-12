"""Train/evaluate semantic Sparse-MoE-SAM; see README for reproduction limits."""
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from .data import PlantDiseaseDataset, prepare_splits
    from .losses import SegmentationLoss, BinaryMetrics
    from .sam_moe import SparseMoESAM, SparseSAMAttention
    from .sparse_modules import SparseAttention, MixtureOfExperts, ConvBlock, ExpertLayer, EnhancedASPP
    from .deeplab_baseline import EnhancedDeepLabV3Plus
    from .inspect_model import source_fingerprints
except ImportError:
    from data import PlantDiseaseDataset, prepare_splits
    from losses import SegmentationLoss, BinaryMetrics
    from sam_moe import SparseMoESAM, SparseSAMAttention
    from sparse_modules import SparseAttention, MixtureOfExperts, ConvBlock, ExpertLayer, EnhancedASPP
    from deeplab_baseline import EnhancedDeepLabV3Plus
    from inspect_model import source_fingerprints


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/sam"))
    p.add_argument("--model", choices=["sam", "deeplab"], default="sam")
    p.add_argument("--sam-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    p.add_argument("--sam-checkpoint", type=Path)
    p.add_argument("--allow-random-init", action="store_true", help="Structural/debug runs only")
    p.add_argument("--pretrained-resnet", action="store_true")
    p.add_argument("--groups-json", type=Path)
    p.add_argument("--gate-activation", choices=["relu", "gelu"], default="relu")
    p.add_argument("--balance-mode", choices=["frequency_st", "probability"], default="frequency_st")
    p.add_argument("--boundary-mode", choices=["predicted_sdf", "none"], default="predicted_sdf")
    for name, default in dict(encoder_size=1024, image_size=256, decoder_channels=256,
                              query_chunk_size=128, epochs=100, batch_size=16,
                              accumulation_steps=1, workers=0, seed=42, warmup_epochs=10,
                              freeze_encoder_epochs=10, specialize_epochs=20, gate_hidden_dim=512).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in dict(sparsity_ratio=.1, encoder_lr=3e-4, decoder_lr=1e-3,
                              gate_lr=1e-4, expert_lr=3e-4, min_lr=1e-6, weight_decay=1e-2,
                              balance_weight=.01, sparse_weight=.001, entropy_weight=0., threshold=.5,
                              stage1_temperature=.5, stage2_temperature=.3).items():
        p.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    p.add_argument("--pos-weight", type=float)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-only", action="store_true")
    return p


def make_optimizer(model, args):
    groups = {name: [] for name in ("encoder", "gate", "expert", "decoder")}
    gate_ids = {id(parameter) for module in model.modules() if isinstance(module, MixtureOfExperts)
                for parameter in module.gate.parameters()}
    expert_ids = {id(parameter) for module in model.modules() if isinstance(module, MixtureOfExperts)
                  for parameter in module.experts.parameters()}
    for name, parameter in model.named_parameters():
        group = "encoder" if name.startswith("image_encoder.") else "gate" if id(parameter) in gate_ids else "expert" if id(parameter) in expert_ids else "decoder"
        groups[group].append(parameter)
    rates = dict(encoder=args.encoder_lr, gate=args.gate_lr, expert=args.expert_lr, decoder=args.decoder_lr)
    optimizer = torch.optim.AdamW([{"params": params, "lr": rates[name], "name": name}
                                   for name, params in groups.items() if params], weight_decay=args.weight_decay)
    def schedule(base):
        def factor(epoch):
            if args.warmup_epochs and epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs - 1)
            minimum = args.min_lr / base
            return minimum + (1 - minimum) * (1 + math.cos(math.pi * min(1, max(0, progress)))) / 2
        return factor
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(g["lr"]) for g in optimizer.param_groups])
    return optimizer, scheduler


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None,
              accumulation_steps=1, regularize=True, amp=False, threshold=.5):
    training = optimizer is not None
    metrics, loss_sum, sample_count = BinaryMetrics(threshold), 0., 0
    usage_sum = {"usage_stage1": torch.zeros(4), "usage_stage2": torch.zeros(3)}
    aux_sum = {key: 0. for key in ("balance", "sparse", "gate_entropy", "attention_entropy")}
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, aux = model(images, return_aux=True)
                loss = criterion(logits, masks, aux, regularize=regularize)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss")
            if training:
                # Sample-weighted accumulation, including a partial last batch.
                group_start = (step // accumulation_steps) * accumulation_steps
                group_end = min(group_start + accumulation_steps, len(loader))
                group_samples = min(group_end * loader.batch_size, len(loader.dataset)) - group_start * loader.batch_size
                scaler.scale(loss * images.shape[0] / group_samples).backward()
                if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            size = images.shape[0]
            loss_sum += float(loss.detach()) * size
            sample_count += size
            metrics.update(logits, masks)
            for key in usage_sum:
                usage_sum[key] += aux[key].detach().cpu() * size
            for key in aux_sum:
                aux_sum[key] += float(aux[key].detach()) * size
    if not sample_count:
        raise ValueError("Empty loader")
    return {"loss": loss_sum / sample_count, **metrics.compute(),
            **{key: value / sample_count for key, value in aux_sum.items()},
            **{key: (value / sample_count).tolist() for key, value in usage_sum.items()}}


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def set_attention_temperature(model, temperature):
    for module in model.modules():
        if isinstance(module, (SparseAttention, SparseSAMAttention)):
            module.temperature = temperature


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.image_size) < 1:
        raise ValueError("Epochs, batch size, accumulation and image size must be positive")
    if not 0 < args.sparsity_ratio <= 1 or not 0 <= args.threshold <= 1:
        raise ValueError("Invalid sparsity ratio or threshold")
    if args.model == "sam" and not any((args.sam_checkpoint, args.resume, args.eval_only, args.allow_random_init)):
        raise ValueError("Provide --sam-checkpoint (or --allow-random-init for debugging)")
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.eval_only and any((args.output_dir / name).exists() for name in ("last.pt", "best.pt")):
        raise ValueError("Existing training checkpoints: use --resume or a new output directory")
    manifest = prepare_splits(args.data_root, args.output_dir / "split.json", args.seed, args.groups_json)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(PlantDiseaseDataset(args.data_root, args.image_size,
                        train=split == "train", pairs=manifest[split]), batch_size=args.batch_size,
                        shuffle=split == "train", num_workers=args.workers,
                        generator=generator if split == "train" else None, pin_memory=device.type == "cuda")
               for split in ("train", "val", "test")}
    decoder_options = dict(gate_hidden_dim=args.gate_hidden_dim, gate_activation=args.gate_activation,
                           stage1_temperature=args.stage1_temperature, stage2_temperature=args.stage2_temperature,
                           balance_mode=args.balance_mode, boundary_mode=args.boundary_mode)
    model = (SparseMoESAM(args.sam_type, args.sam_checkpoint, args.encoder_size,
                         args.decoder_channels, args.sparsity_ratio, args.query_chunk_size, **decoder_options)
             if args.model == "sam" else EnhancedDeepLabV3Plus(pretrained=args.pretrained_resnet,
                channels=args.decoder_channels, sparsity_ratio=args.sparsity_ratio,
                query_chunk_size=args.query_chunk_size, **decoder_options)).to(device)
    criterion = SegmentationLoss(args.balance_weight, args.sparse_weight, args.entropy_weight, args.pos_weight).to(device)
    optimizer, scheduler = make_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start, best, history = 0, -1., []
    checkpoint_sources = None
    checkpoint_path = args.resume or (args.output_dir / "best.pt" if args.eval_only else None)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("Checkpoint predates the architecture update; use its matching source version")
        for key in ("model", "sam_type", "encoder_size", "decoder_channels", "image_size", "sparsity_ratio", "seed", "threshold",
                    "gate_hidden_dim", "gate_activation", "stage1_temperature", "stage2_temperature", "balance_mode", "boundary_mode"):
            if checkpoint["config"][key] != getattr(args, key):
                raise ValueError(f"Checkpoint configuration differs: {key}")
        if not args.eval_only:
            # Restoring state alone cannot reproduce a changed augmentation,
            # effective batch, phase schedule, loss or LR schedule.
            for key in ("epochs", "batch_size", "accumulation_steps", "workers", "warmup_epochs",
                        "freeze_encoder_epochs", "specialize_epochs", "encoder_lr", "decoder_lr",
                        "gate_lr", "expert_lr", "min_lr", "weight_decay", "balance_weight",
                        "sparse_weight", "entropy_weight", "pos_weight", "amp", "query_chunk_size"):
                if checkpoint["config"][key] != getattr(args, key):
                    raise ValueError(f"Resume configuration differs: {key}; use the original configuration")
        if checkpoint["split"] != manifest:
            raise ValueError("Checkpoint dataset split differs")
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.model == "sam":
            model.pretrained_loaded = checkpoint.get("pretrained_loaded", False)
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            restore_rng(checkpoint["rng"], generator)
            start, best, history = checkpoint["epoch"] + 1, checkpoint["best_dice"], checkpoint["history"]
        temperature = checkpoint["attention_temperature"]
        checkpoint_sources = checkpoint.get("source_sha256")
        del checkpoint
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_name = "eval_config.json" if args.eval_only else "resume_config.json" if args.resume else "config.json"
    (args.output_dir / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    sources = source_fingerprints()
    source_manifest = {"architecture_version": 2, "source_sha256": sources,
                       "parameter_count": sum(p.numel() for p in model.parameters()),
                       "torch_version": str(torch.__version__),
                       "checkpoint_source_sha256": checkpoint_sources,
                       "note": "Current source identity; not historical experiment provenance."}
    manifest_name = "eval_source_manifest.json" if args.eval_only else "resume_source_manifest.json" if args.resume else "source_manifest.json"
    (args.output_dir / manifest_name).write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(start, args.epochs if not args.eval_only else start):
        frozen = epoch < args.freeze_encoder_epochs
        model.train()
        model.set_encoder_trainable(not frozen)
        if not frozen:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        temperature = .1 ** (epoch / max(1, args.epochs - 1))
        set_attention_temperature(model, temperature)
        specialize = args.specialize_epochs > 0 and epoch >= max(args.freeze_encoder_epochs, args.epochs - args.specialize_epochs)
        criterion.balance_weight = args.balance_weight * (2 if specialize else 1)
        train = run_epoch(model, loaders["train"], device, criterion, optimizer, scaler,
                          args.accumulation_steps, regularize=not frozen, amp=args.amp, threshold=args.threshold)
        model.eval()
        val = run_epoch(model, loaders["val"], device, criterion, regularize=False, amp=args.amp, threshold=args.threshold)
        history.append({"epoch": epoch + 1, "attention_temperature": temperature,
                        "stage1_temperature": model.decoder.stage1.temperature,
                        "stage2_temperature": model.decoder.stage2.temperature, "encoder_frozen": frozen,
                        "train": train, "val": val})
        improved = val["dice"] > best
        best = max(best, val["dice"])
        scheduler.step()
        checkpoint = {"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "epoch": epoch, "best_dice": best, "history": history, "config": config,
                      "split": manifest, "rng": rng_state(generator), "attention_temperature": temperature,
                      "source_sha256": sources,
                      "pretrained_loaded": getattr(model, "pretrained_loaded", args.pretrained_resnet)}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pt")
        del checkpoint
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch + 1}/{args.epochs} train={train['loss']:.4f} val Dice={val['dice']:.4f} IoU={val['iou']:.4f}", flush=True)
    if not args.eval_only:
        best_path = args.output_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError("Resume with the original output directory containing best.pt")
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        temperature = checkpoint["attention_temperature"]
        del checkpoint
    set_attention_temperature(model, temperature)
    model.eval()
    test = run_epoch(model, loaders["test"], device, criterion, regularize=False, amp=args.amp, threshold=args.threshold)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(f"Test: {json.dumps(test)}", flush=True)
    return test


if __name__ == "__main__":
    main()
