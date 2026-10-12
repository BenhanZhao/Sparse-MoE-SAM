"""Synthetic GPU checks of the real ViT-B configuration (no weights/downloads)."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_moe import SparseMoESAM
from losses import SegmentationLoss
from inspect_model import model_settings, parameter_inventory, source_fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    size = 256 if args.full else 64
    model = SparseMoESAM(encoder_size=1024 if args.full else 320,
                         channels=256 if args.full else 16).cuda().train()
    source_snapshot = {path: digest for path, digest in source_fingerprints().items()
                       if path in {"sam_moe.py", "sparse_modules.py", "losses.py",
                                   "inspect_model.py", "tests/gpu_smoke.py"}
                       or path.startswith("modeling/")}
    images = torch.randn(1, 3, size, size, device="cuda")
    targets = (torch.rand(1, 1, size, size, device="cuda") > .8).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda"):
        logits, aux = model(images, return_aux=True)
        loss = SegmentationLoss()(logits, targets, aux)
    if logits.shape != targets.shape or not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise FloatingPointError("Invalid GPU training output or loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    if not finite:
        raise FloatingPointError("Nonfinite GPU gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()
    optimizer_step = bool(optimizer.state) and all(float(state["step"]) == 1
                                                  for state in optimizer.state.values())
    if not optimizer_step:
        raise RuntimeError("AMP optimizer step was skipped or not recorded")
    torch.cuda.synchronize()
    result = {"architecture_version": 2,
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "encoder": "vit_b", "encoder_size": model.encoder_size,
              "decoder_channels": 256 if args.full else 16, "batch": 1,
              "input_size": size, "output_shape": list(logits.shape), "amp": True,
              "synthetic_loss": float(loss.detach()), "finite_gradients": finite,
              "gradient_parameter_tensors": len(gradients),
              "optimizer_step": optimizer_step,
              "optimizer_state_parameter_tensors": len(optimizer.state),
              "parameters": parameter_inventory(model), "settings": model_settings(model),
              "core_source_sha256": source_snapshot,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "single_step_seconds": time.perf_counter() - started,
              "note": "Random initialization; one synthetic step, not a latency benchmark or accuracy evaluation"}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda"):
        inference = model(images)
    if inference.shape != targets.shape or not torch.isfinite(inference).all():
        raise RuntimeError("Invalid GPU inference result")
    result["inference_finite"] = True
    current_sources = source_fingerprints()
    result["source_changed_during_check"] = any(current_sources.get(path) != digest
                                                for path, digest in source_snapshot.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
