"""Offline model inventory; no dataset, checkpoint, training or forward pass.

This reports the current implementation, not historical experimental evidence.
Static module registrations/call expressions do not establish runtime execution.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import torch
from torch import nn

try:
    from .sam_moe import ENCODERS, SparseMoESAM
except ImportError:
    from sam_moe import ENCODERS, SparseMoESAM


ROOT = Path(__file__).resolve().parent
PAPER_REFERENCE_PARAMETERS = 142_700_000
SKIP_SOURCE_DIRECTORIES = {".git", ".venv", "venv", "env", "__pycache__", "node_modules"}


def source_fingerprints(source_root=ROOT):
    """Hash project Python source bytes in reproducible relative-path order."""
    source_root = Path(source_root).resolve()
    hashes = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if any(part in SKIP_SOURCE_DIRECTORIES for part in relative.parts):
            continue
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def parameter_inventory(model):
    """Partition PyTorch's deduplicated named parameters by their owner path."""
    groups = {name: {"total": 0, "trainable": 0, "frozen": 0}
              for name in ("encoder", "decoder", "other")}
    for name, parameter in model.named_parameters():
        group = "encoder" if name.startswith("image_encoder.") else (
            "decoder" if name.startswith("decoder.") else "other")
        count = parameter.numel()
        groups[group]["total"] += count
        groups[group]["trainable" if parameter.requires_grad else "frozen"] += count
    return {"total": sum(group["total"] for group in groups.values()),
            "trainable": sum(group["trainable"] for group in groups.values()),
            "frozen": sum(group["frozen"] for group in groups.values()),
            "groups": groups,
            "counting_method": "Unique registered parameters; buffers are excluded."}


def _attributes(module, names):
    return {name: getattr(module, name, None) for name in names}


def _moe_settings(stage):
    if stage is None:
        return None
    gate = getattr(stage, "gate", None)
    gate_layers = list(gate.modules()) if gate is not None else []
    linear = [layer for layer in gate_layers if isinstance(layer, nn.Linear)]
    activations = [type(layer).__name__ for layer in gate_layers
                   if isinstance(layer, (nn.ReLU, nn.GELU, nn.LeakyReLU, nn.SiLU, nn.Tanh))]
    result = _attributes(stage, ("num_experts", "top_k", "temperature", "noise_scale",
                                 "balance_mode", "dispatch_chunk_size", "out_channels"))
    result.update(gate_hidden_dim=linear[0].out_features if len(linear) > 1 else None,
                  gate_activation=activations,
                  expert_types=[getattr(expert, "expert_type", None)
                                for expert in getattr(stage, "experts", [])])
    return result


def model_settings(model):
    """Read instantiated modules, allowing optional/unavailable attributes."""
    encoder = getattr(model, "image_encoder", None)
    decoder = getattr(model, "decoder", None)
    blocks = list(getattr(encoder, "blocks", []))
    patch_projection = getattr(getattr(encoder, "patch_embed", None), "proj", None)
    embed_dim = getattr(patch_projection, "out_channels", None)
    heads = getattr(getattr(blocks[0], "attn", None), "num_heads", None) if blocks else None
    matching_types = [name for name, config in ENCODERS.items()
                      if (embed_dim, len(blocks), heads) == config[:3]]
    aspp = getattr(decoder, "aspp", None)
    branch_dilations = []
    for branch in getattr(aspp, "branches", []):
        convolution = next((layer for layer in branch.modules() if isinstance(layer, nn.Conv2d)), None)
        branch_dilations.append(list(convolution.dilation) if convolution is not None else None)
    attention = [{"path": f"image_encoder.blocks.{index}.attn",
                  **_attributes(getattr(block, "attn", None),
                                ("num_heads", "sparsity_ratio", "temperature", "query_chunk_size"))}
                 for index, block in enumerate(blocks)]
    fusion_attention = getattr(aspp, "sparse_attention", None)
    if fusion_attention is not None:
        attention.append({"path": "decoder.aspp.sparse_attention",
                          **_attributes(fusion_attention,
                                        ("num_heads", "sparsity_ratio", "temperature", "query_chunk_size"))})
    coarse_head = getattr(decoder, "coarse_head", None)
    lateral_indexes = getattr(model, "lateral_indexes", None)
    lateral_projections = getattr(decoder, "lateral_projections", None)
    return {
        "model_class": type(model).__name__,
        "encoder_size": getattr(model, "encoder_size", None),
        "encoder_variant_inferred_from_dimensions": matching_types[0] if len(matching_types) == 1 else None,
        "encoder_embed_dim": embed_dim,
        "encoder_depth": len(blocks),
        "encoder_heads": heads,
        "encoder_global_attention_indexes": [index for index, block in enumerate(blocks)
                                              if getattr(block, "window_size", None) == 0],
        "patch_kernel": list(patch_projection.kernel_size) if patch_projection is not None else None,
        "patch_stride": list(patch_projection.stride) if patch_projection is not None else None,
        "pretrained_loaded": getattr(model, "pretrained_loaded", None),
        "decoder_channels": getattr(coarse_head, "in_channels", None),
        "aspp_branch_dilations": branch_dilations,
        "lateral_indexes": list(lateral_indexes) if lateral_indexes is not None else None,
        "lateral_projection_count": len(lateral_projections) if lateral_projections is not None else None,
        "refinement_se_class": type(decoder.refinement_se).__name__ if hasattr(decoder, "refinement_se") else None,
        "boundary_mode": getattr(decoder, "boundary_mode", getattr(model, "boundary_mode", None)),
        "stage1": _moe_settings(getattr(decoder, "stage1", None)),
        "stage2": _moe_settings(getattr(decoder, "stage2", None)),
        "attention": attention,
    }


def static_module_structure(model, source_root=ROOT):
    """Registered paths and source-level self calls; deliberately not a trace."""
    source_root = Path(source_root).resolve()
    registrations, classes = [], {}
    for path, module in model.named_modules():
        registrations.append({"path": path or "<root>",
                              "class": f"{type(module).__module__}.{type(module).__qualname__}"})
        module_class = type(module)
        try:
            source_file = Path(inspect.getsourcefile(module_class)).resolve()
            relative = source_file.relative_to(source_root)
        except (TypeError, ValueError):
            continue
        key = f"{module_class.__module__}.{module_class.__qualname__}"
        if key in classes:
            continue
        try:
            source, start_line = inspect.getsourcelines(module_class.forward)
            syntax = ast.parse(textwrap.dedent("".join(source)))
            calls = [{"expression": ast.unparse(node.func),
                      "source_line": start_line + node.lineno - 1}
                     for node in ast.walk(syntax) if isinstance(node, ast.Call)
                     and ast.unparse(node.func).startswith("self.")]
            calls.sort(key=lambda item: item["source_line"])
            classes[key] = {"source_file": relative.as_posix(), "forward_self_calls": calls}
        except (OSError, TypeError, IndentationError, SyntaxError):
            classes[key] = {"source_file": relative.as_posix(), "forward_self_calls": None}
    return {"registered_modules": registrations, "local_class_forward_calls": classes,
            "interpretation": "Static registration paths and source expressions only. "
                              "No forward pass was run; conditional/iterated calls and "
                              "functional tensor operations are not a verified execution graph."}


def build_report(model, source_root=ROOT, requested_configuration=None):
    parameters = parameter_inventory(model)
    delta = parameters["total"] - PAPER_REFERENCE_PARAMETERS
    return {
        "schema_version": 1,
        "audit_scope": {"dataset_loaded": False, "checkpoint_loaded_by_audit": False,
                        "forward_executed": False, "training_executed": False,
                        "accuracy_evaluated": False, "latency_measured": False},
        "requested_configuration": requested_configuration,
        "settings": model_settings(model),
        "parameters": parameters,
        "paper_comparison": {
            "reference_parameter_count": PAPER_REFERENCE_PARAMETERS,
            "reference_label": "Published full model parameter count (142.7M; rounded reference).",
            "delta_actual_minus_reference": delta,
            "delta_percent": 100 * delta / PAPER_REFERENCE_PARAMETERS,
            "status": "unresolved",
            "note": "The original experimental configuration and checkpoint are required "
                    "to reconcile parameter counts. A count match would not verify "
                    "architectural or experimental reproduction.",
        },
        "unresolved": [
            "Historical complete source, model configuration and checkpoint have not been verified.",
            "Paper accuracy, FLOPs reduction and speed claims are not validated by this inventory.",
            "Source hashes identify the current files; they do not prove historical experiment provenance.",
        ],
        "structure": static_module_structure(model, source_root),
        "source_sha256": source_fingerprints(source_root),
        "environment": {"torch_version": torch.__version__},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-type", choices=tuple(ENCODERS), default="vit_b")
    parser.add_argument("--encoder-size", type=int, default=1024)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # Constructor has no checkpoint path; all weights are random and local.
    model = SparseMoESAM(model_type=args.sam_type, encoder_size=args.encoder_size,
                         channels=args.decoder_channels)
    report = build_report(model, requested_configuration={"sam_type": args.sam_type,
                          "encoder_size": args.encoder_size, "decoder_channels": args.decoder_channels})
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return report


if __name__ == "__main__":
    main()
