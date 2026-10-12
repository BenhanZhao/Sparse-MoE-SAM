# Sparse-MoE-SAM

This repaired implementation follows the SAM encoder, sparse ASPP and spatial
two-stage MoE described in [Zhao et al., Plants 2025, 14, 2634](https://doi.org/10.3390/plants14172634).
It is a reconstruction and repair made on 2026-10-12, with explicit engineering
choices where the paper is incomplete or inconsistent. The original complete
experimental source, configuration and checkpoint have not been verified.
This version is not a verified reproduction of the paper's accuracy or speed
claims. The Chinese [paper alignment record](PAPER_ALIGNMENT.md) distinguishes
implemented features, engineering choices, missing details and evidence gaps.

## Inspect without a dataset

```powershell
python inspect_model.py --output runs/model_inventory.json
```

This creates a randomly initialized model locally and records actual parameter
counts, instantiated settings, Python source SHA-256 hashes and static module
structure. It loads no dataset or checkpoint and runs no forward pass. Static
registrations and source call expressions are not a runtime execution trace;
hashes identify the current files, not their historical experimental provenance.
The default ViT-B model has **97,419,068 parameters** (encoder 89,670,912;
decoder 7,748,156), versus the paper's full-model **142.7M** in Table 3.
This discrepancy is unresolved and requires the original experimental model
configuration/checkpoint; matching a parameter count alone would not verify
the architecture or results.
The archived [model inventory](validation/model_audit_current.json) records
the reviewed configuration and source hashes. New reports in `runs/` leave
the published validation records intact.

## Install and data

```powershell
python -m pip install -r requirements.txt
```

Install matching PyTorch/torchvision builds for your CUDA version. The SAM
components are included locally from Meta's Apache-2.0 source; their pinned
commit and local changes are recorded in [SAM_UPSTREAM.json](SAM_UPSTREAM.json)
and [LICENSE-SAM](LICENSE-SAM). No `segment_anything`, `transformers` or Keras installation is
needed. Training downloads no weights automatically.

```text
data/
  images/image1.jpg
  masks/image1.png
```

Image and mask filenames must have the same stem. Image extensions are
case-insensitive. Masks must contain only 0/1 or 0/255 in a single channel;
palette indices are preserved. Multiclass or soft masks require an explicit
data conversion. Images and masks must have matching original dimensions.
Masks use nearest-neighbor resizing; all geometric augmentations are paired.
Only images receive color jitter, Gaussian noise and ImageNet normalization.

The first run writes a seeded 70/15/15 split to `runs/sam/split.json` and later
runs reuse it exactly. Integer/group rounding changes the exact sample ratios.
Use `--groups-json plant_groups.json` to keep images of the same plant or
capture group together, e.g. `{"image1.jpg": "plant_001"}`. Without metadata,
the default is a shuffled image split; this cannot establish independence of
same-plant images. Class stratification requires disease metadata and is not
inferred from filenames. A changed dataset/seed requires a new output directory.

## Train, resume, evaluate

Obtain a corresponding **official SAM ViT-B checkpoint** from
[Meta's SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints),
then run inside this directory:

```powershell
python run_code.py --data-root "D:/datasets/plant" --sam-checkpoint "D:/weights/sam_vit_b_01ec64.pth" --batch-size 1 --accumulation-steps 16 --amp
```

The command uses the published epoch budget (100), 256x256 dataset images,
1024x1024 SAM encoder input, 256 decoder channels, top-2 experts, 0.1 attention
ratio, AdamW, 10 warm-up epochs and max gradient norm 1.0. Encoder input is
resized explicitly to preserve SAM's positional grid and ensure ASPP's
dilation-18 branch has noncentral samples. This encoder/input mapping is an
engineering choice: the paper does not specify the mapping for 256 images.
Keeping the original 256 image at SAM patch stride 16 would produce only a
16x16 grid and a degenerate dilation-18 branch. Alternative encoder sizes must
be multiples of 16 and at least 304; changing this is a new experiment.

Learning rates: encoder 3e-4, other decoder parameters 1e-3, routing MLPs 1e-4,
expert parameters 3e-4. Weight decay is 1e-2 (Section 4.6). The first 10 epochs
freeze the encoder and train segmentation loss only; the next phase unfreezes
the encoder and adds auxiliary losses. In the last 20 epochs the balance
weight doubles. Phase lengths and the doubling factor are engineering defaults
because the paper does not give them. Attention Gumbel temperature anneals
from 1.0 to 0.1; MoE stage temperatures stay at 0.5/0.3 following Section 4.6.
Both gates use hidden dimension 512 and ReLU. Section 4.3 instead specifies
GELU, and Section 4.7 describes routing-temperature annealing; those conflicting
statements are documented in [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md). Use `--gate-hidden-dim`,
`--gate-activation`, `--stage1-temperature` and `--stage2-temperature` for
explicit alternatives. All choices are saved in `config.json`/checkpoints.

`--batch-size 1 --accumulation-steps 16` reduces memory use. Accumulation is
sample-weighted, including the last partial batch. It does not make the
batch-level balance loss identical to a physical batch of 16. Query attention
and expert execution are chunked and checkpointed for training memory.
CUDA mixed precision is optional (`--amp`); singleton batches also work on CPU.

```powershell
python run_code.py --data-root "D:/datasets/plant" --output-dir runs/sam --resume runs/sam/last.pt --batch-size 1 --accumulation-steps 16 --amp
python run_code.py --data-root "D:/datasets/plant" --output-dir runs/sam --eval-only --batch-size 1 --amp
```

Use the same model, image/encoder sizes, gate/boundary/balance settings,
channels, seed, threshold and schedule when resuming. The resume path restores
optimizer, scheduler, scaler, epoch, RNG state and shuffle-generator state.
Current checkpoints use `format_version=2`; checkpoints from the earlier
architecture are rejected and require their matching source version. Evaluation
and resume write `eval_config.json` and `resume_config.json`, respectively,
preserving the original training `config.json`.
Training also records `source_manifest.json` with current Python source SHA-256,
architecture version, parameter count and PyTorch version. Resume/evaluation
write separate `resume_source_manifest.json` / `eval_source_manifest.json` with
both current and checkpoint source hashes, preserving the training manifest.
Hashes document file versions; they do not prove historical experiment results.
`best.pt` is selected using **validation Dice**;
the independent test set is evaluated at the end. `history.json` records
loss, Dice, foreground IoU, two-class mIoU, precision/recall, expert utilization,
balance and attention/gate entropy. Empty foreground agreement scores 1.
Expert usage fractions sum to 2 because each position selects two experts.
These are dataset micro metrics, not per-image/per-disease averages.

`--allow-random-init` is for structural/debug runs. It explicitly permits
training without SAM weights. `--model deeplab --pretrained-resnet` retains the
ResNet50 path as an ablation with output stride 8; it is separate from the SAM
model and requires images large enough for ASPP's feature grid.

## Architecture and outputs

```text
Normalized RGB -> SAM patch embedding + sparse ViT blocks
  -> SAM neck -> ASPP rates 1/6/12/18 + GAP
  -> channel projection -> sparse fusion attention + SE
  -> additive projected intermediate-encoder features + normalization
  -> coarse MoE: four experts, spatial top-2, input stride 4
  -> gated shallow-encoder/coarse fusion + coarse confidence + predicted SDF
  -> shared 1x1 embedding + refinement SE
  -> fine MoE: edge/residual/contour experts, spatial top-2, 2x resolution
  -> output head + deformable residual -> logits at original input H/W
```

`SparseMoESAM.forward(x)` returns **logits**; use `model.eval()` and
`model.predict(x)` or `model(x).sigmoid()` for probabilities. Input tensors
must use the normalization in `data.py`. The task-specific semantic decoder
replaces SAM's prompt-driven mask decoder in this training path. Vanilla SAM
builders, `SamPredictor` and `SamAutomaticMaskGenerator` remain available for
the original prompt-driven model; they do not wrap `SparseMoESAM`.

Spatial experts gather only the 3x3 patches required for routed output
positions, with correct padding and dilation, and scatter-add the weighted
outputs. Standard, dilation-6, depthwise and independently initialized experts
serve the coarse stage. Routing uses a two-layer spatial MLP, noise before
softmax and one normalization of selected logits. LayerNorm per routed token
replaces BN/GN in these experts because spatial normalization would require
computing omitted positions. ASPP uses GroupNorm, including its GAP branch,
to support training batches of one.

`CTM`, `merge_tokens` and `TCBlock` are repaired optional legacy utilities in
`token_clustering.py`; padding weights are zero, empty clusters remain finite
and TCBlock now has trainable masked attention/MLP. They are **not** part of
the default model because the paper does not specify token-clustering insertion.

## Reproduction limits and mathematical choices

- Exact global top-k still computes all query/key scores. Chunking reduces
  temporary storage, but QK work remains quadratic. Inference gathers selected
  V tensors, so value aggregation is O(N k D). No O(N k) end-to-end claim is made.
- Training uses Gumbel noise with a straight-through softmax estimator: hard
  top-k forward values, soft gradients for attention scores. The paper leaves
  the estimator unspecified. This uses dense AV during training; it trades
  compute for a continuous gradient. Sparse expert indices remain discrete,
  with gradients through selected logits and a probability-based balance proxy.
- ASPP response masks are applied after convolution, as in Section 4.4. This
  changes features, and by itself does **not** save branch convolution work.
- BCEWithLogits + soft Dice follows Eq. 61. Default `--balance-mode frequency_st`
  uses Table 8/Eq. 62's hard selection-frequency variance in the forward pass.
  The backward pass uses an explicit straight-through proxy based on `top_k *
  mean_prob`; hard selection counts themselves have no derivative. The estimator
  is an engineering choice. `--balance-mode probability` retains Eq. 44/45's
  alternative probability ratio (`M * sum(mean_prob**2)`; uniform minimum 1).
- The normalized attention L1 term in Eq. 63 is **constant** (each row sums
  to 1), so it cannot learn extra sparsity. Its published coefficient .001 is
  retained and the constant is reported. Top-k enforces sparsity. Optional
  `--entropy-weight` maximizes attention/gate entropy via a negative entropy
  term; default 0 avoids silently changing Table 8's objective.
- Section 4.6 specifies weight decay .01 and sparse weight .001; Section 4.7
  also mentions .0001/.002. Defaults follow Section 4.6 and Table 8. Rates,
  weights and phase lengths are configurable for ablation.
- Intermediate ViT tokens are projected/upsampled and added before the coarse
  MoE. Default ViT-B uses zero-based blocks 2/5/8 (depth quartiles), with block 2
  reused in the fine-stage skip. Layer selection and interpolation are engineering
  choices because the paper does not specify a multiscale SAM feature extractor.
  Gated refinement also has its own SE block before the fine MoE.
- Default `--boundary-mode predicted_sdf` thresholds coarse confidence at 0.5
  and computes CPU OpenCV Euclidean signed distances: inside positive, outside
  negative, normalized by the fine-grid diagonal; empty/full masks give -1/+1.
  The SDF is concatenated before a shared 1x1 embedding and fine-stage experts.
  It uses no ground-truth mask and is nondifferentiable. Its prediction source,
  sign, normalization and shared embedding are explicit engineering choices,
  since the paper does not define inference construction. CPU/device transfers
  add overhead. `--boundary-mode none` disables this branch for an ablation.
- Periodic expert weight rescaling, EMA-based adaptive loss scaling,
  batch-wise gating-output normalization and diversity-based early stopping
  are not implemented. Their operational rules are not defined in the paper.
  Custom sparse CUDA kernels are also absent; the paper lists their development
  as future work. No such feature or measured speedup is claimed here.
- No dataset/checkpoint supplied with this repair supports validating 94.2%
  mIoU, a 23.7% FLOPs reduction or actual speedup. Official SAM encoder weights
  can be loaded strictly (position tables adapted explicitly); numerical
  checkpoint-adaptation tests use synthetic encoder weights.

## Regression checks

```powershell
python -m unittest discover -s tests -v
python tests/gpu_smoke.py --full --output runs/gpu_smoke_full.json
```

CPU checks include masked-attention reference equivalence, straight-through
forward/gradient checks, spatial routing, omitted expert execution, dense
convolution equivalence, singleton gradients, ASPP dilation gradients, dynamic
sizes, padding/all-empty clustering, normalization/masks, group splits, native
SAM prompt prediction and training/checkpoint/evaluation/resume. Synthetic
training uses a small SAM ViT for fast CPU verification. The optional GPU
check uses the **real ViT-B/default 1024 encoder and 256-channel decoder** with
random weights, a singleton 256x256 image, AMP and an AdamW step. It verifies
execution and gradients, not segmentation accuracy or inference latency.

Rechecked on 2026-10-13: all 23 CPU tests passed in PyTorch 2.11.0+cu128. The resume
check branches from the same epoch-one checkpoint and compares model tensors
(`atol=2e-6`, `rtol=1e-5`), optimizer steps/moments and scheduler state.
It also checks usable files after an interruption. It does not promise bitwise
reproducibility across independent runs, hardware or library versions. Small
CPU gradient-accumulation differences were observed in independent synthetic
runs; hard routing and SDF thresholds can amplify their prediction differences.
The version-2 full GPU check on RTX 5070 passed: all 274 parameter tensors with
gradients were finite, AdamW states recorded an actual step, and inference was
finite. Published records are the [CPU test log](validation/repair_tests.log),
[full GPU check](validation/gpu_smoke_full.json) and
[model inventory](validation/model_audit_current.json). They describe the
recorded code/configuration; reruns should write to `runs/` or another output
path rather than overwrite these archived records.
