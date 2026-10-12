# Sparse-MoE-SAM

Binary semantic segmentation using a SAM image encoder, sparse attention,
ASPP, and a coarse/fine spatial mixture-of-experts decoder.

## Install

```powershell
python -m pip install -r requirements.txt
```

Use matching PyTorch and torchvision builds for your CUDA version, with
`torch.amp.GradScaler` and `torchvision.ops.DeformConv2d` support. SAM components
are included locally; their source and license are recorded in
[SAM_UPSTREAM.json](SAM_UPSTREAM.json) and [LICENSE-SAM](LICENSE-SAM).

## Data

```text
data/
  images/image1.jpg
  masks/image1.png
```

Images and masks must have matching filename stems and original dimensions.
Masks must be single-channel binary images containing only 0/1 or 0/255.
Geometric augmentations are paired, and masks use nearest-neighbor resizing.
Images use ImageNet normalization. At least three independent images or
groups are required for train/validation/test splits.

The first run saves a seeded 70/15/15 split in the output directory. Later runs
reuse it. To keep images from the same plant or capture group together, supply
`--groups-json plant_groups.json`, for example `{"image1.jpg": "plant_001"}`.
Use a new output directory when changing the dataset or seed.

## Train

Download the matching encoder checkpoint from
[Meta's SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).
Weights are not downloaded automatically.

```powershell
python run_code.py --data-root "D:/datasets/plant" --sam-checkpoint "D:/weights/sam_vit_b_01ec64.pth" --batch-size 1 --accumulation-steps 16 --amp
```

Defaults use ViT-B, 256x256 dataset images, a 1024x1024 encoder input, 256 decoder
channels, and top-2 spatial routing. Alternative encoder sizes must be multiples
of 16 and at least 304 to support the ASPP dilation branches.

Use `--allow-random-init` for debugging without pretrained weights.
`--model deeplab` selects the separate ResNet50 ablation; `--pretrained-resnet`
requests torchvision weights. Run `python run_code.py --help` for all options.

## Resume and evaluate

```powershell
python run_code.py --data-root "D:/datasets/plant" --output-dir runs/sam --resume runs/sam/last.pt --batch-size 1 --accumulation-steps 16 --amp
python run_code.py --data-root "D:/datasets/plant" --output-dir runs/sam --eval-only --batch-size 1 --amp
```

Resume with the original model, data, batch, loss, and schedule settings.
Checkpoints store optimizer, scheduler, scaler, random states, configuration,
and the split manifest. The current checkpoint format is version 2; earlier
formats require their matching source version. Evaluation loads `best.pt`
from the output directory unless an explicit checkpoint is supplied with
`--resume`.

`best.pt` is selected by validation Dice, and `last.pt` stores the latest epoch.
The test split is evaluated at the end. Output files include configuration,
training history, source fingerprints, and test metrics. Evaluation and resume
keep separate configuration files. Generated outputs belong in `runs/`, which
is excluded from Git.

## Python API

`SparseMoESAM` in `sam_moe.py` accepts Bx3xHxW RGB tensors normalized with
the mean/std in `data.py`. `forward(x)` returns binary logits at the original
input height and width. `forward(x, return_aux=True)` also returns routing
and auxiliary-loss tensors. Set `model.eval()` before prediction and use
`model.predict(x)` or `model(x).sigmoid()` for probabilities.

The semantic model has its own decoder. Vanilla SAM builders,
`SamPredictor`, and `SamAutomaticMaskGenerator` remain available for the
original prompt-driven SAM API. Token-clustering utilities are optional and
are not inserted into the default segmentation model.

## Development commands

```powershell
python -m unittest discover -s tests -v
python tests/gpu_smoke.py --full --output runs/gpu_smoke_full.json
```

The GPU command requires CUDA and uses randomly initialized weights and
synthetic inputs.
