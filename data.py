"""Binary masks, paired augmentation, reproducible 70/15/15 manifests."""
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF, InterpolationMode
from PIL import Image

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_pairs(root):
    root = Path(root)
    images, masks = root / "images", root / "masks"
    mask_files = {}
    for path in sorted(masks.glob("*")):
        if path.suffix.lower() in EXTENSIONS:
            if path.stem in mask_files:
                raise ValueError(f"Ambiguous mask stem: {path.stem}")
            mask_files[path.stem] = path
    pairs, stems = [], set()
    for image in sorted(images.glob("*")):
        if image.suffix.lower() not in EXTENSIONS:
            continue
        if image.stem in stems:
            raise ValueError(f"Ambiguous image stem: {image.stem}")
        stems.add(image.stem)
        if image.stem not in mask_files:
            raise FileNotFoundError(f"Missing mask for {image.name}")
        pairs.append({"image": image.relative_to(root).as_posix(),
                      "mask": mask_files[image.stem].relative_to(root).as_posix()})
    if not pairs:
        raise ValueError(f"No paired images in {root}")
    return pairs


def prepare_splits(root, manifest_path, seed=42, groups_path=None):
    """Reuse exact manifest; optional image filename -> plant/group mapping."""
    path = Path(manifest_path)
    pairs = discover_pairs(root)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows = [pair for split in ("train", "val", "test") for pair in manifest[split]]
        canonical = lambda pair: (pair["image"], pair["mask"])
        if len(rows) != len(pairs) or len(set(map(canonical, rows))) != len(rows) or set(map(canonical, rows)) != set(map(canonical, pairs)):
            raise ValueError("Manifest differs from current dataset; use a new output directory")
        if manifest["seed"] != seed:
            raise ValueError("Existing manifest seed differs; use its seed or a new output directory")
        if groups_path and json.loads(Path(groups_path).read_text(encoding="utf-8")) != manifest.get("groups"):
            raise ValueError("Group metadata differs from the existing manifest")
        if any(not manifest[s] for s in ("train", "val", "test")):
            raise ValueError("Every split must contain samples")
        return manifest
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8")) if groups_path else None
    buckets = {}
    for pair in pairs:
        name = Path(pair["image"]).name
        if groups is not None and name not in groups:
            raise ValueError(f"Missing group for {name}")
        group = str(groups[name]) if groups is not None else name
        buckets.setdefault(group, []).append(pair)
    group_names = sorted(buckets)
    if len(group_names) < 3:
        raise ValueError("At least 3 independent images/groups are needed for train/val/test")
    random.Random(seed).shuffle(group_names)
    n = len(group_names)
    train_n = min(n - 2, max(1, int(0.70 * n)))
    val_n = min(n - train_n - 1, max(1, int(0.15 * n)))
    manifest = {"seed": seed, "ratios": [0.7, 0.15, 0.15], "grouped": groups is not None}
    for split, names in zip(("train", "val", "test"),
                            (group_names[:train_n], group_names[train_n:train_n + val_n], group_names[train_n + val_n:])):
        manifest[split] = [pair for name in names for pair in buckets[name]]
    if groups is not None:
        manifest["groups"] = groups
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), train=True, pairs=None, seed=42,
                 augment=None):
        self.root = Path(root_dir)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.augment = train if augment is None else augment
        if pairs is None:
            all_pairs = discover_pairs(self.root)
            if len(all_pairs) < 3:
                raise ValueError("At least 3 images are needed for train/val/test")
            random.Random(seed).shuffle(all_pairs)
            count = len(all_pairs)
            split = min(count - 2, max(1, int(count * .7)))
            val_count = min(count - split - 1, max(1, int(count * .15)))
            pairs = all_pairs[:split] if train else all_pairs[split:split + val_count]
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("Empty dataset split")
        self.jitter = ColorJitter(brightness=.2, contrast=.2, saturation=.1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        with Image.open(self.root / pair["image"]) as source:
            image = source.convert("RGB")
        with Image.open(self.root / pair["mask"]) as source:
            # Preserve 0/1 palette values instead of converting them to colors.
            array = np.asarray(source)
            if array.ndim != 2 or not (np.isin(array, [0, 1]).all() or np.isin(array, [0, 255]).all()):
                raise ValueError(f"Expected a 2D binary 0/1 or 0/255 mask: {pair['mask']}")
            mask = Image.fromarray((array > 0).astype(np.uint8) * 255)
        if image.size != mask.size:
            raise ValueError(f"Image/mask dimensions differ: {pair['image']}")
        image = TF.resize(image, self.image_size, InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, InterpolationMode.NEAREST)
        if self.augment:
            for flip in (TF.hflip, TF.vflip):
                if torch.rand(()) < .5:
                    image, mask = flip(image), flip(mask)
            angle = float(torch.empty(()).uniform_(-15, 15))
            scale = float(torch.empty(()).uniform_(.8, 1.2))
            image = TF.affine(image, angle, [0, 0], scale, [0., 0.], InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle, [0, 0], scale, [0., 0.], InterpolationMode.NEAREST)
            image = self.jitter(image)
        image = TF.to_tensor(image)
        if self.augment and torch.rand(()) < .3:
            image = (image + torch.randn_like(image) * .01).clamp(0, 1)
        return TF.normalize(image, MEAN, STD), TF.to_tensor(mask)
