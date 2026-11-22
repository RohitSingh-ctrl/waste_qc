"""PyTorch datasets over the processed folders (classification folders, semantic masks, YOLO-format boxes)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.vision.common import IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


# ------------------------------------------------------------------------------ classification
def classification_transforms(size: int, train: bool, augment: bool = True) -> T.Compose:
    ops: list = [T.Resize((size, size))]
    if train and augment:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.3, 0.3, 0.2)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class FolderClassification(Dataset):
    """``root/<split>/<class>/*.jpg`` -> ``(tensor, label)``. Class order follows ``class_names``."""

    def __init__(self, root: Path, split: str, class_names: Sequence[str], size: int = 224, augment: bool = True) -> None:
        self.items: List[Tuple[Path, int]] = []
        for i, c in enumerate(class_names):
            d = Path(root) / split / c
            if d.is_dir():
                self.items += [(p, i) for p in _images(d)]
        if not self.items:
            raise FileNotFoundError(f"No images in {Path(root) / split}")
        self.targets = np.array([t for _, t in self.items])
        self.tf = classification_transforms(size, split == "train", augment)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def class_weights(targets: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (counters the rare ``trash`` class)."""
    counts = np.bincount(targets, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------------------ segmentation
class SemanticSegmentation(Dataset):
    """``root/images/<split>`` + ``root/masks/<split>`` (0 = background, k+1 = class k)."""

    def __init__(self, root: Path, split: str, size: int = 128, augment: bool = True) -> None:
        self.imgs = _images(Path(root) / "images" / split)
        self.mask_dir = Path(root) / "masks" / split
        if not self.imgs:
            raise FileNotFoundError(f"No images in {Path(root) / 'images' / split}")
        self.size, self.augment = size, augment and split == "train"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(self.mask_dir / f"{p.stem}.png").resize((self.size, self.size), Image.NEAREST)
        if self.augment and random.random() < 0.5:
            img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        x = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(T.ToTensor()(img))
        return x, torch.from_numpy(np.asarray(mask, dtype=np.int64))


# ------------------------------------------------------------------------------ detection
class YoloFolderDetection(Dataset):
    """Reads the YOLO export (``images/<split>`` + ``labels/<split>``) so paths stay portable (Drive/Colab).

    Returns torchvision-style targets: absolute ``xyxy`` boxes and labels shifted by +1 (0 is background).
    """

    def __init__(self, root: Path, split: str, augment: bool = True) -> None:
        self.imgs = _images(Path(root) / "images" / split)
        self.label_dir = Path(root) / "labels" / split
        if not self.imgs:
            raise FileNotFoundError(f"No images in {Path(root) / 'images' / split}")
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        p = self.imgs[i]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        boxes, labels = [], []
        lf = self.label_dir / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()
                x0, y0, x1, y1 = (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h, \
                                 (float(cx) + float(bw) / 2) * w, (float(cy) + float(bh) / 2) * h
                if x1 - x0 > 1 and y1 - y0 > 1:
                    boxes.append([x0, y0, x1, y1])
                    labels.append(int(c) + 1)
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        x = T.ToTensor()(img)
        if self.augment and random.random() < 0.5:
            x = x.flip(-1)
            b = b[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        return x, {"boxes": b, "labels": torch.tensor(labels, dtype=torch.int64)}


def detection_collate(batch):
    return tuple(zip(*batch))
