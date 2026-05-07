"""
dresden_dataset.py
Dresden Image Database loader.

Directory structure expected:
    <DATASET_DIR>/
        <BrandA>/
            <ModelA1>/
                <DeviceA1a>/   (optional third level)
                    img_001.jpg
                    ...
        <BrandB>/
            ...

Or flat brand/model (no device sub-folder):
    <DATASET_DIR>/
        <BrandA>/
            <ModelA1>/
                img_001.jpg

The loader auto-detects the structure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from AI.config import DATASET_DIR, TARGET_PATCHES, SEED
from AI.dataset.patch_extractor import PatchExtractor
from AI.dataset.transforms import PatchTransform


# ─── Helpers ──────────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _load_image(path: Path) -> np.ndarray:
    """Load image as uint8 (H,W,3) BGR→RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_label_maps(
    dataset_dir: Path = DATASET_DIR,
) -> Tuple[
    Dict[str, int],          # brand_to_idx
    Dict[str, int],          # model_to_idx  (global)
    Dict[str, List[str]],    # brand_to_models
    List[Tuple[Path, str, str, str]],  # [(img_path, brand, model, device)]
]:
    """
    Walk the dataset directory and build label dictionaries.

    Returns
    -------
    brand_to_idx    : {'Nikon': 0, 'Sony': 1, ...}
    model_to_idx    : {'Nikon_D200': 0, 'Sony_DSC': 1, ...}
    brand_to_models : {'Nikon': ['Nikon_D200', ...], ...}
    records         : list of (img_path, brand, model, device)
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            "Please set AI/config.py :: DATASET_DIR to your Dresden DB root."
        )

    records: List[Tuple[Path, str, str, str]] = []
    brands: List[str] = []
    models: List[str] = []

    for brand_dir in sorted(dataset_dir.iterdir()):
        if not brand_dir.is_dir():
            continue
        brand = brand_dir.name

        for model_dir in sorted(brand_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = f"{brand}_{model_dir.name}"

            # Check for device sub-directories or flat images
            sub_items = list(model_dir.iterdir())
            has_device_dirs = any(s.is_dir() for s in sub_items)

            if has_device_dirs:
                for device_dir in sorted(s for s in sub_items if s.is_dir()):
                    device = device_dir.name
                    for img_path in sorted(device_dir.iterdir()):
                        if img_path.suffix.lower() in IMAGE_EXTS:
                            records.append((img_path, brand, model, device))
            else:
                device = "default"
                for img_path in sorted(model_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTS:
                        records.append((img_path, brand, model, device))

            if model not in models:
                models.append(model)

        if brand not in brands:
            brands.append(brand)

    brand_to_idx  = {b: i for i, b in enumerate(brands)}
    model_to_idx  = {m: i for i, m in enumerate(models)}
    brand_to_models: Dict[str, List[str]] = {b: [] for b in brands}
    for _, brand, model, _ in records:
        if model not in brand_to_models[brand]:
            brand_to_models[brand].append(model)

    return brand_to_idx, model_to_idx, brand_to_models, records


# ─── Dataset classes ──────────────────────────────────────────────────────────

class DresdenDataset(Dataset):
    """
    PyTorch Dataset that yields individual patches.

    Parameters
    ----------
    records         : subset of the full records list (from build_label_maps)
    brand_to_idx    : brand name → integer label
    model_to_idx    : model name → integer label
    level           : 'brand' → labels are brand indices
                      'model' → labels are model indices
    brand_filter    : if level=='model', restrict to this brand only
    mode            : 'train' | 'val' | 'test'
    patches_per_img : target number of patches per image (default TARGET_PATCHES)
    """

    def __init__(
        self,
        records:         List[Tuple[Path, str, str, str]],
        brand_to_idx:    Dict[str, int],
        model_to_idx:    Dict[str, int],
        level:           str = "brand",
        brand_filter:    Optional[str] = None,
        mode:            str = "train",
        patches_per_img: int = TARGET_PATCHES,
    ):
        assert level in ("brand", "model")
        self.brand_to_idx = brand_to_idx
        self.model_to_idx = model_to_idx
        self.level        = level
        self.mode         = mode

        if brand_filter:
            records = [(p, br, mo, dev) for p, br, mo, dev in records
                       if br == brand_filter]

        self.records         = records
        self.patches_per_img = patches_per_img
        self.extractor       = PatchExtractor(target_p=patches_per_img, seed=SEED)
        self.transform       = PatchTransform(mode=mode)

        # Pre-compute flat list: (img_path, label)
        self._items: List[Tuple[Path, int]] = []
        for path, brand, model, _ in records:
            label = (brand_to_idx[brand] if level == "brand"
                     else model_to_idx[model])
            self._items.append((path, label))

    def __len__(self) -> int:
        # Each image produces up to patches_per_img samples
        return len(self._items) * self.patches_per_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_idx   = idx // self.patches_per_img
        patch_idx = idx  % self.patches_per_img

        path, label = self._items[img_idx]
        img     = _load_image(path)
        patches = self.extractor.extract(img)          # (N, 3, 128, 128)

        # Wrap around if fewer patches than expected
        patch_idx = patch_idx % max(len(patches), 1)
        patch     = patches[patch_idx]                 # (3, 128, 128)
        tensor    = self.transform(patch)
        return tensor, label


class ImagePatchDataset(Dataset):
    """
    Lightweight inference-only dataset: extracts patches from a single image.
    No labels. Used by HierarchicalPredictor.
    """

    def __init__(self, image: np.ndarray, patches_per_img: int = TARGET_PATCHES):
        self.extractor = PatchExtractor(target_p=patches_per_img)
        self.transform = PatchTransform(mode="test")
        self.patches   = self.extractor.extract(image)  # (N, 3, 128, 128)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.patches[idx])
