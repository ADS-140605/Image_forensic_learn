"""
distribution.py
Top-down hierarchical patch distribution to ensure equal representation of
brands, models, and devices during training.

Strategy (from the paper):
  - At the brand level, all brands should contribute equally.
  - Within each brand, all models should contribute equally.
  - Within each model, all devices should contribute equally.

Implementation: a custom WeightedRandomSampler-compatible weight vector.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_sample_weights(
    records: List[Tuple[Path, str, str, str]],
    level: str = "brand",
    brand_filter: Optional[str] = None,
) -> List[float]:
    """
    Compute per-image sample weights for hierarchical balanced sampling.

    The weight of each image is inversely proportional to the product of:
      (number of brands) × (number of models in that brand) × (number of devices in that model)

    This ensures every brand, model, and device is equally represented in
    expectation under weighted random sampling.

    Parameters
    ----------
    records      : full list from build_label_maps()
    level        : 'brand' or 'model'
    brand_filter : restrict to one brand when level=='model'

    Returns
    -------
    weights : list of floats, one per image record (same order as records)
    """
    if brand_filter:
        records = [(p, b, m, d) for p, b, m, d in records if b == brand_filter]

    # Count unique entities at each level
    brand_models:  Dict[str, set] = defaultdict(set)
    model_devices: Dict[str, set] = defaultdict(set)
    brand_counts:  Dict[str, int] = defaultdict(int)

    for _, brand, model, device in records:
        brand_models[brand].add(model)
        model_devices[model].add(device)
        brand_counts[brand] += 1

    num_brands = len(brand_models)
    weights: List[float] = []

    for _, brand, model, device in records:
        n_models  = len(brand_models[brand])
        n_devices = len(model_devices[model])
        # Weight = 1 / (brands × models_per_brand × devices_per_model)
        w = 1.0 / (num_brands * n_models * n_devices)
        weights.append(w)

    return weights


class HierarchicalSampler:
    """
    Wraps WeightedRandomSampler with hierarchical weights.

    Usage
    -----
    sampler = HierarchicalSampler(records, level='brand')
    loader  = DataLoader(dataset, batch_size=64, sampler=sampler.sampler)
    """

    def __init__(
        self,
        records: List[Tuple[Path, str, str, str]],
        level:   str = "brand",
        brand_filter: Optional[str] = None,
        patches_per_img: int = 200,
        replacement: bool = True,
    ):
        weights_per_image = compute_sample_weights(records, level, brand_filter)

        # Expand weights: each image has patches_per_img samples in the dataset
        expanded: List[float] = []
        for w in weights_per_image:
            expanded.extend([w] * patches_per_img)

        num_samples = len(expanded)
        self.sampler = WeightedRandomSampler(
            weights=expanded,
            num_samples=num_samples,
            replacement=replacement,
        )
