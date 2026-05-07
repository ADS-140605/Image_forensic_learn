"""
hierarchical.py
Orchestrates the two-level hierarchical training:
  Level 1 → Brand classifier (all data)
  Level 2 → One per-brand Model classifier
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from AI.config import (
    BATCH_SIZE, CHECKPOINT_DIR, NUM_WORKERS, PIN_MEMORY, SEED, MAX_EPOCHS
)
from AI.dataset.dresden_dataset import (
    DresdenDataset, ImagePatchDataset, build_label_maps
)
from AI.models.convnet import build_brand_model, build_model_classifier
from AI.training.distribution import HierarchicalSampler
from AI.training.trainer import Trainer


class HierarchicalTrainer:
    """
    End-to-end hierarchical training manager.

    Usage
    -----
    ht = HierarchicalTrainer()
    ht.train_all()            # trains L1 brand + all L2 model classifiers
    ht.save_metadata()        # writes label maps to checkpoints/metadata.json
    """

    VAL_SPLIT = 0.15          # 15 % of images used for validation

    def __init__(self, dataset_dir: Optional[Path] = None, device: Optional[torch.device] = None):
        self.device = device

        print("[HierarchicalTrainer] Building label maps …")
        (
            self.brand_to_idx,
            self.model_to_idx,
            self.brand_to_models,
            self.records,
        ) = build_label_maps(dataset_dir)

        self.idx_to_brand = {v: k for k, v in self.brand_to_idx.items()}
        self.idx_to_model = {v: k for k, v in self.model_to_idx.items()}

        print(f"  Brands  : {len(self.brand_to_idx)}")
        print(f"  Models  : {len(self.model_to_idx)}")
        print(f"  Images  : {len(self.records)}")

    # ── public ────────────────────────────────────────────────────────────────
    def train_all(self, max_epochs: int = MAX_EPOCHS):
        """Train Level-1 brand classifier, then all Level-2 model classifiers."""
        print("\n=== Level 1 — Brand Classifier ===")
        self.train_brand_level(max_epochs)

        print("\n=== Level 2 — Per-Brand Model Classifiers ===")
        for brand in self.brand_to_idx:
            n_models = len(self.brand_to_models[brand])
            if n_models < 2:
                print(f"  Skipping '{brand}' (only {n_models} model).")
                continue
            print(f"\n--- Brand: {brand} ({n_models} models) ---")
            self.train_model_level(brand, max_epochs)

        self.save_metadata()
        print("\n[HierarchicalTrainer] Training complete. Metadata saved.")

    def train_brand_level(self, max_epochs: int = MAX_EPOCHS):
        train_recs, val_recs = self._split_records(self.records)

        train_ds = DresdenDataset(
            train_recs, self.brand_to_idx, self.model_to_idx,
            level="brand", mode="train",
        )
        val_ds = DresdenDataset(
            val_recs, self.brand_to_idx, self.model_to_idx,
            level="brand", mode="val",
        )
        sampler = HierarchicalSampler(
            train_recs, level="brand"
        ).sampler

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        model = build_brand_model(num_brands=len(self.brand_to_idx))
        trainer = Trainer(model, name="brand_classifier", device=self.device)
        trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    def train_model_level(self, brand: str, max_epochs: int = MAX_EPOCHS):
        brand_recs = [(p, b, m, d) for p, b, m, d in self.records if b == brand]
        train_recs, val_recs = self._split_records(brand_recs)

        # Build per-brand model index (local 0-based)
        brand_models = self.brand_to_models[brand]
        local_model_to_idx = {m: i for i, m in enumerate(brand_models)}

        train_ds = DresdenDataset(
            train_recs, self.brand_to_idx, local_model_to_idx,
            level="model", brand_filter=brand, mode="train",
        )
        val_ds = DresdenDataset(
            val_recs, self.brand_to_idx, local_model_to_idx,
            level="model", brand_filter=brand, mode="val",
        )
        sampler = HierarchicalSampler(
            train_recs, level="model", brand_filter=brand
        ).sampler

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        )

        model = build_model_classifier(num_models=len(brand_models))
        safe_name = brand.replace(" ", "_").replace("/", "-")
        trainer = Trainer(model, name=f"model_{safe_name}", device=self.device)
        trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    def save_metadata(self):
        """Persist label maps and brand→models mapping for inference."""
        meta = {
            "brand_to_idx":    self.brand_to_idx,
            "model_to_idx":    self.model_to_idx,
            "brand_to_models": {b: list(ms) for b, ms in self.brand_to_models.items()},
        }
        out = CHECKPOINT_DIR / "metadata.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"[HierarchicalTrainer] Metadata → {out}")

    # ── private ───────────────────────────────────────────────────────────────
    def _split_records(
        self,
        records: List[Tuple],
        val_frac: float = VAL_SPLIT,
    ) -> Tuple[List, List]:
        """Stratified split by brand to avoid class imbalance in val set."""
        import random
        rng = random.Random(SEED)

        by_brand: Dict[str, List] = {}
        for rec in records:
            brand = rec[1]
            by_brand.setdefault(brand, []).append(rec)

        train_recs, val_recs = [], []
        for brand_list in by_brand.values():
            rng.shuffle(brand_list)
            n_val = max(1, int(len(brand_list) * val_frac))
            val_recs.extend(brand_list[:n_val])
            train_recs.extend(brand_list[n_val:])

        return train_recs, val_recs
