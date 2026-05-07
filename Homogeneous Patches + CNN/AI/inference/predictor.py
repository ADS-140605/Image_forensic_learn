"""
predictor.py
Hierarchical majority-vote inference.

Workflow:
  1. Extract P patches from the test image.
  2. Run all patches through the Level-1 brand classifier.
  3. Majority-vote → predicted brand.
  4. Run the same patches through the matching Level-2 model classifier.
  5. Majority-vote → predicted model.

Returns both the brand and model name, plus vote distributions.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from AI.config import CHECKPOINT_DIR, NUM_WORKERS, PIN_MEMORY, TARGET_PATCHES
from AI.dataset.dresden_dataset import ImagePatchDataset
from AI.models.convnet import CameraConvNet


class HierarchicalPredictor:
    """
    Loads trained checkpoints and performs hierarchical image-level prediction.

    Parameters
    ----------
    checkpoint_dir : Path to the directory containing brand_classifier/best.pt,
                     model_<Brand>/best.pt, and metadata.json.
    device         : torch.device (auto-detected if None).
    """

    def __init__(
        self,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or self._auto_device()

        meta_path = self.checkpoint_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found at {meta_path}. "
                "Run HierarchicalTrainer.train_all() first."
            )
        meta = json.loads(meta_path.read_text())
        self.brand_to_idx:    Dict[str, int]        = meta["brand_to_idx"]
        self.model_to_idx:    Dict[str, int]        = meta["model_to_idx"]
        self.brand_to_models: Dict[str, List[str]]  = meta["brand_to_models"]
        self.idx_to_brand = {v: k for k, v in self.brand_to_idx.items()}

        # Load Level-1 brand model
        self.brand_model = self._load_model(
            self.checkpoint_dir / "brand_classifier" / "best.pt",
            num_classes=len(self.brand_to_idx),
        )

        # Load all Level-2 model classifiers (lazy: keyed by brand name)
        self._model_classifiers: Dict[str, CameraConvNet] = {}
        self._local_idx_to_model: Dict[str, Dict[int, str]] = {}
        for brand, models in self.brand_to_models.items():
            safe = brand.replace(" ", "_").replace("/", "-")
            ckpt = self.checkpoint_dir / f"model_{safe}" / "best.pt"
            if ckpt.exists() and len(models) >= 2:
                clf = self._load_model(ckpt, num_classes=len(models))
                self._model_classifiers[brand] = clf
                self._local_idx_to_model[brand] = {i: m for i, m in enumerate(models)}

    # ── public ────────────────────────────────────────────────────────────────
    def predict(
        self,
        image: np.ndarray,
        patches_per_img: int = TARGET_PATCHES,
    ) -> Dict:
        """
        Classify a single image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), uint8 or float32 in [0,1].

        Returns
        -------
        dict with keys:
          brand         : str
          model         : str  (or None if Level-2 not available)
          brand_votes   : dict {brand_name: count}
          model_votes   : dict {model_name: count}  (or {})
          brand_probs   : list[float] (per-brand mean softmax)
          model_probs   : list[float] (per-model mean softmax, or [])
        """
        ds     = ImagePatchDataset(image, patches_per_img=patches_per_img)
        loader = DataLoader(
            ds, batch_size=64, shuffle=False,
            num_workers=0,        # single image → no multiprocessing overhead
            pin_memory=PIN_MEMORY,
        )

        # ── Level 1: brand ────────────────────────────────────────────────────
        brand_preds, brand_probs_all = self._run_model(self.brand_model, loader)
        brand_votes  = Counter(self.idx_to_brand[p] for p in brand_preds)
        pred_brand   = brand_votes.most_common(1)[0][0]

        brand_probs_mean = np.array(brand_probs_all).mean(axis=0).tolist()

        # ── Level 2: model ────────────────────────────────────────────────────
        model_votes: Counter = Counter()
        model_probs_mean: List[float] = []
        pred_model: Optional[str] = None

        if pred_brand in self._model_classifiers:
            clf = self._model_classifiers[pred_brand]
            local_map = self._local_idx_to_model[pred_brand]

            model_preds, model_probs_all = self._run_model(clf, loader)
            model_votes      = Counter(local_map[p] for p in model_preds)
            pred_model       = model_votes.most_common(1)[0][0]
            model_probs_mean = np.array(model_probs_all).mean(axis=0).tolist()
        elif pred_brand in self.brand_to_models:
            models = self.brand_to_models[pred_brand]
            if models:
                pred_model = models[0]   # only one model in this brand

        return {
            "brand":       pred_brand,
            "model":       pred_model,
            "brand_votes": dict(brand_votes),
            "model_votes": dict(model_votes),
            "brand_probs": brand_probs_mean,
            "model_probs": model_probs_mean,
            "num_patches": len(ds),
        }

    # ── private ───────────────────────────────────────────────────────────────
    def _run_model(
        self,
        model: CameraConvNet,
        loader: DataLoader,
    ) -> Tuple[List[int], List[List[float]]]:
        model.eval()
        all_preds:  List[int]        = []
        all_probs:  List[List[float]] = []

        with torch.no_grad():
            for patches in loader:
                patches = patches.to(self.device, non_blocking=True)
                probs   = model.predict_proba(patches)          # (B, C)
                preds   = probs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_probs.extend(probs.cpu().tolist())

        return all_preds, all_probs

    def _load_model(self, path: Path, num_classes: int) -> CameraConvNet:
        from AI.models.convnet import CameraConvNet
        model = CameraConvNet(num_classes=num_classes)
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state["model_state"])
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
