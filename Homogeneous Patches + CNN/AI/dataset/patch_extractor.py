"""
patch_extractor.py
Sliding-window patch extraction with integral-image-accelerated homogeneity
filtering and the paper's patch-selection logic.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from AI.config import PATCH_SIZE, STRIDE, TARGET_PATCHES, SIGMA_MIN, SIGMA_MAX
from AI.utils.integral_image import IntegralImage


# ─── Types ────────────────────────────────────────────────────────────────────
PatchRecord = Tuple[np.ndarray, Tuple[int, int], float]   # (patch, (r,c), mean_sigma)


class PatchExtractor:
    """
    Extracts up to P homogeneous patches from an image following the
    paper's specification:

    1. Slide a (PATCH_SIZE × PATCH_SIZE) window with step STRIDE.
    2. Compute per-channel σ via integral images.
    3. Classify each candidate as homogeneous (SIGMA_MIN ≤ σ ≤ SIGMA_MAX
       for all channels) or non-homogeneous.
    4. If ≥ P homogeneous patches → uniform subsample P.
       If < P homogeneous patches → take all + fill with lowest-σ non-homo.
    5. Per-channel mean subtraction (zero-centering) on each patch.

    Parameters
    ----------
    patch_size : int
    stride     : int
    target_p   : int
    sigma_min  : float   (image normalised to [0, 1])
    sigma_max  : float
    seed       : optional int for reproducibility
    """

    def __init__(
        self,
        patch_size: int  = PATCH_SIZE,
        stride:     int  = STRIDE,
        target_p:   int  = TARGET_PATCHES,
        sigma_min:  float = SIGMA_MIN,
        sigma_max:  float = SIGMA_MAX,
        seed:       Optional[int] = None,
    ):
        self.patch_size = patch_size
        self.stride     = stride
        self.target_p   = target_p
        self.sigma_min  = sigma_min
        self.sigma_max  = sigma_max
        self._rng       = random.Random(seed)
        self._np_rng    = np.random.default_rng(seed)

    # ── public ────────────────────────────────────────────────────────────────
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), dtype uint8 **or** float32 in [0,1].

        Returns
        -------
        patches : np.ndarray, shape (N, 3, patch_size, patch_size) float32,
                  where N ≤ target_p.
        """
        img_f = self._to_float(image)          # (H, W, 3) float64 in [0,1]
        ii    = IntegralImage(img_f)

        homo_patches:     List[PatchRecord] = []
        nonhomo_patches:  List[PatchRecord] = []

        H, W = img_f.shape[:2]
        ps   = self.patch_size
        st   = self.stride

        for r in range(0, H - ps + 1, st):
            for c in range(0, W - ps + 1, st):
                sigma = ii.region_std(r, c, r + ps - 1, c + ps - 1)   # (3,)
                mean_sigma = float(sigma.mean())
                is_homo = bool(np.all(sigma >= self.sigma_min) and
                               np.all(sigma <= self.sigma_max))

                patch = img_f[r:r + ps, c:c + ps, :].copy()            # (ps, ps, 3)
                record: PatchRecord = (patch, (r, c), mean_sigma)

                if is_homo:
                    homo_patches.append(record)
                else:
                    nonhomo_patches.append(record)

        selected = self._select(homo_patches, nonhomo_patches)
        patches  = self._normalize_and_convert(selected)
        return patches                          # (N, 3, ps, ps) float32

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_float(image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    def _select(
        self,
        homo:    List[PatchRecord],
        nonhomo: List[PatchRecord],
    ) -> List[np.ndarray]:
        P = self.target_p

        if len(homo) >= P:
            # Uniform subsample: evenly spaced indices to cover the image
            indices = self._uniform_indices(len(homo), P)
            chosen  = [homo[i][0] for i in indices]
        else:
            chosen = [rec[0] for rec in homo]
            deficit = P - len(chosen)
            if deficit > 0 and nonhomo:
                # Fill with non-homogeneous patches having the lowest σ
                nonhomo_sorted = sorted(nonhomo, key=lambda x: x[2])
                chosen += [rec[0] for rec in nonhomo_sorted[:deficit]]

        return chosen

    @staticmethod
    def _uniform_indices(total: int, count: int) -> List[int]:
        """Return 'count' evenly spaced indices from [0, total)."""
        if count >= total:
            return list(range(total))
        step = total / count
        return [int(round(i * step)) for i in range(count)]

    def _normalize_and_convert(self, patches: List[np.ndarray]) -> np.ndarray:
        """Per-channel mean subtraction then (N,3,H,W) float32."""
        if not patches:
            return np.empty((0, 3, self.patch_size, self.patch_size), dtype=np.float32)

        arr = np.stack(patches, axis=0).astype(np.float32)  # (N, ps, ps, 3)
        # Per-channel mean subtraction across spatial dimensions
        mean = arr.mean(axis=(1, 2), keepdims=True)         # (N, 1, 1, 3)
        arr  = arr - mean
        arr  = arr.transpose(0, 3, 1, 2)                    # (N, 3, ps, ps)
        return arr
