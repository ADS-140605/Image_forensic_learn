"""
transforms.py
Dataset-level transforms applied per-patch before feeding into the network.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms as T


class PatchTransform:
    """
    Lightweight wrapper around torchvision transforms for train/val modes.
    The raw patches coming from PatchExtractor are already:
      - float32 (N, 3, 128, 128)
      - per-channel mean subtracted (zero-centred)
    So we only need to convert to tensor (and optionally augment for training).
    """

    def __init__(self, mode: str = "val"):
        assert mode in ("train", "val", "test")
        self.mode = mode

        # Light augmentations for training (no colour jitter — we want
        # fingerprint traces, not scene content, so keep radiometric changes minimal)
        if mode == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])
        else:
            self.transform = None

    def __call__(self, patch: np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        patch : np.ndarray, shape (3, H, W), float32, already zero-centred.

        Returns
        -------
        torch.Tensor, shape (3, H, W), float32.
        """
        t = torch.from_numpy(patch)            # (3, H, W)
        if self.transform is not None:
            t = self.transform(t)
        return t
