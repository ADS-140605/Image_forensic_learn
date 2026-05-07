"""
convnet.py
7-Block Convolutional Neural Network for source camera identification.

Architecture (from the paper):
  Block 1 : Conv(96, 7×7, stride=2, valid) → BN → ReLU → MaxPool(2×2, s=2)
  Block 2 : Conv(64, 5×5, stride=1, valid) → BN → ReLU → MaxPool(2×2, s=2)
  Block 3 : Conv(64, 5×5, stride=1, valid) → BN → ReLU → MaxPool(2×2, s=2)
  Block 4 : Conv(128,1×1, stride=1, valid) → BN → ReLU → MaxPool(2×2, s=2)
  Block 5 : Flatten → FC(2048→1024) → Dropout(0.3) → ReLU
  Block 6 : FC(1024→200) → Dropout(0.3) → ReLU
  Block 7 : FC(200→num_classes) → Softmax  [Softmax not included in model;
             handled by CrossEntropyLoss during training]

Input  : (B, 3, 128, 128)
Output : (B, num_classes) logits
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from AI.config import IN_CHANNELS, FC1_UNITS, FC2_UNITS, DROPOUT_RATE


# ─── Building blocks ──────────────────────────────────────────────────────────

def _conv_block(
    in_ch:  int,
    out_ch: int,
    kernel: int,
    stride: int = 1,
) -> nn.Sequential:
    """Conv → BN → ReLU → MaxPool(2×2, stride=2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=0),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


# ─── Main model ───────────────────────────────────────────────────────────────

class CameraConvNet(nn.Module):
    """
    Parameters
    ----------
    num_classes : int
        Number of output classes (brands for L1, models for L2).
    in_channels : int
        Number of input channels (default 3 for RGB).
    """

    def __init__(self, num_classes: int, in_channels: int = IN_CHANNELS):
        super().__init__()
        self.num_classes = num_classes

        # ── Convolutional blocks ──────────────────────────────────────────────
        # Input: (B, 3, 128, 128)
        # After B1: floor((128-7)/2)+1 = 61 → MaxPool → 30
        self.block1 = _conv_block(in_channels, 96, kernel=7, stride=2)

        # After B2: floor((30-5)/1)+1 = 26 → MaxPool → 13
        self.block2 = _conv_block(96, 64, kernel=5, stride=1)

        # After B3: floor((13-5)/1)+1 = 9  → MaxPool → 4
        self.block3 = _conv_block(64, 64, kernel=5, stride=1)

        # After B4: floor((4-1)/1)+1  = 4  → MaxPool → 2
        self.block4 = _conv_block(64, 128, kernel=1, stride=1)

        # Spatial size after B4: 2×2, channels=128 → 128*2*2 = 512
        self._flat_dim = 128 * 2 * 2   # = 512

        # ── Fully-connected blocks ────────────────────────────────────────────
        # Block 5: flatten → FC(512→1024) + Dropout
        self.block5 = nn.Sequential(
            nn.Linear(self._flat_dim, FC1_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
        )

        # Block 6: FC(1024→200) + Dropout
        self.block6 = nn.Sequential(
            nn.Linear(FC1_UNITS, FC2_UNITS),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
        )

        # Block 7: output head
        self.block7 = nn.Linear(FC2_UNITS, num_classes)

        self._init_weights()

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1)   # (B, flat_dim)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)           # logits (B, num_classes)
        return x

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return softmax probabilities (inference helper)."""
        return torch.softmax(self.forward(x), dim=-1)

    # ── initialisation ────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)


# ─── Factory functions ────────────────────────────────────────────────────────

def build_brand_model(num_brands: int) -> CameraConvNet:
    """Instantiate the Level-1 brand classifier."""
    return CameraConvNet(num_classes=num_brands)


def build_model_classifier(num_models: int) -> CameraConvNet:
    """Instantiate a Level-2 model classifier for a specific brand."""
    return CameraConvNet(num_classes=num_models)
