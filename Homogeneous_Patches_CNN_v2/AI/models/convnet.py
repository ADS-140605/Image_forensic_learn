"""
convnet.py
Source Camera Identification ConvNet architecture from:
"Camera model identification based on forensic traces extracted from homogeneous patches"
Expert Systems With Applications 206 (2022) 117769

Architecture:
  - Block 1: Conv(96, 7x7, s=2) -> BN -> ReLU -> MaxPool(2x2, s=2)
  - Block 2: Conv(64, 5x5, s=1) -> BN -> ReLU -> MaxPool(2x2, s=2)
  - Block 3: Conv(64, 5x5, s=1) -> BN -> ReLU -> MaxPool(2x2, s=2)
  - Block 4: Conv(128, 1x1, s=1) -> BN -> ReLU (No Pooling to keep 4x4 spatial)
  - Flatten to 2048-element feature vector (128 * 4 * 4 = 2048)
  - FC1: 1024 units -> Dropout(0.3) -> ReLU
  - FC2: 200 units -> Dropout(0.3) -> ReLU
  - FC3: num_classes -> Softmax (implicit in CrossEntropy)
"""

import torch
import torch.nn as nn
from torch import Tensor

class CameraConvNet(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        
        # Block 1: Input (B, 3, 128, 128)
        # Conv: floor((128-7)/2)+1 = 61. MaxPool: floor((61-2)/2)+1 = 30.
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: (B, 96, 30, 30)
        # Conv: floor((30-5)/1)+1 = 26. MaxPool: floor((26-2)/2)+1 = 13.
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: (B, 64, 13, 13)
        # Conv: floor((13-5)/1)+1 = 9. MaxPool: floor((9-2)/2)+1 = 4.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: (B, 64, 4, 4)
        # Conv: floor((4-1)/1)+1 = 4. NO MAXPOOL.
        # This results in 4x4 spatial size.
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.flatten_dim = 128 * 4 * 4 # 2048
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(200, num_classes)
        )
        
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
