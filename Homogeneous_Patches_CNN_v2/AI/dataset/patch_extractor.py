"""
patch_extractor.py
Logic for extracting and selecting homogeneous patches.
"""
import random
import numpy as np
from AI.utils.integral_image import IntegralImage
from AI.config import PATCH_SIZE, STRIDE, TARGET_PATCHES, SIGMA_MIN, SIGMA_MAX

class PatchExtractor:
    def __init__(self, patch_size=PATCH_SIZE, stride=STRIDE, target_p=TARGET_PATCHES,
                 sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=None):
        self.patch_size = patch_size
        self.stride = stride
        self.target_p = target_p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._rng = random.Random(seed)

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Convert to float [0, 1]
        img_f = image.astype(np.float64) / 255.0
        ii = IntegralImage(img_f)
        
        H, W = img_f.shape[:2]
        ps = self.patch_size
        st = self.stride
        
        homo_patches = []
        nonhomo_patches = [] # (patch, mean_sigma)
        
        for r in range(0, H - ps + 1, st):
            for c in range(0, W - ps + 1, st):
                sigma = ii.region_std(r, c, r + ps - 1, c + ps - 1)
                is_homo = np.all(sigma >= self.sigma_min) and np.all(sigma <= self.sigma_max)
                
                # We only copy the patch if we decide to keep it later to save memory,
                # but for simplicity we can store coords and mean_sigma first.
                mean_sigma = np.mean(sigma)
                if is_homo:
                    homo_patches.append((r, c))
                else:
                    nonhomo_patches.append(((r, c), mean_sigma))

        # Selection logic
        selected_coords = []
        if len(homo_patches) >= self.target_p:
            # Uniform sampling
            indices = np.linspace(0, len(homo_patches) - 1, self.target_p, dtype=int)
            selected_coords = [homo_patches[i] for i in indices]
        else:
            selected_coords = homo_patches
            deficit = self.target_p - len(selected_coords)
            if deficit > 0 and nonhomo_patches:
                # Fill with lowest sigma non-homo
                nonhomo_patches.sort(key=lambda x: x[1])
                selected_coords += [x[0] for x in nonhomo_patches[:deficit]]

        # Actual extraction and pre-processing
        patches = []
        for r, c in selected_coords:
            patch = img_f[r:r+ps, c:c+ps, :].copy()
            # Per-channel mean subtraction
            patch -= np.mean(patch, axis=(0, 1), keepdims=True)
            # Transpose to (C, H, W)
            patch = patch.transpose(2, 0, 1)
            patches.append(patch)
            
        if not patches:
            return np.empty((0, 3, ps, ps), dtype=np.float32)
            
        return np.array(patches, dtype=np.float32)
