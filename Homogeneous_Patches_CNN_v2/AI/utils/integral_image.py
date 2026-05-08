"""
integral_image.py
Efficient O(1) patch statistics using summed-area tables.
"""
import numpy as np

class IntegralImage:
    def __init__(self, image: np.ndarray):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        self._image = image.astype(np.float64)
        self._ii = np.zeros((image.shape[0] + 1, image.shape[1] + 1, image.shape[2]), dtype=np.float64)
        self._ii2 = np.zeros_like(self._ii)
        self._build()

    def _build(self):
        img = self._image
        img2 = img ** 2
        self._ii[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
        self._ii2[1:, 1:] = img2.cumsum(axis=0).cumsum(axis=1)

    def region_sum(self, r1, c1, r2, c2):
        ii = self._ii
        return (ii[r2+1, c2+1] - ii[r1, c2+1] - ii[r2+1, c1] + ii[r1, c1])

    def region_sum_sq(self, r1, c1, r2, c2):
        ii2 = self._ii2
        return (ii2[r2+1, c2+1] - ii2[r1, c2+1] - ii2[r2+1, c1] + ii2[r1, c1])

    def region_std(self, r1, c1, r2, c2):
        n = (r2 - r1 + 1) * (c2 - c1 + 1)
        s = self.region_sum(r1, c1, r2, c2)
        s2 = self.region_sum_sq(r1, c1, r2, c2)
        mean = s / n
        var = np.maximum(s2 / n - mean ** 2, 0.0)
        return np.sqrt(var)
