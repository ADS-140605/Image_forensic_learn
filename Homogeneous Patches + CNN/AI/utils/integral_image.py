"""
integral_image.py
Efficient O(1) patch statistics using summed-area tables (integral images).
"""
import numpy as np


class IntegralImage:
    """
    Computes and stores the integral image and squared-integral image for an
    input array, enabling O(1) computation of rectangular-region sum, mean,
    variance, and standard deviation.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, C) or (H, W), dtype float32/float64
        Input image normalised to [0, 1].
    """

    def __init__(self, image: np.ndarray):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        self._image = image.astype(np.float64)
        # Summed-area tables: shape (H+1, W+1, C)
        self._ii  = np.zeros((image.shape[0] + 1, image.shape[1] + 1, image.shape[2]),
                             dtype=np.float64)
        self._ii2 = np.zeros_like(self._ii)
        self._build()

    # ── private ──────────────────────────────────────────────────────────────
    def _build(self):
        img  = self._image
        img2 = img ** 2
        # cumulative sum along rows then columns (vectorised over channels)
        self._ii[1:, 1:]  = img.cumsum(axis=0).cumsum(axis=1)
        self._ii2[1:, 1:] = img2.cumsum(axis=0).cumsum(axis=1)

    # ── public API ────────────────────────────────────────────────────────────
    def region_sum(self, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """
        Sum of pixel values in rectangle [(r1,c1), (r2,c2)] **inclusive**.
        Returns array of shape (C,).
        """
        ii = self._ii
        return (ii[r2 + 1, c2 + 1]
                - ii[r1,     c2 + 1]
                - ii[r2 + 1, c1    ]
                + ii[r1,     c1    ])

    def region_sum_sq(self, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """Sum of squared pixel values in rectangle."""
        ii2 = self._ii2
        return (ii2[r2 + 1, c2 + 1]
                - ii2[r1,     c2 + 1]
                - ii2[r2 + 1, c1    ]
                + ii2[r1,     c1    ])

    def region_mean(self, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """Per-channel mean of a rectangular region."""
        n = (r2 - r1 + 1) * (c2 - c1 + 1)
        return self.region_sum(r1, c1, r2, c2) / n

    def region_std(self, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
        """
        Per-channel standard deviation of a rectangular region.
        Uses: Var = E[X²] - (E[X])²
        """
        n    = (r2 - r1 + 1) * (c2 - c1 + 1)
        s    = self.region_sum(r1, c1, r2, c2)
        s2   = self.region_sum_sq(r1, c1, r2, c2)
        mean = s / n
        var  = np.maximum(s2 / n - mean ** 2, 0.0)   # clip rounding errors
        return np.sqrt(var)
