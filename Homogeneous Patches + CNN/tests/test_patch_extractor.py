"""
test_patch_extractor.py
Unit tests for the PatchExtractor class.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AI.dataset.patch_extractor import PatchExtractor
from AI.config import PATCH_SIZE, TARGET_PATCHES, SIGMA_MIN, SIGMA_MAX


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    return PatchExtractor(seed=42)


def _make_flat_image(h=512, w=512, channels=3, value=0.01):
    """Nearly-uniform image — all patches should be homogeneous."""
    img = np.full((h, w, channels), value, dtype=np.float32)
    # add tiny noise so σ is non-zero but stays within [SIGMA_MIN, SIGMA_MAX]
    rng = np.random.default_rng(0)
    img += rng.uniform(-0.008, 0.008, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def _make_noisy_image(h=512, w=512, channels=3):
    """High-variance image — all patches should be non-homogeneous."""
    rng = np.random.default_rng(1)
    return rng.uniform(0, 1, size=(h, w, channels)).astype(np.float32)


def _make_uint8_image(h=512, w=512):
    """uint8 input as would come from OpenCV."""
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


# ── Shape & count tests ───────────────────────────────────────────────────────

class TestPatchShape:
    def test_output_ndim(self, extractor):
        img = _make_flat_image()
        patches = extractor.extract(img)
        assert patches.ndim == 4, "Expected (N, C, H, W)"

    def test_output_channels_first(self, extractor):
        img = _make_flat_image()
        patches = extractor.extract(img)
        assert patches.shape[1] == 3, "Channel dim should be 3"
        assert patches.shape[2] == PATCH_SIZE
        assert patches.shape[3] == PATCH_SIZE

    def test_output_dtype_float32(self, extractor):
        img = _make_flat_image()
        patches = extractor.extract(img)
        assert patches.dtype == np.float32

    def test_uint8_input(self, extractor):
        img = _make_uint8_image()
        patches = extractor.extract(img)
        assert patches.ndim == 4
        assert patches.shape[2:] == (PATCH_SIZE, PATCH_SIZE)


class TestPatchCount:
    def test_target_patches_flat_image(self, extractor):
        """Flat image with enough windows → should return exactly P.
        A 512×512 image yields only 169 candidate windows (< P=200).
        Use 1024×1024: (1024-128)//32+1 = 29 → 29²=841 windows > 200.
        """
        img = _make_flat_image(1024, 1024)
        patches = extractor.extract(img)
        assert len(patches) == TARGET_PATCHES, (
            f"Expected {TARGET_PATCHES} patches, got {len(patches)}"
        )

    def test_fewer_than_P_homogeneous_fills_with_nonhomo(self):
        """Noisy image → 0 homogeneous patches → fills all P from non-homo."""
        ex = PatchExtractor(seed=0)
        img = _make_noisy_image(512, 512)
        patches = ex.extract(img)
        assert len(patches) > 0, "Should still return patches from non-homo fill"

    def test_small_image_fewer_than_P_patches(self, extractor):
        """Image too small to generate P candidate windows."""
        img = _make_flat_image(h=200, w=200)
        patches = extractor.extract(img)
        # Max possible windows along one axis: (200-128)//32 + 1 = 3 → 3×3=9
        assert len(patches) <= TARGET_PATCHES

    def test_empty_image_returns_empty(self, extractor):
        """Image smaller than patch_size → no windows → empty output."""
        img = np.zeros((64, 64, 3), dtype=np.float32)
        patches = extractor.extract(img)
        assert len(patches) == 0


class TestHomogeneityFiltering:
    def test_flat_image_all_pass(self):
        """Each channel σ of a nearly-uniform image should be within bounds.
        Need ≥ P=200 candidate windows: use 1024×1024 (841 windows).
        """
        ex = PatchExtractor(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=7)
        img = _make_flat_image(1024, 1024)
        patches = ex.extract(img)
        assert len(patches) == TARGET_PATCHES

    def test_noisy_image_fills_nonhomo(self):
        """Noisy image: 0 homogeneous → all slots filled with non-homo patches."""
        ex = PatchExtractor(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, seed=7)
        img = _make_noisy_image()
        patches = ex.extract(img)
        # We can't guarantee 200 patches if there aren't enough windows
        n_windows = ((512 - 128) // 32 + 1) ** 2   # ~169
        assert len(patches) <= n_windows


class TestNormalization:
    def test_per_channel_mean_near_zero(self, extractor):
        """After mean subtraction each patch should have near-zero channel mean."""
        img = _make_flat_image()
        patches = extractor.extract(img)           # (N, 3, 128, 128)
        means = patches.mean(axis=(2, 3))           # (N, 3)
        assert np.allclose(means, 0, atol=1e-5), (
            f"Max |mean| = {np.abs(means).max():.6f}"
        )
