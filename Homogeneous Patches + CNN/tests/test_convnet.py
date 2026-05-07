"""
test_convnet.py
Unit tests for the CameraConvNet architecture.
"""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from AI.models.convnet import CameraConvNet, build_brand_model, build_model_classifier
from AI.config import PATCH_SIZE, IN_CHANNELS


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH = 4
H = W = PATCH_SIZE   # 128


@pytest.fixture(params=[3, 8, 15])
def num_classes(request):
    return request.param


@pytest.fixture
def dummy_batch():
    torch.manual_seed(42)
    return torch.randn(BATCH, IN_CHANNELS, H, W)


# ── Forward pass ──────────────────────────────────────────────────────────────

class TestForwardPass:
    def test_output_shape(self, dummy_batch, num_classes):
        model  = CameraConvNet(num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (BATCH, num_classes), (
            f"Expected ({BATCH}, {num_classes}), got {out.shape}"
        )

    def test_output_is_logits_not_probabilities(self, dummy_batch):
        """Output should be raw logits (not softmax'd), so values can be negative."""
        model = CameraConvNet(num_classes=5)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        # If softmax had been applied, all values would be in (0, 1)
        has_negative = (out < 0).any().item()
        assert has_negative, "Logits expected; got all non-negative values (softmax?)"

    def test_predict_proba_sums_to_one(self, dummy_batch):
        model = CameraConvNet(num_classes=6)
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(dummy_batch)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), (
            f"Probabilities do not sum to 1: {sums}"
        )

    def test_single_sample(self):
        model = CameraConvNet(num_classes=4)
        model.eval()
        x = torch.randn(1, IN_CHANNELS, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4)


class TestArchitectureProperties:
    def test_num_parameters_positive(self):
        model  = CameraConvNet(num_classes=10)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 0

    def test_different_class_counts(self):
        for nc in [2, 5, 10, 30, 100]:
            m   = CameraConvNet(num_classes=nc)
            x   = torch.randn(2, IN_CHANNELS, H, W)
            out = m(x)
            assert out.shape == (2, nc), f"Failed for num_classes={nc}"

    def test_factory_brand_model(self):
        model = build_brand_model(num_brands=5)
        x     = torch.randn(2, IN_CHANNELS, H, W)
        out   = model(x)
        assert out.shape == (2, 5)

    def test_factory_model_classifier(self):
        model = build_model_classifier(num_models=8)
        x     = torch.randn(2, IN_CHANNELS, H, W)
        out   = model(x)
        assert out.shape == (2, 8)

    def test_training_mode_dropout_active(self):
        """In train mode, two forward passes should differ (dropout stochastic)."""
        model = CameraConvNet(num_classes=5)
        model.train()
        x    = torch.randn(4, IN_CHANNELS, H, W)
        out1 = model(x)
        out2 = model(x)
        assert not torch.equal(out1, out2), (
            "Dropout in train mode should make outputs stochastic"
        )

    def test_eval_mode_deterministic(self):
        """In eval mode, two forward passes should be identical."""
        model = CameraConvNet(num_classes=5)
        model.eval()
        x = torch.randn(4, IN_CHANNELS, H, W)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1, out2), "Eval mode should be deterministic"


class TestGPUCompatibility:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
    def test_forward_on_gpu(self):
        model = CameraConvNet(num_classes=5).cuda()
        x     = torch.randn(4, IN_CHANNELS, H, W).cuda()
        out   = model(x)
        assert out.device.type == "cuda"
        assert out.shape == (4, 5)
