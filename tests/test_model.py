"""Tests for the U-Net model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from lung_seg.model import UNet


def test_unet_output_shape_matches_input():
    model = UNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_unet_returns_raw_logits_not_probabilities():
    model = UNet(base=8)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    # Probabilities would stay between 0 and 1. Raw logits go outside that range.
    assert out.min() < 0 or out.max() > 1
