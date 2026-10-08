"""Tests for the loss and overlap metrics."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from lung_seg.metrics import DiceBCELoss, dice_score, iou_score


def make_targets():
    targets = torch.zeros(2, 1, 16, 16)
    targets[:, :, 4:12, 4:12] = 1
    return targets


def test_perfect_prediction_scores_one():
    targets = make_targets()
    logits = targets * 20 - 10
    assert dice_score(logits, targets) == pytest.approx(1.0)
    assert iou_score(logits, targets) == pytest.approx(1.0)


def test_completely_wrong_prediction_scores_near_zero():
    targets = make_targets()
    logits = (1 - targets) * 20 - 10
    assert dice_score(logits, targets) < 0.01
    assert iou_score(logits, targets) < 0.01


def test_loss_is_lower_for_a_better_prediction():
    targets = make_targets()
    good = targets * 20 - 10
    bad = (1 - targets) * 20 - 10
    loss_fn = DiceBCELoss()
    assert loss_fn(good, targets) < loss_fn(bad, targets)
