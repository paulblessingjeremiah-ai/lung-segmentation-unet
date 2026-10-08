"""Loss function and overlap metrics for segmentation."""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Binary cross-entropy plus Dice loss, computed on raw logits."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return bce + (1 - dice.mean())


def dice_score(logits, targets, threshold=0.5, eps=1e-7):
    """Mean Dice score over the images in a batch."""
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(dim=1)
    return ((2 * inter + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean().item()


def iou_score(logits, targets, threshold=0.5, eps=1e-7):
    """Mean intersection over union over the images in a batch."""
    preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean().item()
