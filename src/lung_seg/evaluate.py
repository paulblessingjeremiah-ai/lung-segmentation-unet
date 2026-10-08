"""Test-set scoring and prediction plots."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import dice_score, iou_score


def score_dataset(model, dataset, device, threshold=0.5):
    """Score every image in a dataset. Returns per-image results and predicted masks."""
    model.eval()
    results = []
    predictions = []

    with torch.no_grad():
        for i in range(len(dataset)):
            image, mask = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            mask_batch = mask.unsqueeze(0).to(device)
            results.append({
                "source": dataset.pairs[i][2],
                "path": dataset.pairs[i][0],
                "dice": dice_score(logits, mask_batch, threshold),
                "iou": iou_score(logits, mask_batch, threshold),
            })
            pred = (torch.sigmoid(logits) > threshold).float().cpu().squeeze().numpy()
            predictions.append(pred)

    return results, predictions


def summarize(results):
    """Print Dice and IoU overall and for each source dataset."""
    groups = {"all": results}
    for source in sorted({r["source"] for r in results}):
        groups[source] = [r for r in results if r["source"] == source]

    for name, rows in groups.items():
        dice = np.array([r["dice"] for r in rows])
        iou = np.array([r["iou"] for r in rows])
        print(f"{name:<11} n={len(rows):<4} "
              f"Dice mean {dice.mean():.4f} std {dice.std():.4f} median {np.median(dice):.4f} | "
              f"IoU mean {iou.mean():.4f} | worst Dice {dice.min():.4f}")


def plot_predictions(dataset, results, predictions, save_path="test_predictions.png",
                     n_worst=3, n_random=3, seed=0):
    """Plot the worst cases and some random cases. Green is the true outline, red the prediction."""
    dice = np.array([r["dice"] for r in results])
    worst = list(np.argsort(dice)[:n_worst])

    rng = random.Random(seed)
    others = [i for i in range(len(results)) if i not in worst]
    chosen_random = rng.sample(others, n_random)

    chosen = worst + chosen_random
    labels = ["worst"] * len(worst) + ["random"] * len(chosen_random)

    cols = 3
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.5 * rows))
    for ax, idx, label in zip(axes.ravel(), chosen, labels):
        image, mask = dataset[idx]
        ax.imshow(image.squeeze().numpy(), cmap="gray")
        ax.contour(mask.squeeze().numpy(), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(predictions[idx], levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(f"{label} | {results[idx]['source']} | Dice {results[idx]['dice']:.3f}")
        ax.axis("off")

    fig.suptitle("Green = true lung outline, Red = model prediction")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
