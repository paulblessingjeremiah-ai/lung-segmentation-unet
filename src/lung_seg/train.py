"""Training loop for the lung segmentation U-Net."""

import matplotlib.pyplot as plt
import torch

from .metrics import DiceBCELoss, dice_score, iou_score


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over a loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_dice = total_iou = 0.0
    count = 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = images.size(0)
            total_loss += loss.item() * n
            total_dice += dice_score(logits.detach(), masks) * n
            total_iou += iou_score(logits.detach(), masks) * n
            count += n

    return total_loss / count, total_dice / count, total_iou / count


def train_model(model, train_loader, val_loader, device, num_epochs=40, lr=1e-3,
                checkpoint_path="best_unet.pt"):
    """Train the model and keep the checkpoint with the best validation Dice."""
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        tr_loss, tr_dice, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_dice)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)

        line = (f"Epoch {epoch+1}/{num_epochs} | lr {optimizer.param_groups[0]['lr']:.0e} | "
                f"Train loss {tr_loss:.4f}, Dice {tr_dice:.4f} | "
                f"Val loss {va_loss:.4f}, Dice {va_dice:.4f}, IoU {va_iou:.4f}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), checkpoint_path)
            line += "  <- best"
        print(line)

    return history, best_val_dice


def plot_training_curves(history, save_path="training_curves.png"):
    """Plot loss and Dice for the train and validation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_dice"], label="train")
    axes[1].plot(history["val_dice"], label="validation")
    axes[1].set_title("Dice score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
