import os
import random

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATASETS = ["Montgomery", "Shenzhen"]


def find_pairs(root, datasets=DATASETS):
    """Return (image_path, mask_path, source) for every image that has a mask."""
    pairs = []
    for source in datasets:
        img_dir = os.path.join(root, source, "img")
        mask_dir = os.path.join(root, source, "mask")
        for name in sorted(os.listdir(img_dir)):
            mask_path = os.path.join(mask_dir, name)
            if os.path.exists(mask_path):
                pairs.append((os.path.join(img_dir, name), mask_path, source))
    return pairs


def split_pairs(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    """Split into train, val and test, keeping the Montgomery/Shenzhen ratio in each."""
    sources = [p[2] for p in pairs]
    train_val, test = train_test_split(
        pairs, test_size=test_frac, random_state=seed, stratify=sources
    )
    tv_sources = [p[2] for p in train_val]
    rel_val = val_frac / (1 - test_frac)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=tv_sources
    )
    return train, val, test


def augment(image, mask):
    """Same small rotation and scale on image and mask, brightness and contrast on image only."""
    size = image.shape[0]
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    contrast = random.uniform(0.85, 1.15)
    brightness = random.uniform(-0.1, 0.1)
    image = np.clip(image * contrast + brightness, 0.0, 1.0)
    return image.astype(np.float32), mask


class LungDataset(Dataset):
    """Chest X-ray images paired with binary lung masks."""

    def __init__(self, pairs, img_size=256, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path, _ = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.uint8)
        size = (self.img_size, self.img_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        if self.augment:
            image, mask = augment(image, mask)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def get_dataloaders(root, img_size=256, batch_size=8, seed=42, num_workers=2):
    """Build train, validation and test loaders. Only the training set is augmented."""
    pairs = find_pairs(root)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, seed=seed)

    train_ds = LungDataset(train_pairs, img_size, augment=True)
    val_ds = LungDataset(val_pairs, img_size)
    test_ds = LungDataset(test_pairs, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
