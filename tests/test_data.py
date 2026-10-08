"""Tests for data loading. These use small fake images, so the real dataset is not needed."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import torch
from lung_seg.data import LungDataset, find_pairs, split_pairs


def make_fake_root(tmp_path, per_source=20, size=64):
    rng = np.random.default_rng(0)
    for source in ["Montgomery", "Shenzhen"]:
        (tmp_path / source / "img").mkdir(parents=True)
        (tmp_path / source / "mask").mkdir(parents=True)
        for i in range(per_source):
            name = f"{source[:2]}_{i:03d}.png"
            image = rng.integers(0, 256, (size, size), dtype=np.uint8)
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[16:48, 16:48] = 255
            mask[16, 16] = 100  # a soft edge value that must be thresholded away
            cv2.imwrite(str(tmp_path / source / "img" / name), image)
            cv2.imwrite(str(tmp_path / source / "mask" / name), mask)
    return str(tmp_path)


def test_find_pairs_matches_each_image_with_its_mask(tmp_path):
    pairs = find_pairs(make_fake_root(tmp_path))
    assert len(pairs) == 40
    assert all(os.path.basename(p[0]) == os.path.basename(p[1]) for p in pairs)


def test_split_has_no_overlap_and_keeps_every_image(tmp_path):
    pairs = find_pairs(make_fake_root(tmp_path))
    train, val, test = split_pairs(pairs)
    train_set, val_set, test_set = ({p[0] for p in s} for s in (train, val, test))
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    assert len(train) + len(val) + len(test) == len(pairs)


def test_dataset_item_shapes_and_value_ranges(tmp_path):
    pairs = find_pairs(make_fake_root(tmp_path))
    image, mask = LungDataset(pairs, img_size=32)[0]
    assert image.shape == (1, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_augmentation_keeps_mask_binary(tmp_path):
    pairs = find_pairs(make_fake_root(tmp_path))
    dataset = LungDataset(pairs, img_size=32, augment=True)
    for i in range(5):
        _, mask = dataset[i]
        assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
