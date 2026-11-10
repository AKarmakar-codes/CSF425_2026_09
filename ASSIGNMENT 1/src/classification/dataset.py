"""
Dataset and DataLoader utilities for PCB Component Classification.
Supports:
  - On-the-fly data augmentation (flips, rotations, jitter)
  - Balanced class weighting calculation
  - 1-indexed (1-25) to 0-indexed (0-24) label conversion
"""

import os
import json
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from src.classification.preprocessing import transform, crop_rgb
from src.classification.provenance import file_hash
from functools import lru_cache

@lru_cache(maxsize=8192)
def _hash_at_stat(path, size, mtime_ns):
    return file_hash(path)

def verified_hash(path):
    from pathlib import Path
    s = Path(path).stat()
    return _hash_at_stat(str(path), s.st_size, s.st_mtime_ns)


def load_sample_image(item):
    if not 1 <= int(item['class_id']) <= 25:
        raise ValueError('Invalid canonical class')
    storage = item.get('storage')
    if storage not in ('cache', 'source'):
        raise ValueError('Explicit repaired storage policy required')
    if storage == 'cache':
        path = item['crop_path']
        if verified_hash(path) != item['crop_sha256']:
            raise ValueError('Stale/corrupt cache; regenerate manifest and cache')
        with Image.open(path) as im:
            if im.format != 'PNG':
                raise ValueError('Lossless PNG cache required')
            return im.convert('RGB')
    path = item['image_path']
    if not item.get('source_image_sha256') or verified_hash(path) != item['source_image_sha256']:
        raise ValueError('Source image missing identity or changed; regenerate manifest')
    with Image.open(path) as im:
        return crop_rgb(im, item['bbox'])


def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Return image transformations for training or evaluation."""
    return transform(img_size, is_train)


class PCBComponentCropDataset(Dataset):
    """
    Dataset for cropped PCB component instances.
    Accepts repaired-manifest samples with explicit source/cache storage and identities.
    """

    def __init__(
        self,
        samples: List[Dict],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.samples[idx]
        img_path = item["image_path"]

        image = load_sample_image(item)

        if self.transform:
            image = self.transform(image)

        # Labels in assignment are 1..25; map to 0..24 for CrossEntropy
        target_0_indexed = int(item["class_id"]) - 1
        return image, target_0_indexed, item["sample_id"]


def compute_class_weights(samples: List[Dict], num_classes: int = 25) -> torch.Tensor:
    """
    Compute inverse frequency class weights to counteract severe PCB component imbalance.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for s in samples:
        cid = int(s["class_id"]) - 1
        if 0 <= cid < num_classes:
            counts[cid] += 1.0

    total = np.sum(counts)
    # Effective number of samples weighting or smoothed inverse frequency
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    # Clip extreme weights to avoid gradient explosions
    weights = np.clip(weights, 0.1, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(samples: List[Dict], num_classes: int = 25) -> WeightedRandomSampler:
    """
    Create a sampler that balances class representation in each batch.
    """
    counts = np.zeros(num_classes, dtype=np.float32)
    for s in samples:
        cid = int(s["class_id"]) - 1
        if 0 <= cid < num_classes:
            counts[cid] += 1.0

    sample_weights = []
    for s in samples:
        cid = int(s["class_id"]) - 1
        if 0 <= cid < num_classes and counts[cid] > 0:
            sample_weights.append(1.0 / counts[cid])
        else:
            sample_weights.append(0.0)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True
    )
    return sampler

