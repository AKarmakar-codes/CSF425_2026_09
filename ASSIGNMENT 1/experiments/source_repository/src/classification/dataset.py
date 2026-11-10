"""
Dataset and DataLoader utilities for PCB Component Classification.
Supports:
  - On-the-fly data augmentation (flips, rotations, jitter, affine)
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


def get_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Return image transformations for training or evaluation."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class PCBComponentCropDataset(Dataset):
    """
    Dataset for cropped PCB component instances.
    Accepts samples as:
      - list of dicts: [{'image_path': str, 'bbox': (x1, y1, x2, y2), 'class_id': int}, ...]
      OR
      - directory of pre-cropped images organized by class ID or mapping file.
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

        # If pre-cropped image exists on disk
        if "crop_path" in item and os.path.exists(item["crop_path"]):
            image = Image.open(item["crop_path"]).convert("RGB")
        elif "bbox" in item:
            # Crop on-the-fly from full PCB image
            if not os.path.exists(img_path):
                # Fallback path search for Kaggle/Colab
                base_img = os.path.basename(img_path)
                for cand_root in ["/kaggle/working/data/GraphPCB", "data/GraphPCB", "/kaggle/input/the-graphpcb-dataset/GraphPCB", "/kaggle/input/graphpcb/GraphPCB"]:
                    for sub in ["Graph-W/image_dir", "Graph-F/image_dir"]:
                        cand = os.path.join(cand_root, sub, base_img)
                        if os.path.exists(cand):
                            img_path = cand
                            break
                    if os.path.exists(img_path):
                        break

            full_img = Image.open(img_path).convert("RGB")
            x1, y1, x2, y2 = item["bbox"]
            # Ensure valid bounds
            w, h = full_img.size
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, max(x1 + 1, int(x2))), min(h, max(y1 + 1, int(y2)))
            image = full_img.crop((x1, y1, x2, y2))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Labels in assignment are 1..25; map to 0..24 for CrossEntropy
        target_0_indexed = int(item["class_id"]) - 1
        return image, target_0_indexed, img_path


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


# Re-export compute_class_balanced_weights from loss for convenience
from .loss import compute_class_balanced_weights, ClassBalancedFocalLoss


