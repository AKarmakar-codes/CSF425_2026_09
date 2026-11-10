"""
Dataset and DataLoader utilities for PCB Component Instance Segmentation.
Supports:
  - COCO format annotations
  - Polygon to binary mask rasterization
  - PyTorch Mask R-CNN target formatting:
      - boxes: FloatTensor[N, 4] (x1, y1, x2, y2)
      - labels: Int64Tensor[N] (1-indexed class ID 1..25)
      - masks: UInt8Tensor[N, H, W] (binary masks)
      - image_id: Int64Tensor[1]
      - area: FloatTensor[N]
      - iscrowd: UInt8Tensor[N]
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms


def polygon_to_mask(polygon: List[float], height: int, width: int) -> np.ndarray:
    """Rasterize polygon coordinates [x1, y1, x2, y2, ...] into a binary mask of shape (H, W)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
    cv2.fillPoly(mask, [pts], color=1)
    return mask


class PCBInstanceSegmentationDataset(Dataset):
    """
    Instance segmentation dataset for PCB components.
    Loads images and per-instance pixel masks.
    """

    def __init__(
        self,
        samples: List[Dict],
        img_size: Optional[Tuple[int, int]] = None,
        is_train: bool = True,
    ):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], str]:
        item = self.samples[idx]
        image_path = item["image_path"]

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in item.get("annotations", []):
            cid = int(ann["class_id"])  # 1..25
            if not (1 <= cid <= 25):
                continue

            x1, y1, x2, y2 = ann["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue

            # Binary mask from polygon or bbox
            if "polygon" in ann and ann["polygon"]:
                mask = polygon_to_mask(ann["polygon"], orig_h, orig_w)
            elif "mask_rle" in ann:
                # Run-length encoded mask if available
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
            else:
                # Bbox fallback mask
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 1

            boxes.append([x1, y1, x2, y2])
            labels.append(cid)
            masks.append(mask)
            areas.append((x2 - x1) * (y2 - y1))

        if len(boxes) == 0:
            # Dummy target for images with no valid components
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, orig_h, orig_w), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        # Basic conversions
        img_tensor = transforms.functional.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.uint8),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        return img_tensor, target, image_path


def collate_fn(batch):
    return tuple(zip(*batch))

