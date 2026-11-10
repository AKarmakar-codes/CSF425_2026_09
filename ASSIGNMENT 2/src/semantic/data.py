"""Assignment II source-only tile dataset; does not import Assignment I code."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SourceTileDataset(Dataset):
    def __init__(self, root, split, augment=False, seed=42):
        self.root = Path(root)
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        meta = json.loads((self.root / "manifest.json").read_text())
        self.rows = json.loads((self.root / f"{split}.json").read_text())
        if not self.rows:
            raise ValueError(f"{split} set has no tiles")
        valid_domains = {"Graph-F", "Graph-W", "Graph-W-distilled", "GraphPCB"}
        if any(r.get("domain", "Graph-F") not in valid_domains for r in self.rows):
            raise ValueError(f"Tiles must belong to approved source domains {valid_domains}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        im = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_COLOR)
        ids = cv2.imread(str(self.root / row["instance_map"]), cv2.IMREAD_UNCHANGED)
        if im is None or ids is None:
            raise FileNotFoundError(f"{row['image']} / {row['instance_map']}")
        if ids.ndim != 2 or im.shape[:2] != ids.shape:
            raise ValueError("Image and instance map dimensions disagree")

        labels = [int(x["category_id"]) - 1 for x in row["objects"]]
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
            if rng.random() < 0.5:
                im = im[:, ::-1].copy()
                ids = ids[:, ::-1].copy()
            if rng.random() < 0.5:
                im = im[::-1].copy()
                ids = ids[::-1].copy()
            if rng.random() < 0.7:
                alpha = float(rng.uniform(0.8, 1.2))
                beta = float(rng.uniform(-12, 12))
                im = np.clip(im.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        instances = []
        for obj in row["objects"]:
            instances.append(ids == int(obj["instance_id"]))
        masks = torch.from_numpy(np.stack(instances) if instances else np.zeros((0, *ids.shape), dtype=bool))
        return {
            "image": torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": masks,
            "row": row,
        }


def list_collate(rows):
    return rows
