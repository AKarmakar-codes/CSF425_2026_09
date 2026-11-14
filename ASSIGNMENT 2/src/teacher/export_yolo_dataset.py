"""Export prepared source tiles into Ultralytics YOLO26-seg format.

Conforms strictly to Assignment II Master Execution Plan (Step 2, Rule T1):
- Converts 1024x1024 source tiles and distilled instance masks into normalized YOLO segmentation polygons.
- 0-indexes the official 25 classes (0..24, corresponding to R..JP).
- Generates dataset.yaml pointing to train and val image sets.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import CLASSES, NUM_CLASSES, IGNORE_ID


def mask_to_normalized_polygon(mask: np.ndarray, min_points: int = 3) -> List[float]:
    """Convert a binary mask crop into normalized polygon coordinates [x0, y0, x1, y1, ...]."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Largest contour
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < min_points:
        return []
    # Simplify contour slightly to avoid huge text files while keeping fidelity
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < min_points:
        approx = largest

    coords = approx.reshape(-1, 2)
    norm_coords = []
    for x, y in coords:
        norm_coords.extend([round(float(x) / w, 6), round(float(y) / h, 6)])
    return norm_coords


def bbox_to_normalized_polygon(bbox: List[int], tile_size: int = 1024) -> List[float]:
    """Fallback: convert [x1, y1, x2, y2] bounding box to a normalized 4-point rectangle polygon."""
    x1, y1, x2, y2 = bbox
    w, h = float(tile_size), float(tile_size)
    nx1, ny1 = max(0.0, min(1.0, x1 / w)), max(0.0, min(1.0, y1 / h))
    nx2, ny2 = max(0.0, min(1.0, x2 / w)), max(0.0, min(1.0, y2 / h))
    return [
        round(nx1, 6), round(ny1, 6),
        round(nx2, 6), round(ny1, 6),
        round(nx2, 6), round(ny2, 6),
        round(nx1, 6), round(ny2, 6)
    ]


def export_split(
    tiles_meta_file: Path,
    images_in_dir: Path,
    masks_in_dir: Path,
    images_out_dir: Path,
    labels_out_dir: Path,
    link_mode: str = "symlink",
) -> int:
    """Export one split (train or val) into YOLO images/ and labels/ directories."""
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    if not tiles_meta_file.is_file():
        print(f"[Warning] Meta file not found: {tiles_meta_file}")
        return 0

    with open(tiles_meta_file) as f:
        tiles = json.load(f)

    exported_count = 0
    for t_idx, tile in enumerate(tqdm(tiles, desc=f"Exporting {tiles_meta_file.stem}")):
        # Support both legacy key names and new schema from prepare_source.py
        img_name = tile.get("image_filename") or tile.get("image", "")
        mask_name = tile.get("mask_filename") or tile.get("semantic_mask", "")
        inst_name = tile.get("instance_map", "")  # instances/split/tile.png

        # Strip directory prefix if present (tile records may store full relative path)
        img_basename = Path(img_name).name
        mask_basename = Path(mask_name).name

        src_img = images_in_dir / img_basename
        src_mask = masks_in_dir / mask_basename

        if not src_img.is_file():
            continue

        dst_img = images_out_dir / img_basename
        # Link or copy image
        if not dst_img.exists():
            if link_mode == "symlink":
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        # Parse objects and extract segmentation polygons
        label_txt_path = labels_out_dir / (Path(img_basename).stem + ".txt")
        lines = []

        objects = tile.get("objects", [])
        if objects and src_mask.is_file():
            # Read semantic mask (uint8 class IDs 0..25,255) and instance map (uint16)
            sem = cv2.imread(str(src_mask), cv2.IMREAD_UNCHANGED)

            # Resolve instance map: use "instance_map" key relative path if available,
            # otherwise fall back to the sibling instances/ directory
            inst_map = None
            if inst_name:
                # inst_name is a relative path like "instances/train/tile.png"
                inst_abs = tiles_meta_file.parent / inst_name
                if inst_abs.is_file():
                    inst_map = cv2.imread(str(inst_abs), cv2.IMREAD_UNCHANGED)
            if inst_map is None:
                # Legacy fallback: images_in_dir/../instances/split/<mask_basename>
                inst_mask_path = src_mask.parent.parent / "instances" / src_mask.parent.name / mask_basename
                if inst_mask_path.is_file():
                    inst_map = cv2.imread(str(inst_mask_path), cv2.IMREAD_UNCHANGED)

            for obj in objects:
                cid = obj["category_id"]
                if not (1 <= cid <= 25):
                    continue
                yolo_cls = cid - 1  # 0-indexed (0..24)
                inst_id = obj.get("instance_id")

                poly = []
                if inst_map is not None and inst_id is not None:
                    obj_mask = (inst_map == inst_id)
                    poly = mask_to_normalized_polygon(obj_mask)

                if not poly and "bbox" in obj:
                    poly = bbox_to_normalized_polygon(obj["bbox"], tile_size=1024)

                if len(poly) >= 6:
                    poly_str = " ".join(str(p) for p in poly)
                    lines.append(f"{yolo_cls} {poly_str}")

        with open(label_txt_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        exported_count += 1

    return exported_count


def generate_dataset_yaml(out_dir: Path) -> Path:
    """Generate YOLO dataset.yaml defining the 25 PCB classes."""
    yaml_path = out_dir / "dataset.yaml"
    names_dict = {i: c for i, c in enumerate(CLASSES)}

    content = f"""# Ultralytics YOLO26-seg Dataset Configuration for 25 PCB Classes
# Conforms to Assignment II Master Execution Plan (Rule T1)
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
"""
    for i, name in names_dict.items():
        content += f"  {i}: {name}\n"

    yaml_path.write_text(content)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Export prepared tiles to YOLO segmentation format")
    parser.add_argument("--tiles-dir", default="data/prepared_tiles", help="Path to prepared tiles directory")
    parser.add_argument("--out-dir", default="data/yolo_dataset", help="Output directory for YOLO dataset")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    link_mode = "copy" if args.copy else "symlink"

    print("=== Exporting Source Tiles to YOLO26-seg Dataset Format ===")
    print(f"Source Tiles: {tiles_dir}")
    print(f"Destination:  {out_dir}")

    # Export Train Split
    n_train = export_split(
        tiles_meta_file=tiles_dir / "train.json",
        images_in_dir=tiles_dir / "images" / "train",
        masks_in_dir=tiles_dir / "masks" / "train",
        images_out_dir=out_dir / "images" / "train",
        labels_out_dir=out_dir / "labels" / "train",
        link_mode=link_mode,
    )

    # Export Val Split
    n_val = export_split(
        tiles_meta_file=tiles_dir / "val.json",
        images_in_dir=tiles_dir / "images" / "val",
        masks_in_dir=tiles_dir / "masks" / "val",
        images_out_dir=out_dir / "images" / "val",
        labels_out_dir=out_dir / "labels" / "val",
        link_mode=link_mode,
    )

    yaml_file = generate_dataset_yaml(out_dir)
    print("\n========================================================")
    print(f"Export Complete! Train tiles: {n_train} | Val tiles: {n_val}")
    print(f"YOLO Dataset Configuration: {yaml_file}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
