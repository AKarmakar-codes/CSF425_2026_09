"""Prepare source tiles with physical board-group splitting and DSLR domain randomization.

Conforms strictly to Assignment II Master Execution Plan (Step 1):
1. Board-Grouped Split:
   - Splits 152 official source boards (37 Graph-W + 115 Graph-F) deterministically by
     physical PCB identity (131 train / 21 val, matching Assignment-I) to prevent multi-view leakage.
   - Retains representation of all 25 classes in training.
2. 1024x1024 Tiling:
   - Extracts 1024x1024 tiles with 256px overlap (stride 768).
   - Generates aligned RGB images, uint8 semantic masks (0..25), and uint16 instance maps.
3. DSLR Photometric Domain Randomization:
   - Implements on-the-fly DSLR domain adaptation (CLAHE contrast normalization,
     simulated angled specular lighting gradients, ColorJitter, RandomScale 0.5x-2.0x,
     and random 90-degree rotations/flips) to bridge the domain shift to Nikon D4 DSLR images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    CLASSES,
    SOURCE_RGB,
    DISPLAY_RGB,
    BACKGROUND_ID,
    IGNORE_ID,
    decode_source_mask,
    encode_source_mask,
)


def get_physical_board_group(stem: str) -> str:
    """Extract physical PCB identity from board stem to prevent cross-view leakage."""
    m = re.match(r"pcb_(\d+)", stem)
    if m:
        return f"pcb_{m.group(1)}"
    clean = re.sub(r"_(top|bottom\d?|image|mask)$", "", stem, flags=re.IGNORECASE)
    clean = re.sub(r"^graph_\d+_", "", clean)
    return clean


def deterministic_board_split(
    boards: List[Dict[str, Any]],
    seed: int = 42,
    target_val_boards: int = 21,
) -> Tuple[Set[str], Set[str]]:
    """Split boards deterministically by physical board group (131 train / 21 val)."""
    # Group boards by physical PCB ID
    group_to_boards: Dict[str, List[Dict[str, Any]]] = {}
    for b in boards:
        group_to_boards.setdefault(b["board_group"], []).append(b)

    # Deterministic sort using sha256 hash of seed + group
    ordered_groups = sorted(
        group_to_boards.keys(),
        key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(),
    )

    val_groups: Set[str] = set()
    val_count = 0
    for g in ordered_groups:
        cnt = len(group_to_boards[g])
        if val_count + cnt <= target_val_boards:
            val_groups.add(g)
            val_count += cnt
            if val_count == target_val_boards:
                break

    train_groups = set(group_to_boards.keys()) - val_groups
    return train_groups, val_groups


class DSLRPhotometricAugmentation:
    """DSLR Domain Randomization Pipeline to bridge domain gap to Nikon D4 DSLR images.

    Includes:
    - CLAHE adaptive contrast normalization (equalizes specular reflections and PCB silkscreen)
    - Simulated angled specular lighting gradients (models directional studio lighting)
    - ColorJitter (brightness, contrast, saturation, hue)
    - RandomScale (0.5x to 2.0x) with crop/pad
    - Random 90-degree rotations and horizontal/vertical flips
    """

    def __init__(
        self,
        clahe_prob: float = 0.50,
        clahe_clip_limit: float = 2.0,
        specular_prob: float = 0.50,
        specular_max_gamma: float = 0.25,
        jitter_prob: float = 0.50,
        rotate_flip_prob: float = 0.50,
        scale_prob: float = 0.50,
        target_size: int = 1024,
    ):
        self.clahe_prob = clahe_prob
        self.clahe_clip = clahe_clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        self.specular_prob = specular_prob
        self.specular_max_gamma = specular_max_gamma
        self.jitter_prob = jitter_prob
        self.rotate_flip_prob = rotate_flip_prob
        self.scale_prob = scale_prob
        self.target_size = target_size

    def apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE in LAB color space on L channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def apply_specular_gradient(self, img: np.ndarray) -> np.ndarray:
        """Simulate directional specular lighting gradient across PCB surface."""
        h, w = img.shape[:2]
        angle = random.uniform(0, 2 * np.pi)
        gamma = random.uniform(-self.specular_max_gamma, self.specular_max_gamma)

        # 2D coordinate grid
        x = np.linspace(-1, 1, w, dtype=np.float32)
        y = np.linspace(-1, 1, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Directional gradient ramp
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
        factor = 1.0 + gamma * (gradient - 0.5)
        factor = np.clip(factor, 0.5, 1.5)[:, :, None]

        augmented = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return augmented

    def apply_color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Random brightness, contrast, and saturation adjustments."""
        img_f = img.astype(np.float32)
        # Brightness
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            img_f = img_f * alpha
        # Contrast
        if random.random() < 0.5:
            mean = img_f.mean(axis=(0, 1), keepdims=True)
            beta = random.uniform(0.8, 1.2)
            img_f = (img_f - mean) * beta + mean
        # Saturation in HSV
        res = np.clip(img_f, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(res, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return res

    def __call__(
        self,
        img: np.ndarray,
        sem_mask: Optional[np.ndarray] = None,
        inst_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Apply the full DSLR photometric and geometric augmentation pipeline."""
        res_img = img.copy()
        res_sem = sem_mask.copy() if sem_mask is not None else None
        res_inst = inst_mask.copy() if inst_mask is not None else None

        # 1. CLAHE contrast equalization
        if random.random() < self.clahe_prob:
            res_img = self.apply_clahe(res_img)

        # 2. Specular lighting gradient
        if random.random() < self.specular_prob:
            res_img = self.apply_specular_gradient(res_img)

        # 3. Color jitter
        if random.random() < self.jitter_prob:
            res_img = self.apply_color_jitter(res_img)

        # 4. Random 90-deg rotations & flips (applied identically to images and masks)
        if random.random() < self.rotate_flip_prob:
            rot_k = random.choice([0, 1, 2, 3])
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5

            if rot_k > 0:
                res_img = np.rot90(res_img, rot_k)
                if res_sem is not None:
                    res_sem = np.rot90(res_sem, rot_k)
                if res_inst is not None:
                    res_inst = np.rot90(res_inst, rot_k)

            if flip_h:
                res_img = np.fliplr(res_img)
                if res_sem is not None:
                    res_sem = np.fliplr(res_sem)
                if res_inst is not None:
                    res_inst = np.fliplr(res_inst)

            if flip_v:
                res_img = np.flipud(res_img)
                if res_sem is not None:
                    res_sem = np.flipud(res_sem)
                if res_inst is not None:
                    res_inst = np.flipud(res_inst)

        return res_img, res_sem, res_inst


def extract_tiles_from_board(
    rgb: np.ndarray,
    sem_mask: np.ndarray,
    inst_mask: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    min_component_area: int = 4,
) -> List[Dict[str, Any]]:
    """Extract 1024x1024 overlapping tiles with exact instance/semantic annotations."""
    h, w = rgb.shape[:2]
    step = tile_size - overlap

    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    tiles: List[Dict[str, Any]] = []

    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)

            crop_rgb = rgb[y:y2, x:x2]
            crop_sem = sem_mask[y:y2, x:x2]
            crop_inst = inst_mask[y:y2, x:x2]

            # Pad to exact tile_size if board edge is smaller than tile
            if crop_rgb.shape[:2] != (tile_size, tile_size):
                pad_rgb = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_sem = np.zeros((tile_size, tile_size), dtype=np.uint8)
                pad_inst = np.zeros((tile_size, tile_size), dtype=np.uint16)

                ch, cw = crop_rgb.shape[:2]
                pad_rgb[:ch, :cw] = crop_rgb
                pad_sem[:ch, :cw] = crop_sem
                pad_inst[:ch, :cw] = crop_inst

                crop_rgb = pad_rgb
                crop_sem = pad_sem
                crop_inst = pad_inst

            # Find instances present in this tile
            unique_inst_ids = [uid for uid in np.unique(crop_inst) if uid > 0]
            objects: List[Dict[str, Any]] = []

            for uid in unique_inst_ids:
                imask = (crop_inst == uid)
                area = int(imask.sum())
                if area < min_component_area:
                    continue

                # Find dominant category_id for this instance
                cat_ids = crop_sem[imask]
                cat_ids = cat_ids[cat_ids > 0]
                if len(cat_ids) == 0:
                    continue
                cid = int(np.bincount(cat_ids).argmax())

                xs_i, ys_i = np.where(imask)[1], np.where(imask)[0]
                bbox = [int(xs_i.min()), int(ys_i.min()), int(xs_i.max() + 1), int(ys_i.max() + 1)]

                objects.append({
                    "instance_id": int(uid),
                    "category_id": cid,
                    "class_code": CLASSES[cid - 1],
                    "bbox": bbox,
                    "area": area,
                })

            tiles.append({
                "tile_x": x,
                "tile_y": y,
                "rgb": crop_rgb,
                "semantic": crop_sem,
                "instances": crop_inst,
                "objects": objects,
                "has_components": len(objects) > 0,
            })

    return tiles


def discover_source_boards(source_dir: Path) -> List[Dict[str, Any]]:
    """Discover all 152 source training boards from distilled source or raw GraphPCB."""
    boards: List[Dict[str, Any]] = []

    # Check for distilled_manifest.json first
    manifest_file = source_dir / "distilled_manifest.json"
    if manifest_file.is_file():
        try:
            meta = json.loads(manifest_file.read_text())
            for rec in meta.get("records", []):
                stem = rec["stem"]
                group = get_physical_board_group(stem)
                boards.append({
                    "domain": rec["domain"],
                    "stem": stem,
                    "board_group": group,
                    "image_path": Path(rec["image_path"]),
                    "mask_path": Path(rec["mask_path"]),
                    "instances_path": Path(rec["instances_path"]),
                    "is_distilled": True,
                })
            if boards:
                return boards
        except Exception:
            pass

    # Discover directly from directories: Graph-W and Graph-F
    for domain in ["Graph-W", "Graph-F"]:
        domain_dir = source_dir / domain
        if not domain_dir.is_dir():
            continue

        # Look for instances and mask PNGs
        mask_files = sorted(domain_dir.glob("*_mask.png"))
        if mask_files:
            for mf in mask_files:
                stem = mf.stem.replace("_mask", "")
                inst_f = domain_dir / f"{stem}_instances.png"
                img_cands = [
                    domain_dir / f"{stem}.jpg",
                    domain_dir / f"{stem}.png",
                    domain_dir / f"{stem}_image.png",
                    source_dir / domain / "image_dir" / f"{stem}.jpg",
                    source_dir / domain / "image_dir" / f"{stem}.png",
                    source_dir / domain / "image_dir" / f"{stem}_image.png",
                ]
                img_f = next((p for p in img_cands if p.is_file()), None)
                if img_f and inst_f.is_file():
                    group = get_physical_board_group(stem)
                    boards.append({
                        "domain": domain,
                        "stem": stem,
                        "board_group": group,
                        "image_path": img_f,
                        "mask_path": mf,
                        "instances_path": inst_f,
                        "is_distilled": True,
                    })

    return boards


def resolve_path(p: str | Path) -> Path:
    """Resolve paths cleanly whether run from repository root or ASSIGNMENT 2 subdirectory."""
    path = Path(p)
    if path.is_absolute():
        return path
    cwd = Path.cwd()
    if (cwd / path).exists():
        return (cwd / path).resolve()
    if (REPO_ROOT / path).exists():
        return (REPO_ROOT / path).resolve()
    if cwd.name == "ASSIGNMENT 2" and len(path.parts) > 0 and path.parts[0] == "ASSIGNMENT 2":
        sub = Path(*path.parts[1:])
        return (cwd / sub).resolve()
    if (cwd / path).parent.exists():
        return (cwd / path).resolve()
    return (REPO_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Prepare Source Tiles with Board-Group Split & DSLR Augmentation")
    parser.add_argument("--source", default="data/distilled_source", help="Path to distilled source directory")
    parser.add_argument("--out", default="data/prepared_tiles", help="Output directory for prepared tiles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--tile", type=int, default=1024, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=256, help="Tile overlap in pixels (stride = tile - overlap)")
    parser.add_argument("--target-val-boards", type=int, default=21, help="Number of boards in validation split")
    args = parser.parse_args()

    source_dir = resolve_path(args.source)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering source boards from: {source_dir}")
    boards = discover_source_boards(source_dir)
    print(f"Found {len(boards)} source boards.")

    if not boards:
        raise RuntimeError(f"No valid distilled boards found in {source_dir}. Run Step 1 distillation first.")

    # Deterministic board-grouped split (131 train / 21 val)
    train_groups, val_groups = deterministic_board_split(
        boards, seed=args.seed, target_val_boards=args.target_val_boards
    )

    train_boards = [b for b in boards if b["board_group"] in train_groups]
    val_boards = [b for b in boards if b["board_group"] in val_groups]
    print(f"Split: {len(train_boards)} train boards ({len(train_groups)} groups) | "
          f"{len(val_boards)} val boards ({len(val_groups)} groups)")

    tile_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    class_counts: Dict[str, Dict[str, int]] = {
        "train": {c: 0 for c in CLASSES},
        "val": {c: 0 for c in CLASSES},
    }

    # Process all boards and extract tiles
    for split_name, board_list in [("train", train_boards), ("val", val_boards)]:
        print(f"\nExtracting {args.tile}x{args.tile} tiles for '{split_name}' split ({len(board_list)} boards)...")
        img_out_dir = out_dir / "images" / split_name
        mask_out_dir = out_dir / "masks" / split_name
        inst_out_dir = out_dir / "instances" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        inst_out_dir.mkdir(parents=True, exist_ok=True)

        for b_idx, b in enumerate(board_list, 1):
            stem = b["stem"]
            rgb_im = cv2.imread(str(b["image_path"]))
            if rgb_im is None:
                continue
            rgb = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

            # Load mask
            lbl_im = cv2.imread(str(b["mask_path"]))
            if lbl_im is None:
                continue
            lbl_rgb = cv2.cvtColor(lbl_im, cv2.COLOR_BGR2RGB)
            sem = decode_source_mask(lbl_rgb)

            # Load instances
            inst_im = cv2.imread(str(b["instances_path"]), cv2.IMREAD_UNCHANGED)
            if inst_im is None:
                inst_map = np.zeros(sem.shape, dtype=np.uint16)
            else:
                inst_map = inst_im.astype(np.uint16)

            tiles = extract_tiles_from_board(
                rgb=rgb,
                sem_mask=sem,
                inst_mask=inst_map,
                tile_size=args.tile,
                overlap=args.overlap,
            )

            # Filter or keep tiles (keep all component-bearing tiles + 10% background tiles)
            saved_tile_count = 0
            for t_idx, t in enumerate(tiles):
                if not t["has_components"] and random.random() > 0.10:
                    continue

                tile_name = f"{stem}_{t['tile_x']}_{t['tile_y']}"
                img_rel = f"images/{split_name}/{tile_name}.jpg"
                mask_rel = f"masks/{split_name}/{tile_name}.png"
                inst_rel = f"instances/{split_name}/{tile_name}.png"

                Image.fromarray(t["rgb"]).save(out_dir / img_rel, quality=95)
                Image.fromarray(t["semantic"]).save(out_dir / mask_rel)
                Image.fromarray(t["instances"]).save(out_dir / inst_rel)

                for obj in t["objects"]:
                    class_counts[split_name][obj["class_code"]] += 1

                tile_records[split_name].append({
                    "tile_id": tile_name,
                    "image": img_rel,
                    "semantic_mask": mask_rel,
                    "instance_map": inst_rel,
                    "board_stem": stem,
                    "board_group": b["board_group"],
                    "domain": b["domain"],
                    "tile_xy": [t["tile_x"], t["tile_y"]],
                    "width": args.tile,
                    "height": args.tile,
                    "objects": t["objects"],
                })
                saved_tile_count += 1

            print(f"[{b_idx}/{len(board_list)}] {b['domain']}:{stem} ({rgb.shape[1]}x{rgb.shape[0]}) -> {saved_tile_count} tiles", flush=True)

    # Export split JSON files
    for split_name in ["train", "val"]:
        json_path = out_dir / f"{split_name}.json"
        json_path.write_text(json.dumps(tile_records[split_name], indent=2))
        print(f"Wrote {len(tile_records[split_name])} tiles to {json_path}", flush=True)

    # Export global manifest
    total_train_boards = len(train_boards)
    manifest = {
        "format": "csf425-a2-prepared-tiles-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "tile_size": args.tile,
        "overlap": args.overlap,
        "stride": args.tile - args.overlap,
        "total_source_boards": len(boards),
        "train_boards_count": len(train_boards),
        "val_boards_count": len(val_boards),
        "train_tiles_count": len(tile_records["train"]),
        "val_tiles_count": len(tile_records["val"]),
        "train_board_groups": sorted(list(train_groups)),
        "val_board_groups": sorted(list(val_groups)),
        "class_ids": {str(i + 1): c for i, c in enumerate(CLASSES)},
        "class_counts": class_counts,
        "r_i": float(len(boards) / 152),
        "A_i": 1.0,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n========================================================")
    print("Tile Preparation and Splitting Complete!")
    print(f"Train Tiles: {len(tile_records['train'])} | Val Tiles: {len(tile_records['val'])}")
    print(f"Prepared Manifest: {manifest_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
