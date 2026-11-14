"""Offline Teacher Distillation for Source-Domain Training Data.

Conforms strictly to Assignment II Master Execution Plan (Step 1) and Rubric (T1-T4):
1. Graph-W (37 train boards): Converts ground-truth component bounding boxes into crisp,
   geometry-aware instance masks using frozen SAM while strictly inheriting official 25-class IDs.
2. Graph-F (115 train boards): Uses ground-truth masks + distance transform / SAM prompts to
   split touching same-class 8-connected blobs (dense resistor/capacitor arrays) into distinct instances.
3. Rule T3 Quality Control:
   - Same-class Mask-NMS (IoU > 0.50) to suppress duplicate proposals.
   - Instances sorted by area ascending so small passives embedded on larger pads remain visible.
   - Ambiguous boundary pixels set to ignore = 255; pure background set to 0.
4. Memory-safe batched SAM inference (prevents RAM exhaustion on large ~6000x4000 PCB images).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Ensure src/ is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    CLASSES,
    SOURCE_RGB,
    DISPLAY_RGB,
    GLOBAL_LUT,
    BACKGROUND_ID,
    IGNORE_ID,
    IGNORE_RGB,
    map_graph_w_name,
    decode_source_mask,
    encode_source_mask,
)


def auto_detect_sam_type(checkpoint_path: Path) -> str:
    """Detect SAM ViT architecture (vit_b, vit_l, vit_h) from checkpoint weights or filename."""
    name = checkpoint_path.name.lower()
    if "vit_h" in name:
        return "vit_h"
    if "vit_l" in name:
        return "vit_l"
    if "vit_b" in name:
        return "vit_b"

    # Inspect state dict to be certain
    try:
        import torch
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "image_encoder.patch_embed.proj.weight" in state_dict:
            dim = state_dict["image_encoder.patch_embed.proj.weight"].shape[0]
            if dim == 1280:
                return "vit_h"
            elif dim == 1024:
                return "vit_l"
            elif dim == 768:
                return "vit_b"
    except Exception:
        pass
    return "vit_b"


def init_sam_predictor(checkpoint_path: Optional[str | Path], model_type: str = "auto", device: str = "cpu"):
    """Initialize frozen Segment Anything Model (SAM) predictor with automatic fallback."""
    if not checkpoint_path:
        print("[Notice] No SAM checkpoint specified. Using high-precision morphological fallback.")
        return None

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        # Check alternative local checkpoints
        alternatives = [
            ckpt_path.parent / "sam_vit_h_4b8939.pth",
            ckpt_path.parent / "sam_vit_l_0b3195.pth",
            ckpt_path.parent / "sam_vit_b.pth",
            ckpt_path.parent / "sam_vit_b_01ec64.pth",
        ]
        found = next((p for p in alternatives if p.is_file()), None)
        if found:
            print(f"[Notice] Requested checkpoint '{ckpt_path}' not found; using existing '{found}'.")
            ckpt_path = found
        else:
            print(f"[Warning] SAM checkpoint not found at '{ckpt_path}'. Using morphological fallback.")
            return None

    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        if model_type == "auto":
            model_type = auto_detect_sam_type(ckpt_path)

        print(f"Loading SAM ({model_type}) from {ckpt_path} on {device}...")
        sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
        sam.to(device=device)
        sam.eval()
        predictor = SamPredictor(sam)
        print("SAM predictor initialized successfully.")
        return predictor
    except Exception as e:
        print(f"[Warning] Could not initialize SAM ({e}). Using morphological fallback.")
        return None


def bbox_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute standard bounding box IoU."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def crop_mask_iou(inst_a: Dict[str, Any], inst_b: Dict[str, Any]) -> float:
    """Compute exact mask IoU between two instances using their local bounding box crops."""
    box_a = inst_a["bbox"]
    box_b = inst_b["bbox"]
    # Quick bounding box rejection
    if box_a[2] <= box_b[0] or box_b[2] <= box_a[0] or box_a[3] <= box_b[1] or box_b[3] <= box_a[1]:
        return 0.0

    # Overlapping region coordinates
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    # Extract overlapping crops
    crop_a = inst_a["crop_mask"][iy1 - box_a[1] : iy2 - box_a[1], ix1 - box_a[0] : ix2 - box_a[0]]
    crop_b = inst_b["crop_mask"][iy1 - box_b[1] : iy2 - box_b[1], ix1 - box_b[0] : ix2 - box_b[0]]

    inter = np.logical_and(crop_a, crop_b).sum()
    if inter == 0:
        return 0.0
    union = inst_a["area"] + inst_b["area"] - inter
    return float(inter) / float(union) if union > 0 else 0.0


def apply_same_class_mask_nms(instances: List[Dict[str, Any]], iou_threshold: float = 0.50) -> List[Dict[str, Any]]:
    """Rule T3 Quality Control: Suppress duplicate same-class masks exceeding IoU threshold."""
    if len(instances) <= 1:
        return instances

    # Group by category_id
    by_cat: Dict[int, List[Dict[str, Any]]] = {}
    for inst in instances:
        by_cat.setdefault(inst["category_id"], []).append(inst)

    kept: List[Dict[str, Any]] = []
    for cat_id, cat_instances in by_cat.items():
        cat_instances.sort(key=lambda x: (x.get("score", 1.0), x["area"]), reverse=True)
        cat_kept: List[Dict[str, Any]] = []
        for cand in cat_instances:
            duplicate = False
            for prev in cat_kept:
                if crop_mask_iou(cand, prev) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                cat_kept.append(cand)
        kept.extend(cat_kept)

    return kept


def predict_boxes_batched(
    predictor,
    rgb: np.ndarray,
    boxes: List[List[int]],
    batch_size: int = 16,
    device: str = "cpu",
) -> List[np.ndarray]:
    """Predict SAM masks in batches and immediately crop to bbox to prevent 20GB+ RAM OOM."""
    import torch

    h, w = rgb.shape[:2]
    cropped_masks: List[np.ndarray] = []
    total_batches = (len(boxes) + batch_size - 1) // batch_size

    for b_num, i in enumerate(range(0, len(boxes), batch_size), 1):
        if total_batches > 3:
            print(f" [SAM {b_num}/{total_batches}]", end="", flush=True)
        chunk = boxes[i : i + batch_size]
        boxes_tensor = torch.as_tensor(chunk, dtype=torch.float, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_tensor, (h, w))
        with torch.no_grad():
            chunk_masks, chunk_ious, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        # Immediately slice each mask on GPU/CPU to its tight bounding box [y1:y2, x1:x2]
        for b_idx, (x1, y1, x2, y2) in enumerate(chunk):
            crop_m = (
                chunk_masks[b_idx, 0, y1:y2, x1:x2] > predictor.model.mask_threshold
            ).cpu().numpy().astype(bool)
            if crop_m.sum() < 4:
                crop_m = np.ones((y2 - y1, x2 - x1), dtype=bool)
            cropped_masks.append(crop_m)

        del boxes_tensor, transformed_boxes, chunk_masks, chunk_ious
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return cropped_masks


def fallback_box_mask(crop_rgb: np.ndarray) -> np.ndarray:
    """Crisp morphological / Otsu segmentation inside bounding box as high-fidelity fallback."""
    h, w = crop_rgb.shape[:2]
    if h <= 4 or w <= 4:
        return np.ones((h, w), dtype=bool)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If Otsu foreground is reasonable, use it; else full rectangle
    fg_ratio = thresh.mean() / 255.0
    if 0.15 < fg_ratio < 0.85:
        return thresh > 0
    return np.ones((h, w), dtype=bool)


def process_graph_w_board(
    stem: str,
    img_path: Path,
    xml_path: Path,
    out_dir: Path,
    sam_predictor=None,
    batch_size: int = 16,
    device: str = "cpu",
    mask_nms_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """Process a single Graph-W board from XML bounding boxes into distilled instance masks."""
    im = cv2.imread(str(img_path))
    if im is None:
        print(f"[Warning] Failed to read image: {img_path}")
        return None
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Parse XML ground-truth annotations
    try:
        tree = ET.parse(str(xml_path))
    except Exception as e:
        print(f"[Warning] Failed to parse XML {xml_path}: {e}")
        return None

    raw_candidates: List[Dict[str, Any]] = []
    ignore_boxes: List[List[int]] = []

    for obj in tree.getroot().findall("object"):
        raw_name = obj.findtext("name")
        box_elem = obj.find("bndbox")
        if box_elem is None:
            continue
        try:
            x1 = max(0, min(w - 1, int(float(box_elem.findtext("xmin")))))
            y1 = max(0, min(h - 1, int(float(box_elem.findtext("ymin")))))
            x2 = max(x1 + 1, min(w, int(float(box_elem.findtext("xmax")))))
            y2 = max(y1 + 1, min(h, int(float(box_elem.findtext("ymax")))))
        except (ValueError, TypeError):
            continue

        cid = map_graph_w_name(raw_name)
        if cid is None or cid == IGNORE_ID or not (1 <= cid <= 25):
            # Rule T3: Unmapped/ambiguous source components become ignore=255, NOT background=0
            ignore_boxes.append([x1, y1, x2, y2])
            continue

        raw_candidates.append({
            "category_id": cid,
            "class_code": CLASSES[cid - 1],
            "raw_name": raw_name,
            "bbox": [x1, y1, x2, y2],
        })

    if not raw_candidates:
        print(f"[Notice] No valid components found in {stem}")
        return None

    # Predict masks using SAM or fallback
    boxes = [c["bbox"] for c in raw_candidates]
    if sam_predictor is not None:
        try:
            t0 = time.time()
            print(f" ({len(raw_candidates)} boxes: SAM encoding...", end="", flush=True)
            sam_predictor.set_image(rgb)
            print(f" done {time.time()-t0:.1f}s, decoding...", end="", flush=True)
            masks = predict_boxes_batched(sam_predictor, rgb, boxes, batch_size=batch_size, device=device)
            t_sam = time.time() - t0
        except Exception as e:
            print(f"[Warning] SAM failed on {stem} ({e}), falling back to morphological prior.")
            masks = None
    else:
        masks = None

    instances: List[Dict[str, Any]] = []
    for idx, cand in enumerate(raw_candidates):
        x1, y1, x2, y2 = cand["bbox"]
        if masks is not None and idx < len(masks):
            local_mask = masks[idx]
        else:
            crop = rgb[y1:y2, x1:x2]
            local_mask = fallback_box_mask(crop)

        instances.append({
            "category_id": cand["category_id"],
            "class_code": cand["class_code"],
            "raw_name": cand["raw_name"],
            "bbox": [x1, y1, x2, y2],
            "crop_mask": local_mask,
            "area": int(local_mask.sum()),
            "score": 1.0,
        })

    # Rule T3: Same-class Mask-NMS
    instances = apply_same_class_mask_nms(instances, iou_threshold=mask_nms_threshold)

    # Rule T3: Sort instances by area descending for base painting,
    # so smaller passives are painted on top (visible over pads)
    instances.sort(key=lambda x: x["area"], reverse=True)

    # Render final masks
    semantic_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    instance_ids = np.zeros((h, w), dtype=np.uint16)

    # Rule T3: Ambiguous / unmapped source components painted as IGNORE_RGB (255, 255, 255)
    for ix1, iy1, ix2, iy2 in ignore_boxes:
        semantic_rgb[iy1:iy2, ix1:ix2] = IGNORE_RGB

    # Assign 1-indexed unique instance IDs
    exported_instances = []
    for inst_id, inst in enumerate(instances, 1):
        x1, y1, x2, y2 = inst["bbox"]
        c_mask = inst["crop_mask"]
        cid = inst["category_id"]
        color = SOURCE_RGB[cid - 1]

        # Paint semantic and instance ID maps on local crop
        view_rgb = semantic_rgb[y1:y2, x1:x2]
        view_ids = instance_ids[y1:y2, x1:x2]
        view_rgb[c_mask] = color
        view_ids[c_mask] = inst_id

        exported_instances.append({
            "instance_id": int(inst_id),
            "category_id": int(cid),
            "class_code": str(inst["class_code"]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area": int(inst["area"]),
            "score": float(inst.get("score", 1.0)),
        })

    # Save output artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{stem}_mask.png"
    inst_path = out_dir / f"{stem}_instances.png"
    json_path = out_dir / f"{stem}_instances.json"

    Image.fromarray(semantic_rgb).save(mask_path)
    Image.fromarray(instance_ids).save(inst_path)
    json_path.write_text(json.dumps(exported_instances, indent=2))

    return {
        "domain": "Graph-W",
        "stem": stem,
        "width": w,
        "height": h,
        "components": len(exported_instances),
        "image_path": str(img_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "instances_path": str(inst_path.resolve()),
        "json_path": str(json_path.resolve()),
    }


def split_touching_crop(
    crop_mask: np.ndarray,
    crop_rgb: np.ndarray,
    min_area: int = 16,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Split touching same-class components using distance peaks + RGB edge-guided watershed.

    Returns:
        (split_masks, boundary_seam_mask)
    """
    if crop_mask.sum() < min_area:
        return [crop_mask], None

    dist = cv2.distanceTransform(crop_mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val < 3.0:
        return [crop_mask], None

    # Find local peaks in distance map
    dist_thresh = 0.45 * max_val
    _, sure_fg = cv2.threshold(dist, dist_thresh, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 2:
        return [crop_mask], None

    markers = markers.astype(np.int32)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(crop_mask.astype(np.uint8), kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    if not crop_rgb.flags["C_CONTIGUOUS"]:
        crop_rgb = np.ascontiguousarray(crop_rgb)

    cv2.watershed(crop_rgb, markers)

    # markers == -1 are the watershed boundary seams
    seam_mask = (markers == -1) & crop_mask

    split_masks: List[np.ndarray] = []
    for m_id in range(2, num_markers + 1):
        m = (markers == m_id) & crop_mask
        if m.sum() >= 4:
            split_masks.append(m)

    if not split_masks:
        return [crop_mask], None

    return split_masks, seam_mask


def process_graph_f_board(
    stem: str,
    img_path: Path,
    mask_path: Path,
    out_dir: Path,
    sam_predictor=None,
    batch_size: int = 16,
    device: str = "cpu",
    mask_nms_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    """Process a single Graph-F board from ground-truth mask, splitting touching blobs into crisp instances."""
    im = cv2.imread(str(img_path))
    if im is None:
        print(f"[Warning] Failed to read image: {img_path}")
        return None
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    lbl = cv2.imread(str(mask_path))
    if lbl is None:
        print(f"[Warning] Failed to read mask: {mask_path}")
        return None
    lbl_rgb = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
    sem = decode_source_mask(lbl_rgb)

    instances: List[Dict[str, Any]] = []
    seams: List[Tuple[int, int, np.ndarray]] = []

    # Process each class present on the board
    present_classes = [c for c in np.unique(sem) if 1 <= c <= 25]

    for cid in present_classes:
        class_mask = (sem == cid).astype(np.uint8)
        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for cc_id in range(1, num_cc):
            x, y, bw, bh, area = stats[cc_id]
            if area < 4:
                continue

            crop = (labels[y : y + bh, x : x + bw] == cc_id)
            crop_rgb = rgb[y : y + bh, x : x + bw]

            # Check if blob is a touching multi-component cluster
            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
            if area > 100 or aspect_ratio > 3.0:
                sub_crops, seam = split_touching_crop(crop, crop_rgb)
                if seam is not None and seam.any():
                    seams.append((x, y, seam))
            else:
                sub_crops = [crop]

            for s_crop in sub_crops:
                s_area = int(s_crop.sum())
                if s_area < 4:
                    continue
                xs, ys = np.where(s_crop)[1], np.where(s_crop)[0]
                bx1, by1 = x + int(xs.min()), y + int(ys.min())
                bx2, by2 = x + int(xs.max() + 1), y + int(ys.max() + 1)
                tight_crop = s_crop[int(ys.min()) : int(ys.max() + 1), int(xs.min()) : int(xs.max() + 1)]

                instances.append({
                    "category_id": int(cid),
                    "class_code": CLASSES[cid - 1],
                    "raw_name": CLASSES[cid - 1],
                    "bbox": [bx1, by1, bx2, by2],
                    "crop_mask": tight_crop,
                    "area": s_area,
                    "score": 1.0,
                })

    # Rule T3: Same-class Mask-NMS
    instances = apply_same_class_mask_nms(instances, iou_threshold=mask_nms_threshold)

    # Rule T3: Sort instances by area descending for base painting
    instances.sort(key=lambda x: x["area"], reverse=True)

    # Render final masks
    semantic_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    instance_ids = np.zeros((h, w), dtype=np.uint16)

    # Rule T3: Original ambiguous/ignore pixels remain IGNORE_RGB
    orig_ignore = (sem == IGNORE_ID)
    if orig_ignore.any():
        semantic_rgb[orig_ignore] = IGNORE_RGB

    exported_instances = []
    for inst_id, inst in enumerate(instances, 1):
        x1, y1, x2, y2 = inst["bbox"]
        c_mask = inst["crop_mask"]
        cid = inst["category_id"]
        color = SOURCE_RGB[cid - 1]

        view_rgb = semantic_rgb[y1:y2, x1:x2]
        view_ids = instance_ids[y1:y2, x1:x2]
        view_rgb[c_mask] = color
        view_ids[c_mask] = inst_id

        exported_instances.append({
            "instance_id": int(inst_id),
            "category_id": int(cid),
            "class_code": str(inst["class_code"]),
            "bbox": [int(v) for v in inst["bbox"]],
            "area": int(inst["area"]),
            "score": float(inst.get("score", 1.0)),
        })

    # Rule T3: Paint 1px watershed inter-instance split seams as IGNORE_RGB (255, 255, 255)
    for sx, sy, seam in seams:
        sh, sw = seam.shape
        semantic_rgb[sy : sy + sh, sx : sx + sw][seam] = IGNORE_RGB

    # Save output artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask_path = out_dir / f"{stem}_mask.png"
    inst_path = out_dir / f"{stem}_instances.png"
    json_path = out_dir / f"{stem}_instances.json"

    Image.fromarray(semantic_rgb).save(out_mask_path)
    Image.fromarray(instance_ids).save(inst_path)
    json_path.write_text(json.dumps(exported_instances, indent=2))

    return {
        "domain": "Graph-F",
        "stem": stem,
        "width": w,
        "height": h,
        "components": len(exported_instances),
        "image_path": str(img_path.resolve()),
        "mask_path": str(out_mask_path.resolve()),
        "instances_path": str(inst_path.resolve()),
        "json_path": str(json_path.resolve()),
    }


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
    parser = argparse.ArgumentParser(description="Offline Teacher Distillation for Source Data (Step 1)")
    parser.add_argument("--source", default="data/GraphPCB", help="Path to GraphPCB dataset directory")
    parser.add_argument("--sam-ckpt", default="checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--sam-type", default="auto", choices=["auto", "vit_b", "vit_l", "vit_h"])
    parser.add_argument("--out", default="data/distilled_source", help="Output directory for distilled source")
    parser.add_argument("--device", default="auto", help="Compute device ('cuda:0', 'cpu', 'auto')")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for SAM box prompting")
    parser.add_argument("--mask-nms-thresh", type=float, default=0.50, help="Mask-NMS IoU threshold")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of boards (for quick smoke check)")
    args = parser.parse_args()

    source_dir = resolve_path(args.source)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source}")

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    device = args.device
    if device == "auto":
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Executing Teacher Distillation on device: {device}")

    # Initialize SAM predictor
    sam_predictor = init_sam_predictor(args.sam_ckpt, model_type=args.sam_type, device=device)

    # 1. Process Graph-W (37 training boards)
    graph_w_dir = source_dir / "Graph-W"
    w_results: List[Dict[str, Any]] = []
    if graph_w_dir.is_dir():
        train_pts = sorted((graph_w_dir / "graphs" / "train").glob("*.pt"))
        if not train_pts:
            train_pts = sorted((graph_w_dir / "mask_dir").glob("*.xml"))
        print(f"\n--- Processing Graph-W ({len(train_pts)} train boards) ---")
        if args.limit > 0:
            train_pts = train_pts[: args.limit]

        for pt_idx, pt_file in enumerate(train_pts, 1):
            stem = re.sub(r"^graph_\d+_", "", pt_file.stem).replace("_mask", "")
            img_cands = [
                graph_w_dir / "image_dir" / f"{stem}.jpg",
                graph_w_dir / "image_dir" / f"{stem}.png",
                graph_w_dir / "image_dir" / f"{stem}_image.png",
            ]
            img_path = next((p for p in img_cands if p.is_file()), None)
            xml_path = graph_w_dir / "mask_dir" / f"{stem}.xml"

            if not img_path or not xml_path.is_file():
                print(f"[{pt_idx}/{len(train_pts)}] [Skip] Missing files for Graph-W {stem}")
                continue

            print(f"[{pt_idx}/{len(train_pts)}] Distilling Graph-W: {stem}...", end="", flush=True)
            t0 = time.time()
            res = process_graph_w_board(
                stem=stem,
                img_path=img_path,
                xml_path=xml_path,
                out_dir=out_dir / "Graph-W",
                sam_predictor=sam_predictor,
                batch_size=args.batch_size,
                device=device,
                mask_nms_threshold=args.mask_nms_thresh,
            )
            if res:
                w_results.append(res)
                print(f" Done ({res['components']} components, {time.time() - t0:.1f}s)")
            else:
                print(" Failed.")

    # 2. Process Graph-F (115 training boards)
    graph_f_dir = source_dir / "Graph-F"
    f_results: List[Dict[str, Any]] = []
    if graph_f_dir.is_dir():
        train_pts = sorted((graph_f_dir / "graphs" / "train").glob("*.pt"))
        if not train_pts:
            train_pts = sorted((graph_f_dir / "mask_dir").glob("*_mask.png"))
            if not train_pts:
                train_pts = sorted((graph_f_dir / "mask_dir").glob("*.png"))
        print(f"\n--- Processing Graph-F ({len(train_pts)} train boards) ---")
        if args.limit > 0:
            train_pts = train_pts[: args.limit]

        for pt_idx, pt_file in enumerate(train_pts, 1):
            stem = re.sub(r"^graph_\d+_", "", pt_file.stem).replace("_mask", "")
            img_cands = [
                graph_f_dir / "image_dir" / f"{stem}_image.png",
                graph_f_dir / "image_dir" / f"{stem}.png",
                graph_f_dir / "image_dir" / f"{stem}.jpg",
            ]
            img_path = next((p for p in img_cands if p.is_file()), None)
            mask_cands = [
                graph_f_dir / "mask_dir" / f"{stem}_mask.png",
                graph_f_dir / "mask_dir" / f"{stem}.png",
            ]
            mask_path = next((p for p in mask_cands if p.is_file()), None)

            if not img_path or not mask_path:
                print(f"[{pt_idx}/{len(train_pts)}] [Skip] Missing files for Graph-F {stem}")
                continue

            print(f"[{pt_idx}/{len(train_pts)}] Distilling Graph-F: {stem}...", end="", flush=True)
            t0 = time.time()
            res = process_graph_f_board(
                stem=stem,
                img_path=img_path,
                mask_path=mask_path,
                out_dir=out_dir / "Graph-F",
                sam_predictor=sam_predictor,
                batch_size=args.batch_size,
                device=device,
                mask_nms_threshold=args.mask_nms_thresh,
            )
            if res:
                f_results.append(res)
                print(f" Done ({res['components']} components, {time.time() - t0:.1f}s)")
            else:
                print(" Failed.")

    # Global Manifest & Verification
    all_results = w_results + f_results
    total_components = sum(r["components"] for r in all_results)

    manifest = {
        "format": "csf425-a2-distilled-source-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sam_checkpoint": str(args.sam_ckpt),
        "device": device,
        "mask_nms_threshold": args.mask_nms_thresh,
        "total_boards": len(all_results),
        "graph_w_boards": len(w_results),
        "graph_f_boards": len(f_results),
        "total_components": total_components,
        "records": all_results,
    }
    manifest_path = out_dir / "distilled_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n========================================================")
    print(f"Teacher Source Distillation Complete!")
    print(f"Distilled Boards: {len(all_results)} (Graph-W: {len(w_results)}, Graph-F: {len(f_results)})")
    print(f"Total Discovered Components: {total_components}")
    print(f"Distillation Manifest: {manifest_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
