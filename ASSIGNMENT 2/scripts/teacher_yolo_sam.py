"""YOLO26-L / YOLO + SAM Teacher Pseudo-Ground-Truth Pipeline.

Conforms strictly to Section 3 (T1-T4) of DL_CSF425_Assignment-II.pdf:
T1. 25-class component detection and proposal generation.
T2. Frozen SAM geometry refinement (boxes as prompts -> crisp binary masks).
T3. Fixed annotation QC (same-class mask-NMS, board mask filtering, overlap resolution).
T4. Canonical versioned output (class_id, instance_id, bbox, score, encoded masks).

Can be executed locally or inside a Kaggle GPU notebook.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

CLASSES = (
    "R", "C", "U", "Q", "J", "L", "RA", "D", "RN", "TP",
    "IC", "P", "CR", "M", "BTN", "FB", "CRA", "SW", "T", "F",
    "V", "LED", "S", "QA", "JP"
)

SOURCE_RGB = (
    (255, 0, 0), (255, 255, 0), (0, 234, 255), (170, 0, 255), (255, 127, 0),
    (191, 255, 0), (0, 149, 255), (106, 255, 0), (0, 64, 255), (237, 185, 185),
    (185, 215, 237), (231, 233, 185), (220, 185, 237), (185, 237, 224), (143, 35, 35),
    (35, 98, 143), (143, 106, 35), (107, 35, 143), (79, 143, 35), (115, 115, 115),
    (204, 204, 204), (245, 130, 48), (220, 190, 255), (170, 255, 195), (255, 250, 200)
)

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def download_sam_weights(model_type: str = "vit_b", save_dir: str = "checkpoints") -> Path:
    """Download pretrained SAM weights if not present locally."""
    save_path = Path(save_dir) / f"sam_{model_type}.pth"
    if save_path.is_file() and save_path.stat().st_size > 10_000_000:
        return save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = SAM_URLS.get(model_type, SAM_URLS["vit_b"])
    print(f"Downloading SAM {model_type} weights from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved SAM weights to {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")
    return save_path


def load_sam(checkpoint_path: Path | str, model_type: str = "vit_b", device: str = "cuda"):
    """Initialize frozen Segment Anything Model (SAM)."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)
    except ImportError:
        print("[Error] 'segment_anything' not installed. Install via:")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        raise


def load_yolo(yolo_path: str | Path, device: str = "cuda"):
    """Load adapted YOLO detector/segmenter."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(yolo_path))
        model.to(device)
        return model
    except ImportError:
        print("[Error] 'ultralytics' not installed. Install via: pip install ultralytics")
        raise


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two boolean masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def run_teacher_on_image(
    image_path: Path | str,
    yolo_model,
    sam_predictor,
    out_dir: Path | str,
    board_mask_path: Path | str | None = None,
    tile_size: int = 1024,
    overlap: int = 256,
    conf_thresh: float = 0.30,
    nms_iou_thresh: float = 0.50,
    device: str = "cuda",
) -> dict:
    """Execute complete T1-T4 pipeline on a high-resolution PCB image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    im_bgr = cv2.imread(str(image_path))
    if im_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Optional Whole-Board Mask for Background Suppression (T3)
    board_mask = None
    if board_mask_path and Path(board_mask_path).is_file():
        bm = cv2.imread(str(board_mask_path), cv2.IMREAD_GRAYSCALE)
        if bm is not None:
            board_mask = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST) > 128

    # T1: Sliding Tile YOLO Proposals
    step = tile_size - overlap
    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    proposals = []
    print(f"[{stem}] Running T1 (YOLO) across {len(ys)*len(xs)} tiles...")
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tile_rgb = rgb[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            th, tw = tile_rgb.shape[:2]
            if th < 64 or tw < 64:
                continue

            # Run YOLO on tile
            res = yolo_model.predict(tile_rgb, conf=conf_thresh, verbose=False, device=device)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            for box in res.boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item()) + 1  # 1-based class ID (1..25)
                score = float(box.conf[0].item())

                # Map tile coords back to global image coords
                gx1 = int(round(coords[0] + x))
                gy1 = int(round(coords[1] + y))
                gx2 = int(round(coords[2] + x))
                gy2 = int(round(coords[3] + y))

                # Clamp
                gx1, gy1 = max(0, gx1), max(0, gy1)
                gx2, gy2 = min(w, gx2), min(h, gy2)
                if (gx2 - gx1) < 4 or (gy2 - gy1) < 4:
                    continue

                # Filter out proposal if completely outside board mask
                if board_mask is not None and not board_mask[gy1:gy2, gx1:gx2].any():
                    continue

                proposals.append({
                    "class_id": cls_id,
                    "score": score,
                    "bbox": [gx1, gy1, gx2, gy2],
                })

    print(f"[{stem}] T1 generated {len(proposals)} raw proposals. Running T2 (SAM geometry refinement)...")

    # T2: SAM Geometry Refinement
    # Sort proposals descending by score
    proposals.sort(key=lambda p: -p["score"])
    sam_predictor.set_image(rgb)

    refined_instances = []
    for prop in proposals:
        box = prop["bbox"]
        boxes_tensor = torch.tensor([box], device=device)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_tensor, (h, w))

        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        bin_mask = masks[0, 0].cpu().numpy().astype(bool)

        # Apply board mask clipping
        if board_mask is not None:
            bin_mask = np.logical_and(bin_mask, board_mask)

        area = int(bin_mask.sum())
        if area < 16:  # Discard empty / tiny artifact masks
            continue

        refined_instances.append({
            "class_id": prop["class_id"],
            "score": prop["score"],
            "bbox": box,
            "mask": bin_mask,
            "area": area,
        })

    # T3: Fixed Quality-Control (Same-Class Mask-NMS)
    print(f"[{stem}] Running T3 (Quality Control & Mask-NMS)...")
    kept_instances = []
    for inst in refined_instances:
        duplicate = False
        for kept in kept_instances:
            if inst["class_id"] == kept["class_id"]:
                iou = mask_iou(inst["mask"], kept["mask"])
                if iou >= nms_iou_thresh:
                    duplicate = True
                    break
        if not duplicate:
            kept_instances.append(inst)

    print(f"[{stem}] Retained {len(kept_instances)} quality-controlled instances.")

    # T4: Rendering & Manifest Export
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    instance_id_map = np.zeros((h, w), dtype=np.int32)
    export_instances = []

    # Paint instances ordered from lowest to highest score so higher confidence takes foreground
    for idx, inst in enumerate(reversed(kept_instances), start=1):
        m = inst["mask"]
        cid = inst["class_id"]
        color = SOURCE_RGB[cid - 1] if (1 <= cid <= 25) else (255, 255, 255)
        color_mask[m] = color
        instance_id_map[m] = idx

        export_instances.append({
            "instance_id": idx,
            "class_id": cid,
            "class_code": CLASSES[cid - 1] if (1 <= cid <= 25) else "UNKNOWN",
            "score": round(inst["score"], 4),
            "bbox": inst["bbox"],
            "area": inst["area"],
        })

    # Transparent RGB Overlay
    overlay = rgb.copy()
    fg = color_mask.any(axis=-1)
    overlay[fg] = (0.5 * rgb[fg] + 0.5 * color_mask[fg]).astype(np.uint8)

    # Save artifacts with canonical naming
    rgb_path = out_dir / f"{stem}_rgb.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    black_path = out_dir / f"{stem}_black_masks.png"
    id_path = out_dir / f"{stem}_instances.png"
    json_path = out_dir / f"{stem}_instances.json"

    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(color_mask).save(black_path)
    Image.fromarray(instance_id_map.astype(np.uint16)).save(id_path)
    json_path.write_text(json.dumps(export_instances, indent=2))

    print(f"[{stem}] Successfully exported teacher deliverables to {out_dir}/")
    return {
        "stem": stem,
        "instances_count": len(export_instances),
        "json_path": str(json_path),
        "mask_path": str(black_path),
    }


def main():
    p = argparse.ArgumentParser(description="YOLO + SAM Teacher Pipeline (CS F425 Assignment II)")
    p.add_argument("--image", required=True, help="Input PCB image path (or directory of images)")
    p.add_argument("--yolo", default="checkpoints/best_segment.ckpt", help="Path to 25-class adapted YOLO model (.ckpt or .pt)")
    p.add_argument("--sam-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--sam-ckpt", default=None, help="Path to SAM checkpoint (auto-downloads if missing)")
    p.add_argument("--out", default="results/teacher_reference", help="Output directory")
    p.add_argument("--board-mask", default=None, help="Optional whole-PCB foreground mask")
    p.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="Same-class Mask-NMS IoU threshold")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 1. Download or locate SAM weights
    sam_ckpt = Path(args.sam_ckpt) if args.sam_ckpt else download_sam_weights(args.sam_type, "checkpoints")

    # 2. Initialize Models
    print(f"Loading YOLO from {args.yolo} on {args.device}...")
    yolo_model = load_yolo(args.yolo, args.device)

    print(f"Loading frozen SAM ({args.sam_type}) from {sam_ckpt} on {args.device}...")
    sam_pred = load_sam(sam_ckpt, args.sam_type, args.device)

    # 3. Process image(s)
    target_path = Path(args.image)
    images = [target_path] if target_path.is_file() else sorted(target_path.glob("*.png")) + sorted(target_path.glob("*.jpg"))

    print(f"Starting teacher extraction for {len(images)} image(s)...")
    for img_p in images:
        run_teacher_on_image(
            img_p,
            yolo_model=yolo_model,
            sam_predictor=sam_pred,
            out_dir=args.out,
            board_mask_path=args.board_mask,
            conf_thresh=args.conf,
            nms_iou_thresh=args.iou,
            device=args.device,
        )


if __name__ == "__main__":
    main()
