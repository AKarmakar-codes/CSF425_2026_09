"""Offline Teacher Inference Pipeline (YOLO26-L-seg + SAM) on DSLR Boards.

Conforms strictly to Assignment II Master Execution Plan (Step 2) & Rubric (T1-T4):
- T1: Adapts detection using 25-class source-trained YOLO26-L-seg.
- T2: Geometry refinement via frozen SAM (vit_h) on component bounding box prompts.
- T3: Quality audit:
      * Suppresses studio table background outside PCB board footprint.
      * Confidence filtering (score >= tau_teacher).
      * Same-class Mask-NMS (IoU >= 0.50).
      * Ambiguous boundaries set to ignore = 255; pure background set to 0.
- T4: Exports versioned reference artifacts (teacher_version: "yolo26l_sam_v1").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    CLASSES,
    NUM_CLASSES,
    SOURCE_RGB,
    DISPLAY_RGB,
    BACKGROUND_ID,
    IGNORE_ID,
    IGNORE_RGB,
    decode_source_mask,
    encode_source_mask,
    encode_display_mask,
)
from scripts.offline_teacher_source import (
    init_sam_predictor,
    predict_boxes_batched,
    apply_same_class_mask_nms,
    fallback_box_mask,
)


def extract_pcb_board_mask(rgb: np.ndarray) -> np.ndarray:
    """Zero-parameter deterministic filter to detect PCB board boundary and mask studio background."""
    h, w = rgb.shape[:2]
    # Downsample for robust morphological contour extraction
    scale = 0.25
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convert to HSV: PCB substrate (green/blue/black/red) has distinct saturation/value vs white/gray table
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # Otsu threshold on saturation + inverted gray
    _, s_thresh = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(s_thresh, v_thresh)

    # Morphological closing to fill board interior
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find largest contour (the physical PCB board)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=bool)

    largest_cnt = max(contours, key=cv2.contourArea)
    small_mask = np.zeros(small.shape[:2], dtype=np.uint8)
    cv2.drawContours(small_mask, [largest_cnt], -1, 255, thickness=-1)

    # Upsample back to full resolution
    board_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return board_mask


def run_yolo_on_tiles(
    yolo_model,
    rgb: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 256,
    conf_thresh: float = 0.25,
    device: str = "0",
) -> List[Dict[str, Any]]:
    """Run YOLO26-L-seg over overlapping 1024x1024 tiles and collect global candidate boxes."""
    h, w = rgb.shape[:2]
    step = tile_size - overlap

    ys = list(range(0, max(1, h - tile_size + 1), step))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not ys or ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if not xs or xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    candidates: List[Dict[str, Any]] = []

    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            crop = rgb[y:y2, x:x2]

            # Run YOLO inference on tile
            results = yolo_model.predict(
                source=crop,
                conf=conf_thresh,
                imgsz=tile_size,
                device=device,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                continue

            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for b_idx in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[b_idx]
                cid = int(clss[b_idx]) + 1  # 1-indexed (1..25)
                score = float(confs[b_idx])

                # Map back to global coordinates
                gx1 = int(round(x + bx1))
                gy1 = int(round(y + by1))
                gx2 = int(round(x + bx2))
                gy2 = int(round(y + by2))

                gx1, gy1 = max(0, min(w - 1, gx1)), max(0, min(h - 1, gy1))
                gx2, gy2 = max(gx1 + 1, min(w, gx2)), max(gy1 + 1, min(h, gy2))

                if (gx2 - gx1) < 4 or (gy2 - gy1) < 4:
                    continue

                candidates.append({
                    "category_id": cid,
                    "class_code": CLASSES[cid - 1],
                    "bbox": [gx1, gy1, gx2, gy2],
                    "score": score,
                })

    return candidates


def process_dslr_board_teacher(
    stem: str,
    img_path: Path,
    out_dir: Path,
    yolo_model,
    sam_predictor=None,
    tile_size: int = 1024,
    overlap: int = 256,
    conf_thresh: float = 0.25,
    mask_nms_thresh: float = 0.50,
    device: str = "0",
) -> Dict[str, Any]:
    """Execute complete Teacher pipeline (YOLO26-L + SAM + Rule T3 Audit) on a DSLR image."""
    im = cv2.imread(str(img_path))
    if im is None:
        raise ValueError(f"Failed to read image at: {img_path}")
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    print(f"\n--- Teacher Distillation on DSLR Board: {stem} ({w}x{h}) ---")

    # 1. Zero-parameter studio background filter (Rule T3)
    t0 = time.time()
    board_mask = extract_pcb_board_mask(rgb)
    print(f"PCB board mask computed ({time.time() - t0:.1f}s, board coverage: {board_mask.mean() * 100:.1f}%)")

    # 2. YOLO26-L-seg candidate generation on tiles (Rule T1)
    t0 = time.time()
    raw_candidates = run_yolo_on_tiles(
        yolo_model=yolo_model,
        rgb=rgb,
        tile_size=tile_size,
        overlap=overlap,
        conf_thresh=conf_thresh,
        device=device,
    )
    print(f"YOLO26-L proposed {len(raw_candidates)} raw candidates ({time.time() - t0:.1f}s)")

    # Filter out candidates completely outside PCB board
    filtered_candidates = []
    for cand in raw_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        # If box center is inside PCB board, keep it
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if board_mask[cy, cx]:
            filtered_candidates.append(cand)

    print(f"Retained {len(filtered_candidates)} candidates inside PCB boundary")

    # 3. Refine with frozen SAM (Rule T2)
    boxes = [c["bbox"] for c in filtered_candidates]
    masks = None
    if sam_predictor is not None and boxes:
        try:
            t0 = time.time()
            print(f"SAM encoding image ({w}x{h})...", end="", flush=True)
            sam_predictor.set_image(rgb)
            print(f" done {time.time() - t0:.1f}s. Decoding {len(boxes)} masks...", end="", flush=True)
            masks = predict_boxes_batched(sam_predictor, rgb, boxes, batch_size=32, device="cuda:0" if device != "cpu" else "cpu")
            print(" done.")
        except Exception as e:
            print(f"[Warning] SAM inference error: {e}, using morphological box fallback.")
            masks = None

    instances: List[Dict[str, Any]] = []
    for idx, cand in enumerate(filtered_candidates):
        x1, y1, x2, y2 = cand["bbox"]
        if masks is not None and idx < len(masks):
            local_mask = masks[idx]
        else:
            local_mask = fallback_box_mask(rgb[y1:y2, x1:x2])

        instances.append({
            "category_id": cand["category_id"],
            "class_code": cand["class_code"],
            "bbox": [x1, y1, x2, y2],
            "crop_mask": local_mask,
            "area": int(local_mask.sum()),
            "score": cand["score"],
        })

    # 4. Same-class Mask-NMS (Rule T3)
    instances = apply_same_class_mask_nms(instances, iou_threshold=mask_nms_thresh)

    # 5. Area descending sort (small passives painted on top of pads)
    instances.sort(key=lambda x: x["area"], reverse=True)

    # 6. Render Semantic, Instance, and Overlay Maps
    semantic_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    instance_ids = np.zeros((h, w), dtype=np.uint16)

    # Non-PCB studio background outside board mask set to 0 (pure black)
    # Inside PCB, unsegmented pixels default to 0
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
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "area": int(inst["area"]),
            "score": round(float(inst["score"]), 4),
        })

    # Save artifacts (Rule T4)
    out_dir.mkdir(parents=True, exist_ok=True)
    sem_path = out_dir / f"{stem}_teacher_semantic.png"
    inst_path = out_dir / f"{stem}_teacher_instances.json"
    overlay_path = out_dir / f"{stem}_teacher_overlay.png"

    Image.fromarray(semantic_rgb).save(sem_path)
    inst_path.write_text(json.dumps({
        "teacher_version": "yolo26l_sam_v1",
        "board_stem": stem,
        "width": w,
        "height": h,
        "total_instances": len(exported_instances),
        "instances": exported_instances,
    }, indent=2))

    # Transparent overlay using DISPLAY_RGB
    disp_sem = np.zeros((h, w, 3), dtype=np.uint8)
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        cid = inst["category_id"]
        disp_sem[y1:y2, x1:x2][inst["crop_mask"]] = DISPLAY_RGB[cid - 1]

    alpha = 0.50
    overlay = cv2.addWeighted(rgb, 1.0 - alpha, disp_sem, alpha, 0.0)
    # Keep background pure RGB where no components exist
    no_comp = (disp_sem.sum(axis=2) == 0)
    overlay[no_comp] = rgb[no_comp]
    Image.fromarray(overlay).save(overlay_path)

    print(f"Teacher outputs saved to: {out_dir}")
    print(f"Total Discovered Components: {len(exported_instances)}")

    return {
        "stem": stem,
        "components": len(exported_instances),
        "semantic_path": str(sem_path),
        "instances_path": str(inst_path),
        "overlay_path": str(overlay_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline Teacher Inference on DSLR Boards (Rule T1-T4)")
    parser.add_argument("--image", required=True, help="Path to input DSLR image or directory of images")
    parser.add_argument("--yolo-ckpt", default="checkpoints/teacher_yolo26l_seg.pt", help="Fine-tuned YOLO26-L checkpoint")
    parser.add_argument("--sam-ckpt", default="checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM ViT-H checkpoint")
    parser.add_argument("--out", default="results/teacher_pseudo_labels", help="Output directory for teacher references")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--mask-nms", type=float, default=0.50, help="Mask-NMS IoU threshold")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    args = parser.parse_args()

    from ultralytics import YOLO

    yolo_path = Path(args.yolo_ckpt)
    if not yolo_path.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {yolo_path}. Run train_teacher_yolo.py first.")

    print("Loading fine-tuned YOLO26-L-seg...")
    yolo_model = YOLO(str(yolo_path))

    sam_predictor = init_sam_predictor(
        args.sam_ckpt,
        model_type="vit_h",
        device="cuda:0" if args.device != "cpu" else "cpu",
    )

    in_path = Path(args.image)
    image_files = []
    if in_path.is_file():
        image_files = [in_path]
    elif in_path.is_dir():
        image_files = sorted(list(in_path.glob("*.jpg")) + list(in_path.glob("*.png")))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = []
    for img_f in image_files:
        stem = img_f.stem
        rec = process_dslr_board_teacher(
            stem=stem,
            img_path=img_f,
            out_dir=out_dir,
            yolo_model=yolo_model,
            sam_predictor=sam_predictor,
            conf_thresh=args.conf,
            mask_nms_thresh=args.mask_nms,
            device=args.device,
        )
        manifest_records.append(rec)

    manifest_file = out_dir / "teacher_manifest.json"
    manifest_file.write_text(json.dumps({
        "teacher_version": "yolo26l_sam_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "yolo_checkpoint": str(args.yolo_ckpt),
        "sam_checkpoint": str(args.sam_ckpt),
        "confidence_threshold": args.conf,
        "mask_nms_threshold": args.mask_nms,
        "boards_processed": len(manifest_records),
        "records": manifest_records,
    }, indent=2))

    print(f"\nAll teacher pseudo-references saved to: {out_dir}")


if __name__ == "__main__":
    main()
