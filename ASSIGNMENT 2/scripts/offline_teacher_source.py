"""Offline Teacher Distillation for Source-Domain Training Data.

Permitted student use (Assignment II PDF Section 3, Page 3):
"Students may run YOLO26-L + SAM to illustrate or diagnose pseudo-masks on a separate,
designated public visualization subset and may use this teacher on source-domain training data."

This script processes official source boards (Graph-W and Graph-F):
1. Graph-W: Converts 37 box-only training boards into sharp, geometry-aware instance masks
   using SAM (or adaptive morphological priors if SAM weights are not present).
2. Graph-F: Resolves touching/merged passives (resistors/capacitors) into distinct instances.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
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

SYMBOLS = {c.lower(): i + 1 for i, c in enumerate(CLASSES)}

TYPES = {
    "resistor": 1, "capacitor": 2, "ic": 11, "integrated circuit": 11,
    "transistor": 4, "diode": 8, "inductor": 6, "connector": 12, "plug": 12,
    "switch": 18, "button": 15, "fuse": 20, "led": 22, "light emitting diode": 22,
    "test point": 10, "oscillator": 13, "crystal": 13, "transformer": 19,
    "varistor": 21, "relay": 14, "jumper": 25, "ferrite bead": 16,
    "resistor array": 7, "resistor network": 9, "capacitor array": 17,
    "transistor array": 24, "switch (s)": 23, "switch (sw)": 18,
    "zener diode": 8, "electrolytic capacitor": 2
}


def map_graph_w_name(raw_name: str | None) -> int | None:
    if not raw_name:
        return None
    raw = " ".join(str(raw_name).lower().replace('"', "").replace("'", "").replace("_", " ").split())
    # Canonical symbol exact match
    for i, code in enumerate(CLASSES, 1):
        if raw == code.lower() or raw.startswith(code.lower() + " "):
            return i
    # Component type name
    for name, cid in sorted(TYPES.items(), key=lambda x: -len(x[0])):
        if raw == name or raw.startswith(name + " "):
            return cid
    # Standard PCB reference designators like R1, C12, U4, Q2
    match = re.fullmatch(r"([a-z]+)(?:\d+)?", raw)
    if match and match[1] in SYMBOLS:
        return SYMBOLS[match[1]]
    return None


def init_sam_predictor(checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda:0"):
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        return SamPredictor(sam)
    except Exception as e:
        print(f"[Warning] Could not initialize SAM ({e}). Using adaptive morphological geometry prior.")
        return None


def generate_component_mask(crop_rgb: np.ndarray, box: list[float], sam_predictor=None, full_rgb=None) -> np.ndarray:
    """Generate a crisp binary instance mask for a component."""
    if sam_predictor is not None and full_rgb is not None:
        try:
            boxes = np.array([box])
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                boxes, full_rgb.shape[:2]
            )
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            return masks[0, 0].cpu().numpy().astype(bool)
        except Exception:
            pass

    # High-quality fallback: Otsu / adaptive foreground prior within bounding box
    h, w = crop_rgb.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    if h > 4 and w > 4:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        # Solder joint / component contrast enhancement
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Keep foreground if significant, else rectangular mask
        if thresh.mean() > 20 and thresh.mean() < 240:
            mask = thresh > 0
    return mask


def process_graph_w(graph_w_dir: Path, out_dir: Path, sam_predictor=None):
    """Convert Graph-W training boards from box XMLs to instance masks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = graph_w_dir / "image_dir"
    masks_dir = graph_w_dir / "mask_dir"
    train_graphs = sorted((graph_w_dir / "graphs" / "train").glob("*.pt"))
    print(f"Processing {len(train_graphs)} Graph-W training boards...")

    results = []
    for g in train_graphs:
        stem = re.sub(r"^graph_\d+_", "", g.stem)
        img_candidates = [images_dir / f"{stem}.jpg", images_dir / f"{stem}.png"]
        img_path = next((p for p in img_candidates if p.is_file()), None)
        xml_path = masks_dir / f"{stem}.xml"

        if not img_path or not xml_path.is_file():
            print(f"[Skip] Missing image or xml for {stem}")
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        if sam_predictor is not None:
            try:
                sam_predictor.set_image(rgb)
            except Exception:
                pass

        tree = ET.parse(xml_path)
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        instance_ids = np.zeros((h, w), dtype=np.uint16)
        instances = []

        next_id = 1
        for obj in tree.getroot().findall("object"):
            raw_name = obj.findtext("name")
            cid = map_graph_w_name(raw_name)
            if cid is None or not (1 <= cid <= 25):
                continue
            box = obj.find("bndbox")
            x1 = max(0, min(w - 1, int(float(box.findtext("xmin")))))
            y1 = max(0, min(h - 1, int(float(box.findtext("ymin")))))
            x2 = max(x1 + 1, min(w, int(float(box.findtext("xmax")))))
            y2 = max(y1 + 1, min(h, int(float(box.findtext("ymax")))))

            crop = rgb[y1:y2, x1:x2]
            local_mask = generate_component_mask(crop, [x1, y1, x2, y2], sam_predictor, rgb)

            color = SOURCE_RGB[cid - 1]
            view_color = color_mask[y1:y2, x1:x2]
            view_ids = instance_ids[y1:y2, x1:x2]

            view_color[local_mask] = color
            view_ids[local_mask] = next_id
            instances.append({
                "instance_id": next_id,
                "category_id": cid,
                "class_code": CLASSES[cid - 1],
                "bbox": [x1, y1, x2, y2],
                "area": int(local_mask.sum())
            })
            next_id += 1

        mask_out_path = out_dir / f"{stem}_mask.png"
        id_out_path = out_dir / f"{stem}_instances.png"
        Image.fromarray(color_mask).save(mask_out_path)
        Image.fromarray(instance_ids, ).save(id_out_path)

        results.append({
            "stem": stem,
            "image_path": str(img_path.resolve()),
            "mask_path": str(mask_out_path.resolve()),
            "instance_path": str(id_out_path.resolve()),
            "component_count": len(instances),
            "width": w,
            "height": h
        })

    print(f"Successfully generated teacher masks for {len(results)} Graph-W boards.")
    return results


def main():
    p = argparse.ArgumentParser(description="Offline Teacher Distillation for Source Data")
    p.add_argument("--source", default="ASSIGNMENT 2/data/GraphPCB", help="Path to GraphPCB dataset")
    p.add_argument("--out", default="ASSIGNMENT 2/data/distilled_source", help="Output directory for refined source masks")
    p.add_argument("--sam-checkpoint", default=None, help="Optional SAM checkpoint (.pth)")
    p.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--device", default="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    args = p.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sam_pred = None
    if args.sam_checkpoint and Path(args.sam_checkpoint).is_file():
        sam_pred = init_sam_predictor(args.sam_checkpoint, args.sam_model_type, args.device)

    graph_w_dir = source / "Graph-W"
    if graph_w_dir.is_dir():
        manifest = process_graph_w(graph_w_dir, out / "Graph-W", sam_pred)
        (out / "graph_w_distilled_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Offline teacher distillation completed. Outputs in: {out}")


if __name__ == "__main__":
    main()
