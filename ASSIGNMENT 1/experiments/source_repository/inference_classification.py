#!/usr/bin/env python3
"""
Standalone inference script for PCB Component Classification.
Contains all model construction, preprocessing, forward pass, and visualization logic.
NO project-specific internal imports permitted; only standard library and external packages.

Usage:
  python inference_classification.py --image <path_to_image> --checkpoint checkpoints/best_classification.ckpt --output results/classification/pred_sample.png
"""

import os
import sys
import argparse
import math
import json
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont

# Canonical 25-Class Taxonomy and Colors from Assignment Specifications
TAXONOMY_25 = [
    {"id": 1, "name": "Resistor (R)", "symbol": "R", "color": "#E53935", "rgb": (229, 57, 53)},
    {"id": 2, "name": "Capacitor (C)", "symbol": "C", "color": "#1E88E5", "rgb": (30, 136, 229)},
    {"id": 3, "name": "Integrated Circuit (U)", "symbol": "U", "color": "#00ACC1", "rgb": (0, 172, 193)},
    {"id": 4, "name": "Discrete Transistor (Q)", "symbol": "Q", "color": "#8E24AA", "rgb": (142, 36, 170)},
    {"id": 5, "name": "Connector (J)", "symbol": "J", "color": "#FB8C00", "rgb": (251, 140, 0)},
    {"id": 6, "name": "Inductor (L)", "symbol": "L", "color": "#43A047", "rgb": (67, 160, 71)},
    {"id": 7, "name": "Resistor Coil (RA)", "symbol": "RA", "color": "#6D4C41", "rgb": (109, 76, 65)},
    {"id": 8, "name": "Diode (D)", "symbol": "D", "color": "#D81B60", "rgb": (216, 27, 96)},
    {"id": 9, "name": "Resistor Network (RN)", "symbol": "RN", "color": "#3949AB", "rgb": (57, 73, 171)},
    {"id": 10, "name": "Test Point (TP)", "symbol": "TP", "color": "#C0CA33", "rgb": (192, 202, 51)},
    {"id": 11, "name": "Integrated Circuit (IC)", "symbol": "IC", "color": "#00897B", "rgb": (0, 137, 123)},
    {"id": 12, "name": "Plug (P)", "symbol": "P", "color": "#FFB300", "rgb": (255, 179, 0)},
    {"id": 13, "name": "Thyristor (CR)", "symbol": "CR", "color": "#5E35B1", "rgb": (94, 53, 177)},
    {"id": 14, "name": "Motor (M)", "symbol": "M", "color": "#039BE5", "rgb": (3, 155, 229)},
    {"id": 15, "name": "Button (BTN)", "symbol": "BTN", "color": "#757575", "rgb": (117, 117, 117)},
    {"id": 16, "name": "Ferrite Bead (FB)", "symbol": "FB", "color": "#FDD835", "rgb": (253, 216, 53)},
    {"id": 17, "name": "CRA", "symbol": "CRA", "color": "#212121", "rgb": (33, 33, 33)},
    {"id": 18, "name": "Switch (SW)", "symbol": "SW", "color": "#7E57C2", "rgb": (126, 87, 194)},
    {"id": 19, "name": "Transformer (T)", "symbol": "T", "color": "#F4511E", "rgb": (244, 81, 30)},
    {"id": 20, "name": "Fuse (F)", "symbol": "F", "color": "#2E7D32", "rgb": (46, 125, 50)},
    {"id": 21, "name": "Vacuum Tube (V)", "symbol": "V", "color": "#283593", "rgb": (40, 53, 147)},
    {"id": 22, "name": "Light Emitting Diode (LED)", "symbol": "LED", "color": "#7CB342", "rgb": (124, 179, 66)},
    {"id": 23, "name": "Switch (S)", "symbol": "S", "color": "#546E7A", "rgb": (84, 110, 122)},
    {"id": 24, "name": "QA", "symbol": "QA", "color": "#C2185B", "rgb": (194, 24, 91)},
    {"id": 25, "name": "Jumper Link (JP)", "symbol": "JP", "color": "#F9A825", "rgb": (249, 168, 37)},
]

ID_TO_CLASS = {item["id"]: item for item in TAXONOMY_25}


def build_standalone_classifier(model_name: str = "efficientnet_b0", num_classes: int = 25) -> nn.Module:
    """Build standalone classification model matching training architecture."""
    model_name = model_name.lower()
    if "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif "efficientnet_b2" in model_name:
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif "mobilenet_v3_large" in model_name:
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif "convnext_tiny" in model_name:
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        # Default fallback
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, str, float]:
    """Load trained checkpoint and return model, model_name, and frozen threshold."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    threshold = checkpoint.get("threshold", 0.5)

    model = build_standalone_classifier(model_name=model_name, num_classes=25)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name, threshold


def classify_crop(
    model: nn.Module,
    crop_image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[int, float]:
    """Run forward pass on a single component crop. Returns (class_id_1_indexed, confidence)."""
    tensor = transform(crop_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx_0 = int(np.argmax(probs))
    conf = float(probs[pred_idx_0])
    return pred_idx_0 + 1, conf


def auto_detect_or_lookup_boxes(image_path: str, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    """
    Look up ground-truth or dataset annotations for this board image from:
    1) Annotation XML in mask_dir if Graph-W image
    2) data/dataset_manifest.json
    3) Fallback: whole-board box [(0, 0, w, h)]
    """
    boxes = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    clean_base = base_name.replace("_image", "")

    # 1. Check XML in mask_dir if Graph-W image
    possible_xml_paths = [
        image_path.replace("image_dir", "mask_dir").replace(".jpg", ".xml").replace(".png", ".xml"),
        os.path.join(os.path.dirname(image_path), "..", "mask_dir", f"{base_name}.xml"),
        os.path.join(os.path.dirname(image_path), "..", "mask_dir", f"{clean_base}.xml"),
        os.path.join("data", "GraphPCB", "Graph-W", "mask_dir", f"{base_name}.xml"),
        os.path.join("data", "GraphPCB", "Graph-W", "mask_dir", f"{clean_base}.xml"),
    ]
    for xml_path in possible_xml_paths:
        if os.path.exists(xml_path):
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_path)
                for obj in tree.findall("object"):
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)
                        if xmax > xmin and ymax > ymin:
                            boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                if boxes:
                    print(f"Auto-loaded {len(boxes)} component boxes from annotation XML: {xml_path}")
                    return boxes
            except Exception:
                pass

    # 2. Check dataset_manifest.json
    manifest_path = "data/dataset_manifest.json"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            all_crops = manifest.get("train_crops", []) + manifest.get("val_crops", []) + manifest.get("test_crops", [])
            for c in all_crops:
                c_img = c.get("image_path", "")
                if base_name in c_img or clean_base in c_img:
                    bx = c.get("bbox")
                    if bx and len(bx) == 4:
                        boxes.append((int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])))
            if boxes:
                print(f"Auto-loaded {len(boxes)} component boxes from dataset manifest: {manifest_path}")
                return boxes
        except Exception:
            pass

    # 3. Fallback: single board crop
    print(f"Notice: No component box annotations found for {image_path}. Running full image crop.")
    return [(0, 0, w, h)]


def run_full_board_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    threshold: Optional[float] = None,
) -> List[Dict]:
    """
    Run classification on a full PCB board with component boxes.
    Outputs annotated visualization and returns structured detections.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, default_threshold = load_model(checkpoint_path, device)
    effective_threshold = threshold if threshold is not None else default_threshold
    transform = get_inference_transform(224)

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    draw = ImageDraw.Draw(image)

    # If no boxes provided, auto-lookup component boxes
    if not boxes:
        boxes = auto_detect_or_lookup_boxes(image_path, w, h)

    # Adaptive font and line thickness based on image resolution
    line_w = max(2, int(min(w, h) / 600))
    font_size = max(11, int(min(w, h) / 120))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    results = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        class_id, conf = classify_crop(model, crop, transform, device)

        class_info = ID_TO_CLASS.get(class_id, TAXONOMY_25[0])
        rgb = class_info["rgb"]
        symbol = class_info["symbol"]
        class_name = class_info["name"]

        results.append({
            "bbox": [x1, y1, x2, y2],
            "class_id": class_id,
            "class_name": class_name,
            "symbol": symbol,
            "confidence": round(conf, 4),
        })

        # Draw box and label if above threshold
        if conf >= effective_threshold:
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=line_w)
            label_text = f"{symbol} {conf:.2f}"
            if font is not None:
                try:
                    tb = draw.textbbox((0, 0), label_text, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = len(label_text) * 8, 14
            else:
                tw, th = len(label_text) * 8, 14

            banner_y1 = max(0, y1 - th - 4)
            banner_y2 = max(th + 4, y1)
            draw.rectangle([x1, banner_y1, x1 + tw + 6, banner_y2], fill=rgb)
            draw.text((x1 + 3, banner_y1 + 1), label_text, fill=(255, 255, 255), font=font)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(output_path)
    print(f"Saved visualization to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone PCB Component Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input PCB image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_classification.ckpt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="results/classification/prediction.png", help="Path to output visualization")
    parser.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold")
    parser.add_argument("--boxes_json", type=str, default=None, help="Optional JSON file with list of [x1, y1, x2, y2] bounding boxes")
    args = parser.parse_args()

    boxes = None
    if args.boxes_json and os.path.exists(args.boxes_json):
        with open(args.boxes_json, "r") as f:
            boxes = json.load(f)

    detections = run_full_board_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        boxes=boxes,
        threshold=args.threshold,
    )
    print(f"Processed {len(detections)} components.")


if __name__ == "__main__":
    main()

