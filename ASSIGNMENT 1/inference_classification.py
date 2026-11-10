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
import hashlib
from pathlib import Path
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


def build_standalone_classifier(model_name: str = "efficientnet_b0", num_classes: int = 25, dropout_rate: float = .3) -> nn.Module:
    """Build standalone classification model matching training architecture."""
    model_name = model_name.lower()
    if "efficientnet_b0" in model_name:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif "efficientnet_b2" in model_name:
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
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
        raise ValueError(f"Unsupported architecture: {model_name}")
    return model


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def validated_bounds(bbox, size):
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
        raise ValueError("Invalid box")
    x1,y1,x2,y2 = map(float,bbox)
    w,h = size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Box outside image or degenerate")
    return math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2)

def validate_checkpoint_contract(checkpoint):
    meta = checkpoint.get("metadata", {})
    schema = meta.get("schema")
    if schema not in {"classification-checkpoint-v2", "classification-checkpoint-legacy-export-v1"}:
        raise ValueError("Declared checkpoint metadata required; unknown legacy preprocessing is not assumed")
    if meta["class_names"] != [c["name"] for c in TAXONOMY_25]:
        raise ValueError("Canonical class order mismatch")
    c = meta["preprocessing"]
    expected = "rgb-strict-floor-ceil-bilinear-v2" if schema == "classification-checkpoint-v2" else "rgb-legacy-int-bilinear-v1"
    if c["version"] != expected or c["mean"] != [.485,.456,.406] or c["std"] != [.229,.224,.225]:
        raise ValueError("Preprocessing contract mismatch")
    if schema == "classification-checkpoint-legacy-export-v1" and not meta.get("source_checkpoint_sha256"):
        raise ValueError("Legacy export must identify its original checkpoint")
    if checkpoint["decision_policy"]["reject_label"] != -1 or not 0 <= checkpoint["threshold"] <= 1:
        raise ValueError("Decision policy mismatch")

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, str, float]:
    """Load trained checkpoint and return model, model_name, and frozen threshold."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_contract(checkpoint)
    model_name = checkpoint["model_name"]
    threshold = checkpoint["threshold"]

    model = build_standalone_classifier(model_name=model_name, num_classes=25, dropout_rate=checkpoint['metadata']['effective_config']['dropout_rate'])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    model.inference_metadata = checkpoint["metadata"]
    return model, model_name, threshold


def classify_crop(
    model: nn.Module,
    crop_image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
    return_probabilities: bool = False,
) -> Tuple[int, float]:
    """Run forward pass on a single component crop. Returns (class_id_1_indexed, confidence)."""
    tensor = transform(crop_image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx_0 = int(np.argmax(probs))
    conf = float(probs[pred_idx_0])
    if return_probabilities:
        return pred_idx_0 + 1, conf, probs.tolist()
    return pred_idx_0 + 1, conf


def run_full_board_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    threshold: Optional[float] = None,
    single_crop: bool = False,
) -> List[Dict]:
    """
    Run classification on a full PCB board with component boxes.
    Outputs annotated visualization and returns structured detections.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, default_threshold = load_model(checkpoint_path, device)
    effective_threshold = threshold if threshold is not None else default_threshold
    if not 0 <= effective_threshold <= 1:
        raise ValueError("Threshold outside [0,1]")
    transform = get_inference_transform(model.inference_metadata["preprocessing"]["img_size"])

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    # If no boxes provided, run on the whole image as a single crop
    if single_crop:
        if boxes is not None:
            raise ValueError("single-crop and boxes are mutually exclusive")
        boxes = [(0, 0, w, h)]
    elif not boxes:
        raise ValueError("Full-board inference requires annotated/derived boxes; use --single-crop explicitly for a crop")

    results = []
    for x1, y1, x2, y2 in boxes:
        original_bounds = [x1,y1,x2,y2]
        checked = validated_bounds(original_bounds, (w,h))
        if model.inference_metadata["preprocessing"]["version"] == "rgb-legacy-int-bilinear-v1":
            # Match the historical dataset's integer truncation for valid boxes.
            x1,y1,x2,y2 = map(int, original_bounds)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Box becomes degenerate after legacy integer truncation")
        else:
            x1,y1,x2,y2 = checked
        crop = image.crop((x1, y1, x2, y2))
        class_id, conf, probabilities = classify_crop(model, crop, transform, device, return_probabilities=True)

        class_info = ID_TO_CLASS.get(class_id, TAXONOMY_25[0])
        rgb = class_info["rgb"]
        symbol = class_info["symbol"]
        class_name = class_info["name"]

        results.append({
            "bbox": [x1, y1, x2, y2],
            "sample_id": hashlib.sha256((str(Path(image_path).resolve())+str([x1,y1,x2,y2])).encode()).hexdigest(),
            "image_path": str(Path(image_path).resolve()),
            "probabilities": probabilities,
            "class_id": class_id,
            "argmax_prediction": class_id - 1,
            "thresholded_prediction": class_id - 1 if conf >= effective_threshold else -1,
            "accepted": conf >= effective_threshold,
            "threshold": effective_threshold,
            "class_name": class_name,
            "symbol": symbol,
            "confidence": conf,
        })

        # Draw box and label if above threshold
        if conf >= effective_threshold:
            draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)
            label_text = f"{symbol} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 15), x1 + len(label_text) * 8, y1], fill=rgb)
            draw.text((x1 + 2, max(0, y1 - 14)), label_text, fill=(255, 255, 255))

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if Path(output_path).exists() or Path(str(output_path)+".json").exists():
        raise FileExistsError(output_path)
    canvas.save(output_path)
    Path(str(output_path)+".json").write_text(json.dumps({"predictions":results,"metadata":model.inference_metadata,"checkpoint_sha256":hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()},indent=2),encoding="utf8")
    print(f"Saved visualization to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone PCB Component Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input PCB image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_classification.ckpt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="results/classification/prediction.png", help="Path to output visualization")
    parser.add_argument("--threshold", type=float, default=None, help="Optional confidence threshold")
    parser.add_argument("--single-crop", action="store_true")
    parser.add_argument("--boxes_json", "--boxes-json", type=str, default=None, help="Optional JSON file with list of [x1, y1, x2, y2] bounding boxes")
    args = parser.parse_args()

    boxes = None
    if args.boxes_json:
        with open(args.boxes_json, "r") as f:
            boxes = json.load(f)

    detections = run_full_board_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        boxes=boxes,
        threshold=args.threshold,
        single_crop=args.single_crop,
    )
    print(f"Processed {len(detections)} components.")


if __name__ == "__main__":
    main()

