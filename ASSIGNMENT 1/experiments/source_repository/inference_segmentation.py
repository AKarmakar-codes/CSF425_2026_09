#!/usr/bin/env python3
"""
Standalone inference script for PCB Component Instance Segmentation.
Contains all model construction, preprocessing, forward pass, and translucent mask overlay logic.
NO project-specific internal imports permitted; only standard library and external packages.

Usage:
  python inference_segmentation.py --image <path_to_image> --checkpoint checkpoints/best_segment.ckpt --output results/segmentation/pred_sample.png
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from PIL import Image
import cv2

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


def build_standalone_maskrcnn(num_classes: int = 26) -> nn.Module:
    """Build Mask R-CNN model structure (25 component classes + 1 background)."""
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def run_segmentation_inference(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    conf_threshold: float = 0.20,
    alpha: float = 0.5,
) -> List[Dict]:
    """Run instance segmentation and save translucent mask overlay."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # Detect model type: YOLO vs Mask R-CNN
    is_yolo = False
    if checkpoint_path.endswith(".pt") or "yolo" in checkpoint_path.lower() or "best_segment" in checkpoint_path.lower():
        is_yolo = True
    elif os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                is_yolo = False
            elif isinstance(ckpt, dict) and "model" in ckpt:
                is_yolo = True
        except Exception:
            pass

    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    if is_yolo:
        import tempfile
        import shutil
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLO instance segmentation checkpoint detected, but 'ultralytics' is not installed in the active environment.\n"
                "Please run: pip install ultralytics\n"
                "Or provide a Mask R-CNN checkpoint via --checkpoint <path_to_maskrcnn.ckpt>"
            )
        
        yolo_path = checkpoint_path
        if not yolo_path.endswith(".pt"):
            temp_pt = os.path.join(tempfile.gettempdir(), "best_segment_yolo.pt")
            shutil.copyfile(checkpoint_path, temp_pt)
            yolo_path = temp_pt

        yolo_model = YOLO(yolo_path, task="segment")
        yolo_res = yolo_model.predict(
            source=image_path,
            conf=conf_threshold,
            max_det=500,
            device=device,
            verbose=False,
        )[0]

        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            scores = yolo_res.boxes.conf.cpu().numpy()
            labels = yolo_res.boxes.cls.cpu().numpy().astype(int) + 1  # 1-indexed

            if yolo_res.masks is not None:
                raw_masks = yolo_res.masks.data.cpu().numpy()
                masks = []
                for m in raw_masks:
                    m_resized = cv2.resize(m.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    masks.append((m_resized > 0.5).astype(np.uint8))
                masks = np.array(masks)
            else:
                # Bounding box fallback
                masks = []
                for b in boxes:
                    m = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = 1
                    masks.append(m)
                masks = np.array(masks)
        else:
            boxes, scores, labels, masks = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)), np.zeros((0, orig_h, orig_w))
    else:
        # Mask R-CNN
        model = build_standalone_maskrcnn(num_classes=26)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running initialized model.")

        model.to(device)
        model.eval()

        img_tensor = transforms.functional.to_tensor(pil_img).to(device)
        with torch.no_grad():
            preds = model([img_tensor])[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        masks = (preds["masks"].squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)

        # Filter by confidence
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        masks = masks[keep]

    # Render translucent overlays
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        img_cv = np.array(pil_img)[:, :, ::-1]

    overlay = img_cv.copy()
    detections = []

    for mask, cid, score, box in zip(masks, labels, scores, boxes):
        cid = int(cid)
        class_info = ID_TO_CLASS.get(cid, TAXONOMY_25[0])
        r, g, b = class_info["rgb"]
        bgr_color = (b, g, r)
        symbol = class_info["symbol"]
        class_name = class_info["name"]

        bool_mask = mask.astype(bool)
        overlay[bool_mask] = bgr_color

        # Outline
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_cv, contours, -1, bgr_color, 2)

        # Label text
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

        label_text = f"{symbol} {score:.2f}"
        cv2.putText(img_cv, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "bbox": [float(x) for x in box],
            "class_id": cid,
            "class_name": class_name,
            "symbol": symbol,
            "confidence": round(float(score), 4),
        })

    blended = cv2.addWeighted(overlay, alpha, img_cv, 1.0 - alpha, 0)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, blended)
    print(f"Saved translucent segmentation visualization to: {output_path}")
    return detections


def main():
    parser = argparse.ArgumentParser(description="Standalone PCB Instance Segmentation Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input PCB image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_segment.ckpt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="results/segmentation/prediction.png", help="Path to output visualization")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    detections = run_segmentation_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        conf_threshold=args.threshold,
    )
    print(f"Detected {len(detections)} component instances.")


if __name__ == "__main__":
    main()

