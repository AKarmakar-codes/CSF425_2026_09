"""
Complete data preparation and ingestion pipeline for PCB Deep Learning Assignment.
Extracts component instances from Graph-W and Graph-F datasets, assigns canonical 25 classes,
and generates manifests for Exercise 1 (Classification) and Exercise 2 (Instance Segmentation).
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# FPIC color mapping for Graph-F to 25 canonical classes
FPIC_COLOR_TO_CLASS = {
    (255, 0, 0): 1,        # R - Resistor (R)
    (255, 255, 0): 2,      # C - Capacitor (C)
    (0, 234, 255): 3,      # U - Integrated Circuit (U)
    (170, 0, 255): 4,      # Q - Discrete Transistor (Q)
    (255, 127, 0): 5,      # J - Connector (J)
    (191, 255, 0): 6,      # L - Inductor (L)
    (0, 149, 255): 7,      # RA - Resistor Coil (RA)
    (106, 255, 0): 8,      # D - Diode (D)
    (0, 64, 255): 9,       # RN - Resistor Network (RN)
    (237, 185, 185): 10,   # TP - Test Point (TP)
    (185, 215, 237): 11,   # IC - Integrated Circuit (IC)
    (231, 233, 185): 12,   # P - Plug (P)
    (220, 185, 237): 13,   # CR - Thyristor (CR)
    (185, 237, 224): 14,   # M - Motor (M)
    (143, 35, 35): 15,     # BTN - Button (BTN)
    (35, 98, 143): 16,     # FB - Ferrite Bead (FB)
    (143, 106, 35): 17,    # CRA - CRA
    (107, 35, 143): 18,    # SW - Switch (SW)
    (79, 143, 35): 19,     # T - Transformer (T)
    (115, 115, 115): 20,   # F - Fuse (F)
    (204, 204, 204): 21,   # V - Vacuum Tube (V)
    (245, 130, 48): 22,    # LED - Light Emitting Diode (LED)
    (220, 190, 255): 23,   # S - Switch (S)
    (170, 255, 195): 24,   # QA - QA
    (255, 250, 200): 25,   # JP - Jumper Link (JP)
}


def map_graph_w_name(raw_name: str) -> Optional[int]:
    """
    Map Graph-W XML object name to canonical 25 classes. Returns None if ignore/void.
    Handles component value suffixes (e.g. 'capacitor unknown') without dropping valid components,
    and correctly differentiates Jumpers (25) vs Connectors (5), IC-U (3) vs IC-IC (11),
    and Switch-SW (18) vs Switch-S (23).
    """
    import re
    raw = raw_name.lower().strip().strip('"').strip("'")

    # 1. Silkscreen text and board feature blacklist
    if raw.startswith("text") or raw.startswith("pins") or raw.startswith("pads"):
        return None
    if raw.startswith("component text"):
        return None
    if any(k in raw for k in ["hole", "heatsink", "logo"]):
        return None
    # Truly unidentifiable objects without component class
    if raw in ["unknown", "unknown unknown", "unknown unlabeled", "unknown na", "unlabeled"]:
        return None

    # 2. Multi-word and specific patterns FIRST (order matters!)
    if "resistor network" in raw or raw.startswith("rn") or "resnetw" in raw:
        return 9
    elif "resistor coil" in raw or raw.startswith("ra"):
        return 7
    elif "ferrite bead" in raw or raw.startswith("fb") or "emi filter" in raw:
        return 16
    elif raw.startswith("cra"):
        return 17
    elif "thyristor" in raw or raw.startswith("cr"):
        return 13
    elif "light emitting diode" in raw or raw.startswith("led") or " led" in raw or raw.startswith("ld"):
        return 22

    # 3. Jumper Links (25) - MUST be checked before generic connector / 'j' prefix!
    if "jumper" in raw or raw.startswith("jp"):
        return 25

    # 4. Switches: SW (18) vs S (23)
    if "switch" in raw or raw.startswith("sw"):
        if re.search(r"\bs\d+\b", raw):
            return 23
        return 18

    # 5. Connectors (5)
    if "connector" in raw or raw.startswith("j") or "cna" in raw:
        return 5

    # 6. Capacitors (2) (including electrolytic capacitors) - MUST be checked BEFORE ICs!
    # Prevents "electrolytic" from matching substring "ic " inside "electrolytIC "
    if "capacitor" in raw or "electrolytic" in raw:
        return 2
    if raw.startswith("c") and not raw.startswith("cr") and not raw.startswith("cn") and not raw.startswith("cl"):
        return 2

    # 7. ICs: Integrated Circuit (U) [3] vs Integrated Circuit (IC) [11]
    # Use word boundary to avoid matching inside other words
    if re.search(r"\b(ic|integrated circuit)\b", raw) or raw == "ic":
        if re.search(r"\bu\d+\b", raw) or " u" in raw:
            return 3
        return 11
    if raw.startswith("u") or re.search(r"\bu\d+\b", raw):
        return 3

    # 8. Resistors (1) (including potentiometers)
    if "resistor" in raw or "potentiometer" in raw:
        return 1
    if raw.startswith("r") and not raw.startswith("rn") and not raw.startswith("ra"):
        return 1

    # 9. Discrete Transistors (4)
    if "transistor" in raw or raw.startswith("q"):
        return 4

    # 10. Inductors (6)
    if "inductor" in raw or (raw.startswith("l") and not raw.startswith("led") and not raw.startswith("ld")):
        return 6

    # 11. Diodes (8) (including zener diodes and diode arrays)
    if "diode" in raw or "zener" in raw or raw.startswith("d"):
        return 8

    # 12. Test Points (10)
    if "test point" in raw or raw.startswith("tp") or raw.startswith("test"):
        return 10

    # 13. Remaining canonical 25-class assignments
    if "plug" in raw or raw.startswith("p"):
        return 12
    if "motor" in raw or "buzzer" in raw or raw.startswith("m"):
        return 14
    if "button" in raw or raw.startswith("btn"):
        return 15
    if "transformer" in raw or (raw.startswith("t") and "transistor" not in raw):
        return 19
    if "fuse" in raw or raw.startswith("f"):
        return 20
    if "vacuum tube" in raw or raw.startswith("v"):
        return 21
    if "qa" in raw or raw.startswith("qa"):
        return 24

    return None


def get_official_split_names(dataset_root: str) -> Dict[str, Dict[str, set]]:
    """Get the set of base board names for train and test from the graph split files."""
    splits = {"Graph-W": {"train": set(), "test": set()}, "Graph-F": {"train": set(), "test": set()}}

    for ds in ["Graph-W", "Graph-F"]:
        for split in ["train", "test"]:
            graph_dir = os.path.join(dataset_root, ds, "graphs", split)
            if os.path.exists(graph_dir):
                for f in os.listdir(graph_dir):
                    if f.endswith(".pt"):
                        # Extract clean board name
                        base = f.replace("graph_", "").replace(".pt", "")
                        parts = base.split("_", 1)
                        clean = parts[1] if len(parts) > 1 and parts[0].isdigit() else base
                        splits[ds][split].add(clean)
    return splits


def parse_graph_w(dataset_root: str, split_names: Dict[str, set]) -> Dict[str, List[Dict]]:
    """Parse Graph-W boards from image_dir and mask_dir."""
    img_dir = os.path.join(dataset_root, "Graph-W", "image_dir")
    mask_dir = os.path.join(dataset_root, "Graph-W", "mask_dir")
    boards = {"train": [], "test": []}

    train_names = split_names["Graph-W"]["train"]

    for xml_file in sorted(os.listdir(mask_dir)):
        if not xml_file.endswith(".xml"):
            continue
        base_name = xml_file.replace(".xml", "")
        img_name = f"{base_name}.jpg"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        tree = ET.parse(os.path.join(mask_dir, xml_file))
        root = tree.getroot()

        instances = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text
            cid = map_graph_w_name(raw_name)
            if cid is None:
                continue

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                instances.append({
                    "class_id": cid,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "raw_name": raw_name,
                })

        split_key = "train" if any(k in base_name for k in train_names) else "test"
        boards[split_key].append({
            "board_id": base_name,
            "dataset": "Graph-W",
            "image_path": img_path,
            "annotations": instances,
        })

    return boards


FPIC_PACKED_TO_CLASS = {
    (r << 16) | (g << 8) | b: cid for (r, g, b), cid in FPIC_COLOR_TO_CLASS.items()
}


def parse_graph_f(dataset_root: str, split_names: Dict[str, set]) -> Dict[str, List[Dict]]:
    """Parse Graph-F boards from image_dir and mask_dir using connected components on RGB masks."""
    img_dir = os.path.join(dataset_root, "Graph-F", "image_dir")
    mask_dir = os.path.join(dataset_root, "Graph-F", "mask_dir")
    boards = {"train": [], "test": []}

    train_names = split_names["Graph-F"]["train"]

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith("_mask.png")])
    print(f"Parsing {len(mask_files)} Graph-F board masks...")

    for mask_file in tqdm(mask_files, desc="Parsing Graph-F masks"):
        base_name = mask_file.replace("_mask.png", "")
        img_name = f"{base_name}_image.png"
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        mask_path = os.path.join(mask_dir, mask_file)
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # Fast bit-packed 24-bit representation
        mask_packed = (mask_rgb[:, :, 0].astype(np.int32) << 16) | (mask_rgb[:, :, 1].astype(np.int32) << 8) | mask_rgb[:, :, 2].astype(np.int32)
        unique_colors = np.unique(mask_packed)

        instances = []
        for p_col in unique_colors:
            cid = FPIC_PACKED_TO_CLASS.get(int(p_col))
            if cid is None:
                continue

            match = (mask_packed == p_col).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(match)
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Filter noise
                if area < 9 or w < 3 or h < 3:
                    continue

                instances.append({
                    "class_id": cid,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "area": float(area),
                })

        split_key = "train" if any(k in base_name for k in train_names) else "test"
        boards[split_key].append({
            "board_id": base_name,
            "dataset": "Graph-F",
            "image_path": img_path,
            "mask_path": mask_path,
            "annotations": instances,
        })

    return boards


def build_full_dataset(dataset_root: str = "data/GraphPCB"):
    print(f"Building dataset from: {dataset_root}")
    split_names = get_official_split_names(dataset_root)

    w_boards = parse_graph_w(dataset_root, split_names)
    f_boards = parse_graph_f(dataset_root, split_names)

    train_boards = w_boards["train"] + f_boards["train"]
    test_boards = w_boards["test"] + f_boards["test"]

    print(f"Total Boards - Train: {len(train_boards)} | Test: {len(test_boards)}")

    # Deterministically split 15% of train boards into validation
    np.random.seed(42)
    indices = np.random.permutation(len(train_boards))
    val_size = int(0.15 * len(train_boards))
    val_idx = set(indices[:val_size])

    final_train_boards = [b for i, b in enumerate(train_boards) if i not in val_idx]
    final_val_boards = [b for i, b in enumerate(train_boards) if i in val_idx]

    print(f"Splits - Train: {len(final_train_boards)} | Val: {len(final_val_boards)} | Test: {len(test_boards)}")

    # Pre-crop component patches for high-speed classification training
    def extract_and_save_crops(board_list, split_name, crops_root="data/crops"):
        split_dir = os.path.join(crops_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        crops = []

        print(f"Pre-cropping {split_name} components ({len(board_list)} boards)...")
        for b in tqdm(board_list, desc=f"Cropping {split_name}"):
            img_path = b["image_path"]
            board_id = b["board_id"]
            dataset_type = b["dataset"]
            if not os.path.exists(img_path):
                continue

            with Image.open(img_path) as full_img:
                full_rgb = full_img.convert("RGB")
                w, h = full_rgb.size

                for idx, ann in enumerate(b["annotations"]):
                    crop_filename = f"{dataset_type}_{board_id}_c{idx:05d}.jpg"
                    crop_path = os.path.join(split_dir, crop_filename)

                    if not os.path.exists(crop_path):
                        x1, y1, x2, y2 = ann["bbox"]
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(w, max(x1 + 1, int(x2))), min(h, max(y1 + 1, int(y2)))
                        crop_patch = full_rgb.crop((x1, y1, x2, y2))
                        crop_patch.save(crop_path, "JPEG", quality=95)

                    crops.append({
                        "crop_path": crop_path,
                        "image_path": img_path,
                        "bbox": ann["bbox"],
                        "class_id": ann["class_id"],
                    })
        return crops

    train_crops = extract_and_save_crops(final_train_boards, "train")
    val_crops = extract_and_save_crops(final_val_boards, "val")
    test_crops = extract_and_save_crops(test_boards, "test")

    print(f"Component Crops - Train: {len(train_crops):,} | Val: {len(val_crops):,} | Test: {len(test_crops):,}")

    # Save manifest
    manifest = {
        "train_boards": final_train_boards,
        "val_boards": final_val_boards,
        "test_boards": test_boards,
        "train_crops": train_crops,
        "val_crops": val_crops,
        "test_crops": test_crops,
    }

    os.makedirs("data", exist_ok=True)
    manifest_path = "data/dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Saved complete manifest to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    build_full_dataset()
