"""Prepare source tiles directly from official GraphPCB archive/directory.

Supports both Graph-F (115 training boards) and Graph-W (37 training boards)
for a total of 152 official source training boards.
Strict physical PCB identity grouping prevents cross-split view leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
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
    for i, code in enumerate(CLASSES, 1):
        if raw == code.lower() or raw.startswith(code.lower() + " "):
            return i
    for name, cid in sorted(TYPES.items(), key=lambda x: -len(x[0])):
        if raw == name or raw.startswith(name + " "):
            return cid
    match = re.fullmatch(r"([a-z]+)(?:\d+)?", raw)
    if match and match[1] in SYMBOLS:
        return SYMBOLS[match[1]]
    return None


GLOBAL_LUT = np.full(1 << 24, 255, np.uint8)
GLOBAL_LUT[0] = 0
for cid, c in enumerate(SOURCE_RGB, 1):
    GLOBAL_LUT[(c[0] << 16) | (c[1] << 8) | c[2]] = cid


class Archive:
    def __init__(self, path):
        self.path = Path(path)
        self.z = zipfile.ZipFile(path) if self.path.is_file() else None
        self.names = (
            self.z.namelist()
            if self.z
            else [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]
        )

    def read(self, name):
        return self.z.read(name) if self.z else (self.path / name).read_bytes()


def decode_mask(data):
    from io import BytesIO
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


def present_codes(rgb):
    packed = (rgb[:, :, 0].astype(np.uint32) << 16) | (rgb[:, :, 1].astype(np.uint32) << 8) | rgb[:, :, 2]
    found = set(map(int, np.unique(packed)))
    lut = {(r << 16) | (g << 8) | b: i + 1 for i, (r, g, b) in enumerate(SOURCE_RGB)}
    unknown = found - set([0]) - set(lut)
    if unknown:
        raise ValueError(f"Unmapped source mask colors: {sorted(unknown)[:8]}")
    return {lut[c] for c in found if c in lut}


def get_physical_board_group(stem: str) -> str:
    m = re.match(r"pcb_(\d+)", stem)
    if m:
        return f"pcb_{m.group(1)}"
    clean = re.sub(r"_(top|bottom\d?|image|mask)$", "", stem, flags=re.IGNORECASE)
    clean = re.sub(r"^graph_\d+_", "", clean)
    return clean


def main(a):
    src = Archive(a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    # 1. Discover Graph-F official partitions
    f_graph_paths = sorted(n for n in src.names if re.search(r"/Graph-F/graphs/(train|test)/[^/]+\.pt$", "/" + n))
    f_images = {PurePosixPath(n).name: n for n in src.names if "/Graph-F/image_dir/" in "/" + n}
    f_masks = {PurePosixPath(n).name: n for n in src.names if "/Graph-F/mask_dir/" in "/" + n}

    rows = []
    for graph in f_graph_paths:
        split = re.search(r"/graphs/(train|test)/", "/" + graph).group(1)
        stem = PurePosixPath(graph).stem
        stem = re.sub(r"^graph_\d+_", "", stem)
        candidates = [(stem + "_image.png", stem + "_mask.png"), (stem + ".png", stem + "_mask.png")]
        match = next(((f_images[i], f_masks[m]) for i, m in candidates if i in f_images and m in f_masks), None)
        if not match:
            continue
        board_group = get_physical_board_group(stem)
        rows.append({
            "domain": "Graph-F",
            "split": split,
            "stem": stem,
            "board_group": board_group,
            "image_source": match[0],
            "mask_source": match[1],
            "is_xml": False,
        })

    # 2. Discover Graph-W official partitions (if available)
    w_graph_paths = sorted(n for n in src.names if re.search(r"/Graph-W/graphs/(train|test)/[^/]+\.pt$", "/" + n))
    w_images = {PurePosixPath(n).name: n for n in src.names if "/Graph-W/image_dir/" in "/" + n}
    w_masks = {PurePosixPath(n).name: n for n in src.names if "/Graph-W/mask_dir/" in "/" + n}

    w_rows = []
    for graph in w_graph_paths:
        split = re.search(r"/graphs/(train|test)/", "/" + graph).group(1)
        stem = PurePosixPath(graph).stem
        stem = re.sub(r"^graph_\d+_", "", stem)
        candidates_img = [stem + ".jpg", stem + ".png", stem + "_image.png"]
        candidates_mask = [stem + ".xml", stem + "_mask.png", stem + ".png"]
        img_match = next((w_images[i] for i in candidates_img if i in w_images), None)
        mask_match = next((w_masks[m] for m in candidates_mask if m in w_masks), None)
        if not img_match or not mask_match:
            continue
        board_group = get_physical_board_group(stem)
        is_xml = mask_match.endswith(".xml")
        w_rows.append({
            "domain": "Graph-W",
            "split": split,
            "stem": stem,
            "board_group": board_group,
            "image_source": img_match,
            "mask_source": mask_match,
            "is_xml": is_xml,
        })

    if a.include_graph_w:
        rows.extend(w_rows)

    f_train_count = sum(r["split"] == "train" and r["domain"] == "Graph-F" for r in rows)
    w_train_count = sum(r["split"] == "train" and r["domain"] == "Graph-W" for r in rows)
    print(f"Loaded source training boards: Graph-F={f_train_count}, Graph-W={w_train_count}, Total={f_train_count + w_train_count}")

    train_rows = [r for r in rows if r["split"] == "train"]
    groups = sorted({r["board_group"] for r in train_rows})

    # Deterministic physical board split
    group_classes = {g: set() for g in groups}
    for r in train_rows:
        if not r["is_xml"]:
            group_classes[r["board_group"]].update(present_codes(decode_mask(src.read(r["mask_source"]))))
        else:
            try:
                tree = ET.fromstring(src.read(r["mask_source"]))
                for obj in tree.findall("object"):
                    cid = map_graph_w_name(obj.findtext("name"))
                    if cid and 1 <= cid <= 25:
                        group_classes[r["board_group"]].add(cid)
            except Exception:
                pass

    remaining = {cid: sum(cid in cs for cs in group_classes.values()) for cid in range(1, 26)}
    target_val_count = max(1, round(len(groups) * 0.2))
    order = sorted(groups, key=lambda g: hashlib.sha256(f"{a.seed}:{g}".encode()).hexdigest())
    val_groups = set()

    for g in order:
        if len(val_groups) >= target_val_count:
            break
        if all(remaining[cid] > 1 for cid in group_classes[g]):
            val_groups.add(g)
            for cid in group_classes[g]:
                remaining[cid] -= 1

    if not val_groups:
        raise ValueError("Cannot form validation split while retaining each class in training")

    output = {"train": [], "val": []}
    tile = a.tile
    step = tile - a.overlap

    for r in train_rows:
        dest = "val" if r["board_group"] in val_groups else "train"
        rgb = np.asarray(Image.open(__import__("io").BytesIO(src.read(r["image_source"]))).convert("RGB"))
        h, w = rgb.shape[:2]

        if not r["is_xml"]:
            label = decode_mask(src.read(r["mask_source"]))
            if rgb.shape != label.shape:
                raise ValueError(f"Image/mask size mismatch {r['stem']}")
            packed = (label[:, :, 0].astype(np.uint32) << 16) | (label[:, :, 1].astype(np.uint32) << 8) | label[:, :, 2]
            sem = GLOBAL_LUT[packed]
            if (sem == 255).any():
                raise ValueError(f"Unknown source-mask color in {r['stem']}")
        else:
            # Graph-W XML to semantic and instance representation
            sem = np.zeros((h, w), dtype=np.uint8)
            tree = ET.fromstring(src.read(r["mask_source"]))
            for obj in tree.findall("object"):
                cid = map_graph_w_name(obj.findtext("name"))
                if not cid or not (1 <= cid <= 25):
                    continue
                box = obj.find("bndbox")
                x1 = max(0, min(w - 1, int(float(box.findtext("xmin")))))
                y1 = max(0, min(h - 1, int(float(box.findtext("ymin")))))
                x2 = max(x1 + 1, min(w, int(float(box.findtext("xmax")))))
                y2 = max(y1 + 1, min(h, int(float(box.findtext("ymax")))))
                sem[y1:y2, x1:x2] = cid

        ys = list(range(0, max(1, h - tile + 1), step))
        xs = list(range(0, max(1, w - tile + 1), step))
        if not ys or ys[-1] != max(0, h - tile):
            ys.append(max(0, h - tile))
        if not xs or xs[-1] != max(0, w - tile):
            xs.append(max(0, w - tile))

        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                crop = sem[y : min(y + tile, h), x : min(x + tile, w)]
                im = rgb[y : y + crop.shape[0], x : x + crop.shape[1]]
                if crop.shape != (tile, tile):
                    pad_sem = np.zeros((tile, tile), np.uint8)
                    pad_im = np.zeros((tile, tile, 3), np.uint8)
                    pad_sem[: crop.shape[0], : crop.shape[1]] = crop
                    pad_im[: im.shape[0], : im.shape[1]] = im
                    crop = pad_sem
                    im = pad_im

                ids = np.zeros(crop.shape, np.uint16)
                objects = []
                nextid = 0
                for cid in np.unique(crop):
                    if cid == 0 or cid == 255:
                        continue
                    n, cc, stats, _ = cv2.connectedComponentsWithStats((crop == cid).astype(np.uint8), connectivity=8)
                    for k in range(1, n):
                        x0, y0, bw, bh, area = map(int, stats[k])
                        if area < 4:
                            continue
                        nextid += 1
                        if nextid >= 65535:
                            raise ValueError("Tile contains too many connected components for uint16 instance IDs")
                        ids[cc == k] = nextid
                        objects.append({"instance_id": nextid, "category_id": int(cid)})

                if not objects:
                    continue
                name = f"{r['stem']}_{x}_{y}"
                ip = f"images/{dest}/{name}.jpg"
                mp = f"masks/{dest}/{name}.png"
                (out / ip).parent.mkdir(parents=True, exist_ok=True)
                (out / mp).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(im).save(out / ip, quality=95)
                Image.fromarray(ids).save(out / mp)
                output[dest].append({
                    "image": ip,
                    "instance_map": mp,
                    "domain": r["domain"],
                    "board_id": r["board_group"],
                    "source_stem": r["stem"],
                    "tile_xy": [x, y],
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                    "objects": objects,
                })

    archive_hash = None
    if Path(a.source).is_file():
        digest = hashlib.sha256()
        with Path(a.source).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        archive_hash = digest.hexdigest()

    total_train_boards = len(train_rows)
    manifest = {
        "format": "csf425-a2-tiles-v1",
        "seed": a.seed,
        "tile": tile,
        "overlap": a.overlap,
        "source_data": "Official GraphPCB training partitions (Graph-F and Graph-W)",
        "heldout_official_test_used": False,
        "split_unit": "physical PCB board group",
        "train_board_groups": sorted(set(r["board_group"] for r in train_rows) - val_groups),
        "val_board_groups": sorted(val_groups),
        "source_archive": Path(a.source).name,
        "source_archive_sha256": archive_hash,
        "original_source_train_images": 152,
        "used_graph_f_train_images": f_train_count,
        "used_graph_w_train_images": w_train_count,
        "total_used_source_train_images": total_train_boards,
        "external_images": 0,
        "r_i": float(total_train_boards / 152),
        "A_i": 1.0,
        "class_ids": {str(i + 1): c for i, c in enumerate(CLASSES)},
        "records": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for split, records in output.items():
        (out / f"{split}.json").write_text(json.dumps(records))
    print(
        json.dumps(
            {
                "tiles": {k: len(v) for k, v in output.items()},
                "train_groups": len(manifest["train_board_groups"]),
                "val_groups": len(val_groups),
                "total_train_boards_used": total_train_boards,
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="official GraphPCB archive or extracted directory")
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-graph-w", action="store_true", default=True, help="Include Graph-W training boards")
    args = p.parse_args()
    if args.tile < 128 or args.overlap < 0 or args.overlap >= args.tile:
        raise ValueError("Invalid tile/overlap values")
    main(args)
