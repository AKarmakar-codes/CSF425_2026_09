"""Authoritative constants and taxonomy for CS F425 Assignment II.

Defines the official 25-class taxonomy, source-mask palette, display palette,
color lookup tables, and Graph-W XML component mapping rules.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple
import numpy as np

# Canonical 25 component classes (1-indexed, IDs 1..25)
CLASSES: Tuple[str, ...] = (
    "R",   # 1: Resistor
    "C",   # 2: Capacitor
    "U",   # 3: IC (legacy U)
    "Q",   # 4: Transistor
    "J",   # 5: Connector / Jack
    "L",   # 6: Inductor
    "RA",  # 7: Resistor Array
    "D",   # 8: Diode
    "RN",  # 9: Resistor Network
    "TP",  # 10: Test Point
    "IC",  # 11: Integrated Circuit
    "P",   # 12: Plug / Header Pin
    "CR",  # 13: Crystal / Resonator
    "M",   # 14: Motor / Mechanical
    "BTN", # 15: Push Button
    "FB",  # 16: Ferrite Bead
    "CRA", # 17: Capacitor Array
    "SW",  # 18: Switch (SW)
    "T",   # 19: Transformer
    "F",   # 20: Fuse
    "V",   # 21: Varistor
    "LED", # 22: Light Emitting Diode
    "S",   # 23: Switch (S)
    "QA",  # 24: Transistor Array
    "JP"   # 25: Jumper
)

NUM_CLASSES = 25
BACKGROUND_ID = 0
IGNORE_ID = 255

# Official Source RGB Palette (Ground Truth Color Coding in Graph-F)
SOURCE_RGB: Tuple[Tuple[int, int, int], ...] = (
    (255, 0, 0),     # 1: R
    (255, 255, 0),   # 2: C
    (0, 234, 255),   # 3: U
    (170, 0, 255),   # 4: Q
    (255, 127, 0),   # 5: J
    (191, 255, 0),   # 6: L
    (0, 149, 255),   # 7: RA
    (106, 255, 0),   # 8: D
    (0, 64, 255),    # 9: RN
    (237, 185, 185), # 10: TP
    (185, 215, 237), # 11: IC
    (231, 233, 185), # 12: P
    (220, 185, 237), # 13: CR
    (185, 237, 224), # 14: M
    (143, 35, 35),   # 15: BTN
    (35, 98, 143),   # 16: FB
    (143, 106, 35),  # 17: CRA
    (107, 35, 143),  # 18: SW
    (79, 143, 35),   # 19: T
    (115, 115, 115), # 20: F
    (204, 204, 204), # 21: V
    (245, 130, 48),  # 22: LED
    (220, 190, 255), # 23: S
    (170, 255, 195), # 24: QA
    (255, 250, 200)  # 25: JP
)

# High-Contrast Visual / Display Palette (for Report Visualizations & Overlays)
DISPLAY_RGB: Tuple[Tuple[int, int, int], ...] = (
    (229, 57, 53),   # 1: R
    (30, 136, 229),  # 2: C
    (0, 172, 193),   # 3: U
    (142, 36, 170),  # 4: Q
    (251, 140, 0),   # 5: J
    (67, 160, 71),   # 6: L
    (109, 76, 65),   # 7: RA
    (216, 27, 96),   # 8: D
    (57, 73, 171),   # 9: RN
    (192, 202, 51),  # 10: TP
    (0, 137, 123),   # 11: IC
    (255, 179, 0),   # 12: P
    (94, 53, 177),   # 13: CR
    (3, 155, 229),   # 14: M
    (117, 117, 117), # 15: BTN
    (253, 216, 59),  # 16: FB
    (33, 33, 33),    # 17: CRA
    (126, 87, 194),  # 18: SW
    (244, 81, 30),   # 19: T
    (46, 125, 50),   # 20: F
    (40, 53, 147),   # 21: V
    (124, 179, 66),  # 22: LED
    (84, 110, 122),  # 23: S
    (194, 24, 91),   # 24: QA
    (249, 168, 37)   # 25: JP
)

CLASS_TO_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i + 1: c for i, c in enumerate(CLASSES)}
SYMBOLS = {c.lower(): i + 1 for i, c in enumerate(CLASSES)}

# Exact multi-word and descriptive names used in Graph-W XML annotations
TYPES = {
    "resistor": 1,
    "capacitor": 2,
    "electrolytic capacitor": 2,
    "ic": 11,
    "integrated circuit": 11,
    "transistor": 4,
    "diode": 8,
    "zener diode": 8,
    "inductor": 6,
    "connector": 5, # J: Connector
    "plug": 12,     # P: Plug / Pin header
    "switch": 18,
    "button": 15,
    "fuse": 20,
    "led": 22,
    "light emitting diode": 22,
    "test point": 10,
    "oscillator": 13,
    "crystal": 13,
    "transformer": 19,
    "varistor": 21,
    "relay": 14,
    "jumper": 25,
    "ferrite bead": 16,
    "resistor array": 7,
    "resistor network": 9,
    "capacitor array": 17,
    "transistor array": 24,
    "switch (s)": 23,
    "switch (sw)": 18
}

# Precomputed 24-bit RGB packed color lookup table (LUT) for O(1) mask decoding
GLOBAL_LUT = np.full(1 << 24, IGNORE_ID, dtype=np.uint8)
GLOBAL_LUT[0] = BACKGROUND_ID
for cid, (r, g, b) in enumerate(SOURCE_RGB, 1):
    packed = (r << 16) | (g << 8) | b
    GLOBAL_LUT[packed] = cid


def map_graph_w_name(raw_name: Optional[str]) -> Optional[int]:
    """Map a Graph-W XML component name/annotation to a 1-based class ID (1..25)."""
    if not raw_name:
        return None
    raw = " ".join(str(raw_name).lower().replace('"', "").replace("'", "").replace("_", " ").split())
    if raw.startswith("component text") or raw.startswith("pin"):
        return None
    # 1. Multi-word descriptive type match (longest first)
    for name, cid in sorted(TYPES.items(), key=lambda x: -len(x[0])):
        if raw == name or raw.startswith(name + " "):
            return cid
    # 2. Canonical symbol match
    for i, code in enumerate(CLASSES, 1):
        if raw == code.lower() or raw.startswith(code.lower() + " "):
            return i
    # 3. Standard reference designators (e.g., R1, C12, U4, Q2, JP1)
    m = re.fullmatch(r"([a-z]+)(?:\d+)?", raw)
    if m and m[1] in SYMBOLS:
        return SYMBOLS[m[1]]
    return None


def decode_source_mask(rgb_array: np.ndarray) -> np.ndarray:
    """Decode an RGB ground-truth source mask into a uint8 2D array of class IDs (0..25, 255)."""
    packed = (rgb_array[:, :, 0].astype(np.uint32) << 16) | \
             (rgb_array[:, :, 1].astype(np.uint32) << 8) | \
             rgb_array[:, :, 2].astype(np.uint32)
    return GLOBAL_LUT[packed]


def encode_source_mask(class_id_array: np.ndarray) -> np.ndarray:
    """Encode a 2D array of class IDs (0..25) into an RGB image using the official SOURCE_RGB palette."""
    h, w = class_id_array.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(SOURCE_RGB, 1):
        rgb[class_id_array == cid] = color
    return rgb


def encode_display_mask(class_id_array: np.ndarray) -> np.ndarray:
    """Encode a 2D array of class IDs (0..25) into an RGB image using the high-contrast DISPLAY_RGB palette."""
    h, w = class_id_array.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in enumerate(DISPLAY_RGB, 1):
        rgb[class_id_array == cid] = color
    return rgb
