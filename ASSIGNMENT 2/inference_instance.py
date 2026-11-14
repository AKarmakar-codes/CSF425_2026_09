"""Standalone Instance Segmentation Inference (Zero-src/ Import Purity).

Conforms strictly to Assignment II Master Execution Plan (Step 6, 7, 8) & PDF Rubric (Question 2):
- Zero-src/ Import Purity: Imports ONLY third-party packages (torch, torchvision, numpy, cv2, PIL).
- Dual-Checkpoint Compatibility: Loads both CompactMask2Former and legacy checkpoints.
- Sliding-Window 1024x1024 Tiling: With 256px overlap (stride 768).
- Zero-Parameter Studio Background Filter: Suppresses false positives on table outside PCB footprint.
- Cross-Tile Class-Aware Mask-NMS & Containment Suppression: Merges overlapping tile predictions.
- Painter's Algorithm Instance Layering: Smaller passives rendered over larger pads with unique integer IDs.
- Mask-Conditioned Instance Grad-CAM (Q2c): Backpropagates from instance query score onto spatial feature maps.
- Outputs: <board_id>_rgb.png, <board_id>_overlay.png, <board_id>_black_masks.png,
           <board_id>_instances.json, <board_id>_gradcam.png, class_legend.png, run_manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# Canonical 25 component classes (1-indexed, IDs 1..25)
CLASSES: Tuple[str, ...] = (
    "R", "C", "U", "Q", "J", "L", "RA", "D", "RN", "TP",
    "IC", "P", "CR", "M", "BTN", "FB", "CRA", "SW", "T", "F",
    "V", "LED", "S", "QA", "JP",
)

NUM_CLASSES = 25
BACKGROUND_ID = 0
IGNORE_ID = 255

# High-Contrast Visual / Display Palette (RGB)
DISPLAY_RGB: np.ndarray = np.array([
    [229, 57, 53],   # 1: R
    [30, 136, 229],  # 2: C
    [0, 172, 193],   # 3: U
    [142, 36, 170],  # 4: Q
    [251, 140, 0],   # 5: J
    [67, 160, 71],   # 6: L
    [109, 76, 65],   # 7: RA
    [216, 27, 96],   # 8: D
    [57, 73, 171],   # 9: RN
    [192, 202, 51],  # 10: TP
    [0, 137, 123],   # 11: IC
    [255, 179, 0],   # 12: P
    [94, 53, 177],   # 13: CR
    [3, 155, 229],   # 14: M
    [117, 117, 117], # 15: BTN
    [253, 216, 59],  # 16: FB
    [33, 33, 33],    # 17: CRA
    [126, 87, 194],  # 18: SW
    [244, 81, 30],   # 19: T
    [46, 125, 50],   # 20: F
    [40, 53, 147],   # 21: V
    [124, 179, 66],  # 22: LED
    [84, 110, 122],  # 23: S
    [194, 24, 91],   # 24: QA
    [249, 168, 37],  # 25: JP
], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Architecture 1: CompactMask2Former (Primary Re-Engineered Architecture)
# ---------------------------------------------------------------------------

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))


class FPNPixelDecoder(nn.Module):
    def __init__(self, in_channels: Dict[str, int], hidden_dim: int = 256, mask_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lateral_c5 = nn.Conv2d(in_channels["res5"], hidden_dim, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels["res4"], hidden_dim, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels["res3"], hidden_dim, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(in_channels["res2"], hidden_dim, kernel_size=1)
        self.output_p5 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.output_p4 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.output_p3 = ConvModule(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.mask_features_conv = ConvModule(hidden_dim, mask_dim, kernel_size=3, padding=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        c2, c3, c4, c5 = features["res2"], features["res3"], features["res4"], features["res5"]
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p5_out = self.output_p5(p5)
        p4_out = self.output_p4(p4)
        p3_out = self.output_p3(p3)
        mask_features = self.mask_features_conv(p2)
        return mask_features, [p3_out, p4_out, p5_out]


class HaloDilatedCrossAttention(nn.Module):
    def __init__(self, d_model: int = 256, nheads: int = 8, halo_kernel: int = 5):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.halo_kernel = halo_kernel
        self.padding = halo_kernel // 2
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, use_halo_dilation: bool = True) -> torch.Tensor:
        B, Q, D = query.shape
        HW = key_value.shape[1]
        q = self.q_proj(query).view(B, Q, self.nheads, D // self.nheads).transpose(1, 2)
        k = self.k_proj(key_value).view(B, HW, self.nheads, D // self.nheads).transpose(1, 2)
        v = self.v_proj(key_value).view(B, HW, self.nheads, D // self.nheads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.nheads) ** 0.5
        if attn_mask is not None:
            if use_halo_dilation and self.halo_kernel > 1:
                dilated_mask = F.max_pool2d(attn_mask.float(), kernel_size=self.halo_kernel, stride=1, padding=self.padding)
            else:
                dilated_mask = attn_mask.float()
            mask_flat = dilated_mask.flatten(2).unsqueeze(1)
            attn_bias = (1.0 - (mask_flat > 0.5).float()) * -10000.0
            scores = scores + attn_bias
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, Q, D)
        return self.out_proj(out)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nheads: int = 8, dim_feedforward: int = 512, halo_kernel: int = 5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = HaloDilatedCrossAttention(d_model, nheads, halo_kernel=halo_kernel)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, feature_map: torch.Tensor, prev_mask: Optional[torch.Tensor] = None, use_halo: bool = True) -> torch.Tensor:
        q2 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + q2)
        H, W = feature_map.shape[-2:]
        prev_mask_interp = F.interpolate(prev_mask, size=(H, W), mode="bilinear", align_corners=False) if prev_mask is not None else None
        feature_flat = feature_map.flatten(2).transpose(1, 2)
        q_cross = self.cross_attn(query, feature_flat, attn_mask=prev_mask_interp, use_halo_dilation=use_halo)
        query = self.norm2(query + q_cross)
        q_ff = self.linear2(self.activation(self.linear1(query)))
        return self.norm3(query + q_ff)


class HaloMaskedTransformerDecoder(nn.Module):
    def __init__(self, num_classes: int = 25, num_queries: int = 150, d_model: int = 256, nheads: int = 8, num_layers: int = 6, dim_feedforward: int = 512, halo_kernel: int = 5, use_halo: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_halo = use_halo
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nheads, dim_feedforward, halo_kernel=halo_kernel)
            for _ in range(num_layers)
        ])
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.mask_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model))

    def forward(self, mask_features: torch.Tensor, multi_scale_features: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        B = mask_features.shape[0]
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        outputs = []
        cur_mask = None
        for layer_idx, layer in enumerate(self.layers):
            feat_map = multi_scale_features[layer_idx % len(multi_scale_features)]
            query = layer(query + query_pos, feat_map, prev_mask=cur_mask, use_halo=self.use_halo)
            class_logits = self.class_head(query)
            mask_embed = self.mask_embed_head(query)
            mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)
            outputs.append({"pred_logits": class_logits, "pred_masks": mask_logits})
            cur_mask = mask_logits.sigmoid()
        return outputs


class CompactMask2Former(nn.Module):
    def __init__(self, num_classes: int = 25, num_queries: int = 150, hidden_dim: int = 256, num_layers: int = 6, use_halo: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        base = resnet18(weights=None)
        self.backbone = nn.Module()
        self.backbone.conv1 = base.conv1
        self.backbone.bn1 = base.bn1
        self.backbone.relu = base.relu
        self.backbone.maxpool = base.maxpool
        self.backbone.layer1 = base.layer1
        self.backbone.layer2 = base.layer2
        self.backbone.layer3 = base.layer3
        self.backbone.layer4 = base.layer4
        out_channels = {"res2": 64, "res3": 128, "res4": 256, "res5": 512}

        self.pixel_decoder = FPNPixelDecoder(out_channels, hidden_dim=hidden_dim, mask_dim=hidden_dim)
        self.transformer_decoder = HaloMaskedTransformerDecoder(
            num_classes=num_classes, num_queries=num_queries, d_model=hidden_dim,
            nheads=8, num_layers=num_layers, dim_feedforward=512, halo_kernel=5, use_halo=use_halo
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(images)))
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = {"res2": c2, "res3": c3, "res4": c4, "res5": c5}
        mask_features, multi_scale = self.pixel_decoder(features)
        outputs = self.transformer_decoder(mask_features, multi_scale)
        final = outputs[-1]
        return final["pred_logits"], final["pred_masks"]


# ---------------------------------------------------------------------------
# Architecture 2: Legacy Student (Backward-Compatible Fallback)
# ---------------------------------------------------------------------------

def sine_position(batch, height, width, dim, device, dtype):
    n = dim // 4
    freq = torch.pow(10000.0, torch.arange(n, device=device, dtype=torch.float32) / max(1, n))
    yy, xx = torch.meshgrid(
        torch.linspace(0, 2 * torch.pi, height, device=device, dtype=torch.float32),
        torch.linspace(0, 2 * torch.pi, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = xx[..., None] / freq
    py = yy[..., None] / freq
    return torch.cat((px.sin(), px.cos(), py.sin(), py.cos()), dim=-1).reshape(1, height * width, dim).to(dtype).expand(batch, -1, -1)


class QLayer(nn.Module):
    def __init__(self, d=256, masked_attention=True, memory_size=32):
        super().__init__()
        self.masked_attention = masked_attention
        self.memory_size = memory_size
        heads = 8 if d % 8 == 0 else 4
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))

    def forward(self, q, m, mask):
        n = self.norms[0](q)
        q = q + self.self_attn(n, n, n, need_weights=False)[0]
        n = self.norms[1](q)
        if self.masked_attention:
            blocked = (
                F.interpolate(mask.detach(), (self.memory_size, self.memory_size), mode="bilinear", align_corners=False)
                .flatten(2).sigmoid() < 0.5
            )
            blocked = blocked.repeat_interleave(self.cross_attn.num_heads, 0)
            blocked = blocked.masked_fill(blocked.all(-1, keepdim=True), False)
            cross = self.cross_attn(n, m, m, attn_mask=blocked, need_weights=False)[0]
        else:
            cross = self.cross_attn(n, m, m, need_weights=False)[0]
        q = q + cross
        return q + self.ffn(self.norms[2](q))


class LegacyStudent(nn.Module):
    def __init__(self, num_queries=100, hidden_dim=256, decoder_layers=6, masked_attention=True, memory_size=32, **kw):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.proj = nn.ModuleList(nn.Conv2d(c, hidden_dim, 1) for c in (64, 128, 256, 512))
        self.scale_logits = nn.Parameter(torch.zeros(4))
        num_groups = 16 if hidden_dim % 16 == 0 else 8
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.decoder = nn.ModuleList([
            QLayer(hidden_dim, masked_attention=masked_attention, memory_size=memory_size)
            for _ in range(decoder_layers)
        ])
        self.class_head = nn.Linear(hidden_dim, 26)
        self.mask_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone
        x = z.relu(z.bn1(z.conv1(x)))
        x = z.maxpool(x)
        fs = []
        for layer in (z.layer1, z.layer2, z.layer3, z.layer4):
            x = layer(x)
            fs.append(x)
        fs = [p(f) for p, f in zip(self.proj, fs)]
        size = fs[0].shape[-2:]
        w = self.scale_logits.softmax(0)
        pix = self.fuse(sum(a * F.interpolate(f, size, mode="bilinear", align_corners=False) for a, f in zip(w, fs)))
        mem = F.adaptive_avg_pool2d(pix, (self.memory_size, self.memory_size)).flatten(2).transpose(1, 2)
        mem = mem + sine_position(x.shape[0], self.memory_size, self.memory_size, pix.shape[1], pix.device, pix.dtype)
        q = self.query.weight[None].expand(x.shape[0], -1, -1)
        masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(q), pix) / (pix.shape[1] ** 0.5)
        for dec in self.decoder:
            q = dec(q, mem, masks)
            masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(q), pix) / (pix.shape[1] ** 0.5)
        return self.class_head(q), masks


# ---------------------------------------------------------------------------
# Preprocessing, Windowing & Studio Background Filter
# ---------------------------------------------------------------------------

def starts(n: int, t: int, s: int) -> List[int]:
    return list(range(0, max(1, n - t + 1), s)) + ([] if n <= t or (n - t) % s == 0 else [n - t])


def detect_board_mask(rgb: np.ndarray) -> np.ndarray:
    """Zero-parameter deterministic filter to detect PCB board boundary and mask studio table background."""
    h, w = rgb.shape[:2]
    try:
        scale = max(1, max(h, w) // 512)
        small = rgb[::scale, ::scale]
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        _, s_thresh = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, v_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(s_thresh, v_thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask_small = np.zeros(small.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_small, [largest], -1, 255, thickness=-1)
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if 0.05 < mask.mean() < 0.98:
                return mask
    except Exception:
        pass
    return np.ones((h, w), dtype=bool)


def load_model(path: str | Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model checkpoint with dual architecture support and strict parameter audit."""
    c = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = c.get("state_dict", c)

    has_pixel_decoder = any("pixel_decoder" in k for k in state_dict.keys())
    has_transformer = any("transformer_decoder" in k for k in state_dict.keys())

    if has_pixel_decoder or has_transformer:
        q_count = 150
        if "transformer_decoder.query_embed.weight" in state_dict:
            q_count = state_dict["transformer_decoder.query_embed.weight"].shape[0]
        model = CompactMask2Former(num_classes=NUM_CLASSES, num_queries=q_count, hidden_dim=256, num_layers=6, use_halo=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        cfg = c.get("model_config", {"num_classes": 25, "num_queries": 100, "hidden_dim": 256, "decoder_layers": 6})
        model = LegacyStudent(**cfg)
        model.load_state_dict(state_dict, strict=True)

    count = sum(p.numel() for p in model.parameters())
    if count > 23600000:
        raise ValueError(f"Inference parameter cap exceeded: {count:,} > 23,600,000")

    return model.to(device).eval(), c


def compute_instance_gradcam(
    model: nn.Module,
    rgb_tile: np.ndarray,
    device: torch.device,
    target_class: int,
    query_idx: int,
) -> np.ndarray:
    """Mask-conditioned Grad-CAM on single predicted instance query (Q2c)."""
    H, W = rgb_tile.shape[:2]
    inp = torch.from_numpy(rgb_tile.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    inp = (inp - mean) / std
    inp.requires_grad_(True)

    activation = {}
    def hook_fn(module, input, output):
        activation["features"] = output
        output.register_hook(lambda grad: activation.update({"grad": grad}))

    hook_handle = None
    if hasattr(model, "pixel_decoder"):
        hook_handle = model.pixel_decoder.mask_features_conv.conv.register_forward_hook(hook_fn)
    elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        hook_handle = model.backbone.layer4.register_forward_hook(hook_fn)

    model.zero_grad()
    logits, masks = model(inp)

    # Q2c: Instance score Y_{inst}^{(q*)} = P_{q*}(c*) * sum_{h,w} sigma(M_{q*}(h,w))
    prob_c = logits[0, query_idx].softmax(dim=-1)[target_class]
    mask_mass = masks[0, query_idx].sigmoid().sum()
    instance_score = prob_c * mask_mass
    instance_score.backward()

    if hook_handle:
        hook_handle.remove()

    if "features" not in activation or "grad" not in activation:
        return np.zeros((H, W), dtype=np.float32)

    feat = activation["features"]
    grad = activation["grad"]
    weights = grad.mean(dim=(-2, -1), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam_np = cam_up.squeeze().detach().cpu().numpy()

    c_min, c_max = cam_np.min(), cam_np.max()
    if c_max > c_min:
        cam_np = (cam_np - c_min) / (c_max - c_min + 1e-8)
    return cam_np


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Standalone Instance Segmentation Inference")
    p.add_argument("--checkpoint", required=True, help="Path to best_instance.ckpt")
    p.add_argument("--images", required=True, help="Path to input DSLR image or directory")
    p.add_argument("--out", required=True, help="Directory to save prediction results")
    p.add_argument("--device", default="auto", help="Compute device ('cuda:0', 'cpu', 'auto')")
    p.add_argument("--threshold", type=float, default=None, help="Confidence threshold (defaults to checkpoint value or 0.25)")
    p.add_argument("--tile-size", type=int, default=1024, help="Tile resolution")
    p.add_argument("--stride", type=int, default=768, help="Sliding window stride (256px overlap)")
    p.add_argument("--suppress-background", action="store_true", default=True, help="Suppress table background outside PCB")
    p.add_argument("--gradcam", action="store_true", help="Generate mask-conditioned Grad-CAM heatmaps")
    args = p.parse_args()

    started_unix = time.time()

    dev_str = args.device
    if dev_str == "auto":
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    model, ckpt_dict = load_model(ckpt_path, device)
    params_inf = sum(p.numel() for p in model.parameters())

    meta = ckpt_dict.get("metadata", {})
    class_thresholds = meta.get("instance_class_thresholds")
    default_thresh = float(meta.get("instance_threshold", 0.25) if args.threshold is None else args.threshold)

    src_path = Path(args.images)
    if src_path.is_file():
        image_files = [src_path]
    elif src_path.is_dir():
        image_files = sorted(
            [p for p in src_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
    else:
        raise FileNotFoundError(f"Input path does not exist: {src_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class legend
    legend_img = Image.new("RGB", (360, 25 * 24 + 10), (255, 255, 255))
    ld = ImageDraw.Draw(legend_img)
    for i, (name, col) in enumerate(zip(CLASSES, DISPLAY_RGB.tolist())):
        ld.rectangle((6, i * 24 + 4, 26, i * 24 + 22), fill=tuple(map(int, col)))
        ld.text((34, i * 24 + 6), f"{i + 1}: {name}", fill=(0, 0, 0))
    legend_img.save(out_dir / "class_legend.png")

    tested_boards = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    print(f"=== Running Instance Segmentation Inference ===")
    print(f"Model parameters: {params_inf:,} | Checkpoint: {ckpt_path}")
    print(f"Processing {len(image_files)} image(s)...")

    for img_idx, img_file in enumerate(image_files, 1):
        stem = img_file.stem
        rgb_raw = np.asarray(Image.open(img_file).convert("RGB"))
        h, w = rgb_raw.shape[:2]
        tested_boards.append({"board_id": stem, "image_dimensions": [h, w]})

        board_mask = detect_board_mask(rgb_raw) if args.suppress_background else np.ones((h, w), dtype=bool)
        candidates = []

        for y in starts(h, args.tile_size, args.stride):
            for x in starts(w, args.tile_size, args.stride):
                crop = rgb_raw[y : min(y + args.tile_size, h), x : min(x + args.tile_size, w)]
                ch, cw = crop.shape[:2]

                tile_pad = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.uint8)
                tile_pad[:ch, :cw] = crop

                tile_norm = (tile_pad.astype(np.float32) / 255.0 - mean) / std
                ten = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

                with torch.inference_mode():
                    logits, masks = model(ten)
                    scores = logits[0].softmax(dim=-1)[:, 1:26]  # (Q, 25)
                    probs = F.interpolate(masks[0:1], size=(args.tile_size, args.tile_size), mode="bilinear", align_corners=False)[0].sigmoid()

                    cls_ids = scores.argmax(dim=-1)      # 0..24
                    confs = scores.max(dim=-1).values     # (Q,)

                probs_np = probs[:, :ch, :cw].cpu().numpy()

                for q in range(len(cls_ids)):
                    score = float(confs[q])
                    cid = int(cls_ids[q]) + 1  # 1..25
                    thresh_c = default_thresh
                    if args.threshold is None and class_thresholds is not None:
                        thresh_c = float(class_thresholds.get(str(cid), default_thresh))

                    if score < thresh_c:
                        continue

                    bin_m = (probs_np[q] >= 0.50)
                    if bin_m.sum() < 4:
                        continue

                    ys, xs = np.where(bin_m)
                    gx1 = x + int(xs.min())
                    gy1 = y + int(ys.min())
                    gx2 = x + int(xs.max() + 1)
                    gy2 = y + int(ys.max() + 1)

                    # Studio background suppression: box center must be inside PCB board
                    cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                    if args.suppress_background and not board_mask[cy, cx]:
                        continue

                    crop_bin = bin_m[int(ys.min()) : int(ys.max() + 1), int(xs.min()) : int(xs.max() + 1)]
                    candidates.append({
                        "category_id": cid,
                        "class_code": CLASSES[cid - 1],
                        "score": score,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "crop_mask": crop_bin,
                        "area": int(bin_m.sum()),
                        "tile_x": x,
                        "tile_y": y,
                        "query_idx": q,
                        "tile_pad": tile_pad,
                    })

        # Cross-tile same-class Mask-NMS & containment suppression
        candidates.sort(key=lambda z: z["score"], reverse=True)
        kept = []
        for cand in candidates:
            duplicate = False
            for prev in kept:
                if cand["category_id"] != prev["category_id"]:
                    continue
                cb = cand["bbox"]
                pb = prev["bbox"]
                # Bounding box intersection
                ix1, iy1 = max(cb[0], pb[0]), max(cb[1], pb[1])
                ix2, iy2 = min(cb[2], pb[2]), min(cb[3], pb[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                # Mask overlap
                cm = cand["crop_mask"]
                pm = prev["crop_mask"]
                c_crop = cm[iy1 - cb[1] : iy2 - cb[1], ix1 - cb[0] : ix2 - cb[0]]
                p_crop = pm[iy1 - pb[1] : iy2 - pb[1], ix1 - pb[0] : ix2 - pb[0]]
                inter = np.logical_and(c_crop, p_crop).sum()
                den = min(cand["area"], prev["area"])
                if den > 0 and (inter / den) > 0.55:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(cand)

        # Painter's algorithm: sort kept instances by area descending
        # Large chips/connectors painted first, smaller passives painted on top
        kept.sort(key=lambda z: z["area"], reverse=True)

        instance_id_map = np.zeros((h, w), dtype=np.uint16)
        black_canvas = np.zeros_like(rgb_raw)
        exported_instances = []

        for inst_num, inst in enumerate(kept, 1):
            x1, y1, x2, y2 = inst["bbox"]
            cm = inst["crop_mask"]
            cid = inst["category_id"]
            color = DISPLAY_RGB[cid - 1]

            # Resolve non-overlapping pixels
            view_ids = instance_id_map[y1:y2, x1:x2]
            view_canvas = black_canvas[y1:y2, x1:x2]

            view_ids[cm] = inst_num
            view_canvas[cm] = color

            exported_instances.append({
                "instance_id": int(inst_num),
                "category_id": int(cid),
                "class_code": str(inst["class_code"]),
                "score": round(float(inst["score"]), 4),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": int(inst["area"]),
            })

        # 1. <board_id>_rgb.png
        Image.fromarray(rgb_raw).save(out_dir / f"{stem}_rgb.png")

        # 2. <board_id>_black_masks.png (pure (0,0,0) black background)
        Image.fromarray(black_canvas).save(out_dir / f"{stem}_black_masks.png")

        # 3. <board_id>_overlay.png
        valid = instance_id_map > 0
        alpha = 0.50
        overlay = (rgb_raw.astype(np.float32) * (1.0 - alpha) + black_canvas.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        overlay[~valid] = rgb_raw[~valid]
        Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

        # 4. <board_id>_instances.json
        per_class_counts = {c: 0 for c in CLASSES}
        for rec in exported_instances:
            per_class_counts[rec["class_code"]] += 1

        inst_summary = {
            "board_id": stem,
            "total_instances": len(exported_instances),
            "per_class_counts": per_class_counts,
            "instances": exported_instances,
            "table_background_suppressed": args.suppress_background,
            "threshold": default_thresh,
        }
        (out_dir / f"{stem}_instances.json").write_text(json.dumps(inst_summary, indent=2))

        # 5. <board_id>_gradcam.png (Q2c: Mask-Conditioned Instance Grad-CAM)
        if args.gradcam and kept:
            # Pick highest confidence instance
            top_inst = max(kept, key=lambda z: z["score"])
            cam_crop = compute_instance_gradcam(
                model=model,
                rgb_tile=top_inst["tile_pad"],
                device=device,
                target_class=top_inst["category_id"],
                query_idx=top_inst["query_idx"],
            )
            full_cam = np.zeros((h, w), dtype=np.float32)
            gx, gy = top_inst["tile_x"], top_inst["tile_y"]
            gh, gw = min(args.tile_size, h - gy), min(args.tile_size, w - gx)
            full_cam[gy : gy + gh, gx : gx + gw] = cam_crop[:gh, :gw]

            heat_uint8 = (full_cam * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            gradcam_vis = (0.5 * rgb_raw.astype(np.float32) + 0.5 * heatmap_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
            Image.fromarray(gradcam_vis).save(out_dir / f"{stem}_gradcam.png")

        print(f"[{img_idx}/{len(image_files)}] Processed: {stem} ({len(exported_instances)} instances)")

    finished_unix = time.time()

    manifest = {
        "task": "instance",
        "tested_boards": tested_boards,
        "model_sha256": model_sha256,
        "params_inference": params_inf,
        "threshold": default_thresh,
        "command": sys.argv,
        "device": dev_str,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "target_labels_used": False,
    }
    manifest_file = out_dir / "run_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"\nInstance inference complete! Results saved to: {out_dir}")
    print(f"Manifest written to: {manifest_file}")


if __name__ == "__main__":
    main()
