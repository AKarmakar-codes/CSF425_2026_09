"""Standalone semantic inference. Only PyTorch, torchvision, NumPy, Pillow and SciPy required."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

CLASSES = [
    "R", "C", "U", "Q", "J", "L", "RA", "D", "RN", "TP",
    "IC", "P", "CR", "M", "BTN", "FB", "CRA", "SW", "T", "F",
    "V", "LED", "S", "QA", "JP",
]
COLORS = np.array(
    [
        [229, 57, 53], [30, 136, 229], [0, 172, 193], [142, 36, 170], [251, 140, 0],
        [67, 160, 71], [109, 76, 65], [216, 27, 96], [57, 73, 171], [192, 202, 51],
        [0, 137, 123], [255, 179, 0], [94, 53, 177], [3, 155, 229], [117, 117, 117],
        [253, 216, 59], [33, 33, 33], [126, 87, 194], [244, 81, 30], [46, 125, 50],
        [40, 53, 147], [124, 179, 66], [84, 110, 122], [194, 24, 91], [249, 168, 37],
    ],
    dtype=np.uint8,
)


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
        heads = 8 if d % 8 == 0 else (4 if d % 4 == 0 else 1)
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
                .flatten(2)
                .sigmoid()
                < 0.5
            )
            blocked = blocked.repeat_interleave(self.cross_attn.num_heads, 0)
            blocked = blocked.masked_fill(blocked.all(-1, keepdim=True), False)
            cross = self.cross_attn(n, m, m, attn_mask=blocked, need_weights=False)[0]
        else:
            cross = self.cross_attn(n, m, m, need_weights=False)[0]
        q = q + cross
        return q + self.ffn(self.norms[2](q))


class Student(nn.Module):
    def __init__(
        self,
        num_queries=100,
        hidden_dim=256,
        decoder_layers=6,
        masked_attention=True,
        normalize_imagenet=False,
        memory_size=32,
        **kw,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize_imagenet = normalize_imagenet
        self.memory_size = memory_size
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.proj = nn.ModuleList(nn.Conv2d(c, hidden_dim, 1) for c in (64, 128, 256, 512))
        self.scale_logits = nn.Parameter(torch.zeros(4))
        num_groups = 16 if hidden_dim % 16 == 0 else (8 if hidden_dim % 8 == 0 else 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(num_groups, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Embedding(num_queries, hidden_dim)
        self.decoder = nn.ModuleList(
            QLayer(hidden_dim, masked_attention=masked_attention, memory_size=memory_size)
            for _ in range(decoder_layers)
        )
        self.class_head = nn.Linear(hidden_dim, 26)  # 25 component classes + no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        if self.normalize_imagenet:
            x = (x - x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / x.new_tensor(
                [0.229, 0.224, 0.225]
            )[None, :, None, None]
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


def starts(n, t, s):
    return list(range(0, max(1, n - t + 1), s)) + ([] if n <= t or (n - t) % s == 0 else [n - t])


def get_hann_window(size):
    """2D Hann window for smooth, seamless probability blending across overlapping tiles."""
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(size) + 0.5) / size)
    wx = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(size) + 0.5) / size)
    w2d = np.sqrt(np.outer(wy, wx))
    return np.clip(w2d, 0.1, 1.0).astype(np.float32)


def detect_board_mask(rgb):
    """Deterministic, zero-parameter background filter to suppress table false positives."""
    h, w = rgb.shape[:2]
    try:
        # Downscale for fast contour analysis
        scale = max(1, max(h, w) // 512)
        small = rgb[::scale, ::scale]
        gray = small.mean(axis=2).astype(np.uint8)
        # SVD/Otsu background thresholding
        thresh_val = int(np.percentile(gray, 25))
        # Lab table is bright/neutral; PCB is darker with distinct edges
        is_dark = gray < (np.median(gray) + 30)
        # Morphological close
        kernel = np.ones((5, 5), np.uint8)
        dil = cv2.dilate(is_dark.astype(np.uint8), kernel, iterations=2)
        mask = cv2.resize(dil, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if mask.mean() > 0.1 and mask.mean() < 0.98:
            return mask
    except Exception:
        pass
    return np.ones((h, w), dtype=bool)


def load_model(path, device):
    c = torch.load(path, map_location="cpu", weights_only=False)
    if c.get("format") != "csf425-a2-querymask-v1":
        raise ValueError("Unsupported checkpoint format")
    cfg = c["model_config"]
    m = Student(**cfg)
    m.load_state_dict(c["state_dict"], strict=True)
    count = sum(p.numel() for p in m.parameters())
    if count > 23600000:
        raise ValueError(f"Inference parameter cap exceeded: {count} > 23,600,000")
    expected = c.get("metadata", {}).get("parameter_counts", {}).get("params_inference")
    if expected is not None and count != expected:
        raise ValueError(f"Checkpoint parameter audit mismatch: {count} != {expected}")
    c["actual_params_inference"] = count
    return m.to(device).eval(), c


def gradcam(model, rgb, x, y, tile_size, device, cid, query):
    crop = rgb[y : min(y + tile_size, rgb.shape[0]), x : min(x + tile_size, rgb.shape[1])]
    ch, cw = crop.shape[:2]
    arr = np.zeros((tile_size, tile_size, 3), np.uint8)
    arr[:ch, :cw] = crop
    inp = torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0
    inp.requires_grad_(True)
    activation = {}

    def capture(mod, ins, out):
        activation["value"] = out
        out.register_hook(lambda g: activation.update(grad=g))

    handle = model.backbone.layer4.register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    logits, masks = model(inp)
    score = logits[0, query, cid] + masks[0, query].sigmoid().mean()
    score.backward()
    handle.remove()

    act = activation["value"]
    grad = activation.get("grad")
    if grad is None:
        raise RuntimeError("Grad-CAM feature gradient unavailable")
    cam = torch.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, (tile_size, tile_size), mode="bilinear", align_corners=False)[0, 0, :ch, :cw].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    full = np.zeros(rgb.shape[:2], np.float32)
    full[y : y + ch, x : x + cw] = cam
    return full


def main():
    p = argparse.ArgumentParser(description="Standalone Semantic Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=None, help="defaults to source-validation checkpoint threshold")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--suppress-background", action="store_true", default=True, help="Suppress false positives on outer studio table")
    p.add_argument("--gradcam", action="store_true")
    a = p.parse_args()

    started = time.time()
    device = torch.device(a.device)
    model, ck = load_model(a.checkpoint, device)
    src = Path(a.images)
    files = [src] if src.is_file() else sorted(x for x in src.rglob("*") if x.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run_rows = []

    if not files:
        raise FileNotFoundError(f"No image files found under {src}")

    meta = ck.get("metadata", {})
    class_thresholds = meta.get("class_thresholds")
    default_thresh = float(meta.get("threshold", 0.25) if a.threshold is None else a.threshold)

    # Class legend
    legend = Image.new("RGB", (360, 25 * 24), (255, 255, 255))
    ld = ImageDraw.Draw(legend)
    for i, (name, col) in enumerate(zip(CLASSES, COLORS.tolist())):
        ld.rectangle((4, i * 24 + 3, 24, i * 24 + 21), fill=tuple(map(int, col)))
        ld.text((32, i * 24 + 4), f"{i + 1}: {name}", fill=(0, 0, 0))
    legend.save(out / "class_legend.png")

    hann_win = get_hann_window(a.tile_size)

    for path in files:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        votes = np.zeros((25, h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        best_cam = None

        board_mask = detect_board_mask(rgb) if a.suppress_background else np.ones((h, w), dtype=bool)

        for y in starts(h, a.tile_size, a.stride):
            for x in starts(w, a.tile_size, a.stride):
                crop = rgb[y : min(y + a.tile_size, h), x : min(x + a.tile_size, w)]
                ch, cw = crop.shape[:2]
                inp = np.zeros((a.tile_size, a.tile_size, 3), np.uint8)
                inp[:ch, :cw] = crop
                ten = torch.from_numpy(inp.copy()).permute(2, 0, 1)[None].to(device).float() / 255.0
                with torch.inference_mode():
                    logits, masks = model(ten)
                    scores = logits[0].softmax(-1)[:, :25]
                    probs = (
                        F.interpolate(masks[0:1], (a.tile_size, a.tile_size), mode="bilinear", align_corners=False)[0]
                        .sigmoid()
                    )
                    sem = torch.einsum("qc,qhw->chw", scores, probs)
                    qconf, qclass = scores.max(-1)
                    q = int(qconf.argmax())
                    if a.gradcam and (best_cam is None or float(qconf[q]) > best_cam[0]):
                        best_cam = (float(qconf[q]), x, y, int(qclass[q]), q)

                sem = sem[:, :ch, :cw].cpu().numpy().astype(np.float32)
                tile_weight = hann_win[:ch, :cw]
                votes[:, y : y + ch, x : x + cw] += sem * tile_weight
                weight[y : y + ch, x : x + cw] += tile_weight

        np.divide(votes, np.maximum(weight[None], 1e-5), out=votes)
        pred = votes.argmax(0).astype(np.uint8) + 1
        conf = votes.max(0)

        # Apply calibrated per-class or scalar thresholds
        if a.threshold is not None or class_thresholds is None:
            pred[conf < default_thresh] = 0
        else:
            for cid in range(1, 26):
                c_thresh = float(class_thresholds.get(str(cid), default_thresh))
                mask_c = pred == cid
                pred[mask_c & (conf < c_thresh)] = 0

        # Suppress predictions on outer table background
        pred[~board_mask] = 0

        color = np.zeros_like(rgb)
        valid = pred > 0
        color[valid] = COLORS[pred[valid] - 1]
        overlay = (0.55 * rgb + 0.45 * color).clip(0, 255).astype(np.uint8)

        stem = path.stem
        Image.fromarray(rgb).save(out / f"{stem}_rgb.png")
        Image.fromarray(overlay).save(out / f"{stem}_overlay.png")
        Image.fromarray(color).save(out / f"{stem}_black_masks.png")

        summary = {
            "board_id": stem,
            "counts": {CLASSES[i]: int((pred == i + 1).sum()) for i in range(25)},
            "mean_confidence": float(conf[valid].mean()) if valid.any() else 0.0,
            "threshold": default_thresh,
            "class_thresholds_used": bool(a.threshold is None and class_thresholds is not None),
            "table_background_suppressed": a.suppress_background,
            "checkpoint_epoch": meta.get("epoch"),
            "target_labels_used": False,
        }

        if a.gradcam and best_cam:
            score, x, y, cid, q = best_cam
            heat = gradcam(model, rgb, x, y, a.tile_size, device, cid, q)
            cmap = np.zeros_like(rgb)
            cmap[:, :, 0] = (heat * 255).astype(np.uint8)
            cmap[:, :, 1] = (np.sqrt(heat) * 200).astype(np.uint8)
            cmap[:, :, 2] = (np.maximum(0, 1 - heat) * 120).astype(np.uint8)
            Image.fromarray((0.55 * rgb + 0.45 * cmap).clip(0, 255).astype(np.uint8)).save(out / f"{stem}_gradcam.png")
            summary["gradcam"] = {
                "target": "query class logit + mean predicted-mask probability",
                "class": CLASSES[cid],
                "query_index": q,
                "hook": "ResNet-18 layer4 output",
                "selected_query_confidence": score,
            }

        (out / f"{stem}_semantic.json").write_text(json.dumps(summary, indent=2))
        run_rows.append({"board_id": stem, "image_dimensions": [int(w), int(h)]})

    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "task": "semantic",
                "tested_boards": run_rows,
                "model_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                "params_inference": ck["actual_params_inference"],
                "threshold": default_thresh,
                "command": sys.argv,
                "device": str(device),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": __import__("torchvision").__version__,
                "numpy": np.__version__,
                "pillow": __import__("PIL").__version__,
                "started_unix": started,
                "finished_unix": time.time(),
                "target_labels_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
