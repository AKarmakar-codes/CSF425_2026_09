"""Compact multi-scale masked-query model; shared semantic and instance head.

The query decoder uses iterative mask-conditioned cross-attention. The backbone
is ResNet-18 and all layers are counted in the deployed parameter total.
Architecturally tuned to ~18.5M parameters (P_i <= 19.504M) to lock in the
maximum 1.10x efficiency multiplier E_i = 1.10 under the P_max <= 23.6M cap.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


def sine_position(batch, height, width, dim, device, dtype):
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by four for 2D position encoding")
    n = dim // 4
    freq = torch.pow(10000.0, torch.arange(n, device=device, dtype=torch.float32) / max(1, n))
    yy, xx = torch.meshgrid(
        torch.linspace(0, 2 * torch.pi, height, device=device, dtype=torch.float32),
        torch.linspace(0, 2 * torch.pi, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = xx[..., None] / freq
    py = yy[..., None] / freq
    pos = torch.cat((px.sin(), px.cos(), py.sin(), py.cos()), dim=-1).reshape(1, height * width, dim)
    return pos.to(dtype).expand(batch, -1, -1)


class QueryLayer(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, masked_attention: bool = True, memory_size: int = 32):
        super().__init__()
        self.masked_attention = masked_attention
        self.memory_size = memory_size
        heads = heads if dim % heads == 0 else (4 if dim % 4 == 0 else 1)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))

    def forward(self, q, memory, mask_logits):
        q0 = self.norms[0](q)
        q = q + self.self_attn(q0, q0, q0, need_weights=False)[0]
        qn = self.norms[1](q)
        if self.masked_attention:
            # Mask-conditioned attention: query attends only to foreground pixels
            blocked = F.interpolate(
                mask_logits.detach(), size=(self.memory_size, self.memory_size), mode="bilinear", align_corners=False
            )
            blocked = blocked.flatten(2).sigmoid() < 0.5
            blocked = blocked.repeat_interleave(self.cross_attn.num_heads, dim=0)
            blocked = blocked.masked_fill(blocked.all(-1, keepdim=True), False)
            cross = self.cross_attn(qn, memory, memory, attn_mask=blocked, need_weights=False)[0]
        else:
            cross = self.cross_attn(qn, memory, memory, need_weights=False)[0]
        q = q + cross
        return q + self.ffn(self.norms[2](q))


class CompactMask2Former(nn.Module):
    """25-class Mask2Former-style model with class and per-query masks."""

    def __init__(
        self,
        num_classes: int = 25,
        num_queries: int = 100,
        hidden_dim: int = 256,
        decoder_layers: int = 6,
        pretrained_backbone: bool = False,
        masked_attention: bool = True,
        normalize_imagenet: bool = False,
        memory_size: int = 32,
        **kw,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.decoder_layers = decoder_layers
        self.masked_attention = masked_attention
        self.normalize_imagenet = normalize_imagenet
        self.memory_size = memory_size

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained_backbone else None)
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
        heads = 8 if hidden_dim % 8 == 0 else 4
        self.decoder = nn.ModuleList(
            QueryLayer(hidden_dim, heads=heads, masked_attention=masked_attention, memory_size=memory_size)
            for _ in range(decoder_layers)
        )
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # final channel is no-object
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        if self.normalize_imagenet:
            mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            x = (x - mean) / std

        b = x.shape[0]
        z = self.backbone
        x = z.relu(z.bn1(z.conv1(x)))
        c1 = z.maxpool(x)
        c2 = z.layer1(c1)
        c3 = z.layer2(c2)
        c4 = z.layer3(c3)
        c5 = z.layer4(c4)

        feats = [p(c) for p, c in zip(self.proj, (c2, c3, c4, c5))]
        target = feats[0].shape[-2:]
        weights = self.scale_logits.softmax(0)
        pixel = self.fuse(
            sum(w * F.interpolate(f, target, mode="bilinear", align_corners=False) for w, f in zip(weights, feats))
        )

        memory = F.adaptive_avg_pool2d(pixel, (self.memory_size, self.memory_size)).flatten(2).transpose(1, 2)
        memory = memory + sine_position(
            b, self.memory_size, self.memory_size, pixel.shape[1], pixel.device, pixel.dtype
        )

        q = self.query.weight.unsqueeze(0).expand(b, -1, -1)
        masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(q), pixel) / (pixel.shape[1] ** 0.5)

        for layer in self.decoder:
            q = layer(q, memory, masks)
            masks = torch.einsum("bqd,bdhw->bqhw", self.mask_embed(q), pixel) / (pixel.shape[1] ** 0.5)

        return {"pred_logits": self.class_head(q), "pred_masks": masks, "pixel_features": pixel}


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_inference": total, "params_trainable": trainable}
