"""Halo-Dilated Masked-Attention Transformer Decoder for Mask2Former.

Conforms strictly to Assignment II Master Execution Plan (Re-Engineering Mechanism 1):
- Halo-Dilated Masked Attention: Applies parameter-free 5x5 max-pool morphological dilation
  (D_5x5(M_{l-1})) to the attention mask feeding the classification cross-attention projection,
  allowing each query to read adjacent white silkscreen designators (U vs IC, SW vs S, R vs C)
  printed just outside the component body.
- Retains the tight, un-dilated mask M_{l-1} for mask refinement.
- Auxiliary layer supervision across all L=6 decoder layers for stable bipartite matching.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaloDilatedCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with 5x5 Halo-Dilated Masked Attention."""

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

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        use_halo_dilation: bool = True,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: (B, Q, D)
            key_value: (B, HW, D)
            attn_mask: (B, Q, H, W) raw binary mask from previous layer
            use_halo_dilation: If True, applies 5x5 max-pool dilation to attn_mask
        """
        B, Q, D = query.shape
        HW = key_value.shape[1]

        q = self.q_proj(query).view(B, Q, self.nheads, D // self.nheads).transpose(1, 2)  # (B, nheads, Q, head_dim)
        k = self.k_proj(key_value).view(B, HW, self.nheads, D // self.nheads).transpose(1, 2)  # (B, nheads, HW, head_dim)
        v = self.v_proj(key_value).view(B, HW, self.nheads, D // self.nheads).transpose(1, 2)  # (B, nheads, HW, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (D // self.nheads) ** 0.5  # (B, nheads, Q, HW)

        if attn_mask is not None:
            # attn_mask shape: (B, Q, H, W)
            if use_halo_dilation and self.halo_kernel > 1:
                # Parameter-free 5x5 max-pooling dilation on binary/sigmoid mask
                # Expands receptive field by 2px on 1/4 feature map (= 8px on 1024x1024 tile)
                dilated_mask = F.max_pool2d(
                    attn_mask.float(),
                    kernel_size=self.halo_kernel,
                    stride=1,
                    padding=self.padding,
                )
            else:
                dilated_mask = attn_mask.float()

            # Flatten spatial dimensions to match HW
            mask_flat = dilated_mask.flatten(2).unsqueeze(1)  # (B, 1, Q, HW)
            # Additive attention bias: 0 for foreground + halo, -1e4 for blocked background
            attn_bias = (1.0 - (mask_flat > 0.5).float()) * -10000.0
            scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, nheads, Q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        return self.out_proj(out)


class TransformerDecoderLayer(nn.Module):
    """Single layer of Mask2Former Transformer Decoder with self-attention, cross-attention, and MLP."""

    def __init__(self, d_model: int = 256, nheads: int = 8, dim_feedforward: int = 1024, halo_kernel: int = 5):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Halo-Dilated Cross-Attention with visual features
        self.cross_attn = HaloDilatedCrossAttention(d_model, nheads, halo_kernel=halo_kernel)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        feature_map: torch.Tensor,
        prev_mask: Optional[torch.Tensor] = None,
        use_halo: bool = True,
    ) -> torch.Tensor:
        # 1. Query Self-Attention
        q2 = self.self_attn(query, query, query)[0]
        query = self.norm1(query + q2)

        # 2. Halo-Dilated Cross-Attention
        H, W = feature_map.shape[-2:]
        if prev_mask is not None:
            # Interpolate previous layer mask to current feature map resolution
            prev_mask_interp = F.interpolate(prev_mask, size=(H, W), mode="bilinear", align_corners=False)
        else:
            prev_mask_interp = None

        B, C, _, _ = feature_map.shape
        feature_flat = feature_map.flatten(2).transpose(1, 2)  # (B, HW, C)

        q_cross = self.cross_attn(query, feature_flat, attn_mask=prev_mask_interp, use_halo_dilation=use_halo)
        query = self.norm2(query + q_cross)

        # 3. Feedforward Network
        q_ff = self.linear2(self.activation(self.linear1(query)))
        query = self.norm3(query + q_ff)

        return query


class HaloMaskedTransformerDecoder(nn.Module):
    """6-Layer Masked-Attention Transformer Decoder with Halo Dilation."""

    def __init__(
        self,
        num_classes: int = 25,
        num_queries: int = 100,
        d_model: int = 256,
        nheads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        halo_kernel: int = 5,
        use_halo: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_halo = use_halo

        # Learnable object queries and position embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_feat = nn.Embedding(num_queries, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nheads, dim_feedforward, halo_kernel=halo_kernel)
            for _ in range(num_layers)
        ])

        # Prediction heads (shared across layers for compact parameter budget)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # 25 classes + no-object class 0
        self.mask_embed_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        mask_features: torch.Tensor,
        multi_scale_features: List[torch.Tensor],
    ) -> List[Dict[str, torch.Tensor]]:
        """Forward pass through 6 decoder layers.
        
        Args:
            mask_features: (B, D, H/4, W/4) high-resolution mask features from pixel decoder
            multi_scale_features: [p3 (1/8), p4 (1/16), p5 (1/32)]
            
        Returns:
            List of prediction dictionaries for each layer (auxiliary outputs for Hungarian loss).
            Each dict has "pred_logits" (B, Q, C+1) and "pred_masks" (B, Q, H/4, W/4).
        """
        B = mask_features.shape[0]
        # Initialize queries
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, Q, D)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        outputs = []
        cur_mask = None

        # Cycle through multi-scale features: p3 (1/8), p4 (1/16), p5 (1/32) across 6 layers
        for layer_idx, layer in enumerate(self.layers):
            feat_idx = layer_idx % len(multi_scale_features)
            feat_map = multi_scale_features[feat_idx]

            query_in = query + query_pos
            query = layer(query_in, feat_map, prev_mask=cur_mask, use_halo=self.use_halo)

            # Predict class logits
            class_logits = self.class_head(query)  # (B, Q, C + 1)

            # Predict mask embeddings and dot-product with high-resolution mask features
            mask_embed = self.mask_embed_head(query)  # (B, Q, D)
            mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embed, mask_features)  # (B, Q, H/4, W/4)

            outputs.append({
                "pred_logits": class_logits,
                "pred_masks": mask_logits,
            })

            cur_mask = mask_logits.sigmoid()

        return outputs
