"""Compact Re-Engineered Mask2Former Architecture for Q1 (Semantic) & Q2 (Instance).

Conforms strictly to Assignment II Master Execution Plan (Architecture & Model Budget):
- Backbone: ResNet-18 (~11.2M params).
- Pixel Decoder: FPN multi-scale fusion with 1/4-scale mask features (~2.4M params).
- Transformer Decoder: 6-layer Halo-Dilated Masked Attention with Q=100 queries (~4.9M params).
- Total Parameters: ~18.50M <= 19.504M (Bonus cap, E_i = 1.10x multiplier).
- Unified Checkpoint: Generates both Q1 semantic masks and Q2 instance masks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import ResNet18Backbone
from src.models.pixel_decoder import FPNPixelDecoder
from src.models.transformer_decoder import HaloMaskedTransformerDecoder
from src.constants import NUM_CLASSES


class CompactMask2Former(nn.Module):
    """Unified compact Mask2Former supporting both Semantic (Q1) and Instance (Q2) PCB segmentation."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 6,
        pretrained_backbone: bool = True,
        use_halo: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 1. Backbone: ResNet-18
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        # 2. Pixel Decoder: FPN
        self.pixel_decoder = FPNPixelDecoder(
            in_channels=self.backbone.out_channels,
            hidden_dim=hidden_dim,
            mask_dim=hidden_dim,
        )

        # 3. Transformer Decoder: Halo-Dilated Masked Attention
        self.transformer_decoder = HaloMaskedTransformerDecoder(
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=hidden_dim,
            nheads=8,
            num_layers=num_layers,
            dim_feedforward=512,
            halo_kernel=5,
            use_halo=use_halo,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count total, inference, and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # All weights in this architecture are active at inference
        return {
            "params_total": total,
            "params_inference": total,
            "params_trainable": trainable,
        }

    def forward(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward pass.
        
        Args:
            images: (B, 3, H, W) normalized image tensor.
            
        Returns:
            List of 6 prediction dicts (one per decoder layer for auxiliary Hungarian loss).
            Each has:
            - 'pred_logits': (B, Q, num_classes + 1)
            - 'pred_masks': (B, Q, H/4, W/4)
        """
        # 1. Backbone features
        features = self.backbone(images)

        # 2. Pixel decoder features
        mask_features, multi_scale_features = self.pixel_decoder(features)

        # 3. Transformer decoder queries
        outputs = self.transformer_decoder(mask_features, multi_scale_features)
        return outputs

    def predict_semantic(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.50,
    ) -> torch.Tensor:
        """Produce 2D semantic segmentation map (B, H, W) via weighted query probability combination.
        
        Formula: S_c(h, w) = sum_{q=1}^Q P_q(c) * sigma(M_q(h, w))
        """
        outputs = self.forward(images)
        final_out = outputs[-1]

        logits = final_out["pred_logits"]  # (B, Q, C + 1)
        masks = final_out["pred_masks"]    # (B, Q, H/4, W/4)

        B, _, H4, W4 = masks.shape
        H, W = images.shape[-2:]

        # Softmax over classes (excluding class 0 no-object)
        probs = F.softmax(logits, dim=-1)  # (B, Q, C + 1)
        class_probs = probs[:, :, 1:]      # (B, Q, C), 1..25 classes

        # Sigmoid on mask logits upsampled to full resolution
        masks_up = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False).sigmoid()  # (B, Q, H, W)

        # Semantic class probability map: S_c(h, w) = sum_q P_q(c) * M_q(h, w)
        # einsum: (B, Q, C) x (B, Q, H, W) -> (B, C, H, W)
        sem_probs = torch.einsum("bqc,bqhw->bchw", class_probs, masks_up)

        # Assign class with max probability if exceeding score threshold
        max_prob, max_class = sem_probs.max(dim=1)  # (B, H, W), values in 0..24
        pred_sem = max_class + 1                    # 1-indexed (1..25)
        pred_sem[max_prob < score_thresh] = 0        # Background 0

        return pred_sem

    def predict_instance(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.50,
        mask_thresh: float = 0.50,
    ) -> List[List[Dict[str, Any]]]:
        """Produce discrete instance segmentation masks for Q2."""
        outputs = self.forward(images)
        final_out = outputs[-1]

        logits = final_out["pred_logits"]  # (B, Q, C + 1)
        masks = final_out["pred_masks"]    # (B, Q, H/4, W/4)

        B, _, H4, W4 = masks.shape
        H, W = images.shape[-2:]

        probs = F.softmax(logits, dim=-1)
        class_probs, class_ids = probs[:, :, 1:].max(dim=-1)  # (B, Q)
        class_ids = class_ids + 1                             # 1..25

        masks_up = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False).sigmoid()

        batch_instances = []
        for b in range(B):
            insts = []
            for q in range(self.num_queries):
                score = float(class_probs[b, q].item())
                if score < score_thresh:
                    continue

                cid = int(class_ids[b, q].item())
                bin_mask = (masks_up[b, q] > mask_thresh).cpu().numpy().astype(bool)
                area = int(bin_mask.sum())
                if area < 4:
                    continue

                ys, xs = np.where(bin_mask)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                crop_mask = bin_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                insts.append({
                    "category_id": cid,
                    "score": score,
                    "bbox": bbox,
                    "crop_mask": crop_mask,
                    "area": area,
                })

            batch_instances.append(insts)

        return batch_instances
