"""Small query-mask criterion with Hungarian assignment and boundary-aware loss."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def boundary_loss(pred_mask, gt_mask):
    """Laplacian edge penalty to enforce sharp component boundaries."""
    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=pred_mask.device).view(1, 1, 3, 3)
    pred_edge = F.conv2d(pred_mask[:, None], lap, padding=1).abs()
    gt_edge = F.conv2d(gt_mask[:, None], lap, padding=1).abs()
    return F.l1_loss(pred_edge, gt_edge)


def query_loss(outputs, targets, no_object_weight=0.1, boundary_weight=1.0):
    logits, masks = outputs["pred_logits"], outputs["pred_masks"]
    batch, queries = logits.shape[:2]
    class_target = torch.full((batch, queries), logits.shape[-1] - 1, dtype=torch.long, device=logits.device)
    loss_mask = logits.sum() * 0.0
    loss_dice = logits.sum() * 0.0
    loss_bound = logits.sum() * 0.0
    matched = 0

    for b, target in enumerate(targets):
        labels, gt = target["labels"], target["masks"].float()
        if labels.numel() == 0:
            continue
        pred_small = F.interpolate(masks[b : b + 1], (128, 128), mode="bilinear", align_corners=False)[0].sigmoid()
        gt_small = F.interpolate(gt[:, None], (128, 128), mode="nearest")[:, 0]
        prob = logits[b].softmax(-1)[:, :-1]
        cls_cost = -prob[:, labels]

        # Low-resolution matching cost bounds Q x N x pixels memory.
        p = F.adaptive_avg_pool2d(pred_small[:, None], (32, 32))[:, 0].flatten(1).clamp(1e-5, 1 - 1e-5)
        g = F.adaptive_max_pool2d(gt_small[:, None], (32, 32))[:, 0].flatten(1)
        bce_cost = -(p.log()[:, None, :] * g[None, :, :] + (1 - p).log()[:, None, :] * (1 - g)[None, :, :]).mean(-1)
        dice_cost = 1 - (2 * p @ g.T + 1) / (p.sum(-1)[:, None] + g.sum(-1)[None] + 1)

        rows, cols = linear_sum_assignment((cls_cost + bce_cost + dice_cost).detach().cpu().numpy())
        rows = torch.as_tensor(rows, device=logits.device)
        cols = torch.as_tensor(cols, device=logits.device)
        class_target[b, rows] = labels[cols]

        pm, gm = pred_small[rows], gt_small[cols]
        loss_mask = loss_mask + F.binary_cross_entropy(pm.clamp(1e-5, 1 - 1e-5), gm)
        loss_dice = loss_dice + (1 - (2 * (pm * gm).sum((1, 2)) + 1) / (pm.sum((1, 2)) + gm.sum((1, 2)) + 1)).sum()
        if boundary_weight > 0 and len(rows) > 0:
            loss_bound = loss_bound + boundary_loss(pm, gm) * len(rows)
        matched += len(rows)

    weight = torch.ones(logits.shape[-1], device=logits.device)
    weight[-1] = no_object_weight
    loss_cls = F.cross_entropy(logits.transpose(1, 2), class_target, weight=weight)
    denom = max(1, matched)
    total_loss = loss_cls + 2.0 * loss_mask / denom + 2.0 * loss_dice / denom + boundary_weight * loss_bound / denom

    return total_loss, {
        "loss_cls": float(loss_cls.detach()),
        "loss_mask": float((loss_mask / denom).detach()),
        "loss_dice": float((loss_dice / denom).detach()),
        "loss_boundary": float((loss_bound / denom).detach()),
    }
