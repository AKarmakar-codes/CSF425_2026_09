"""Train the shared query-mask student on prepared source tiles only."""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CompactMask2Former, parameter_counts
from loss import query_loss
from data import SourceTileDataset, list_collate


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    data = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}; choose a fresh run directory")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("format") != "csf425-a2-tiles-v1" or manifest.get("heldout_official_test_used") is not False:
        raise ValueError("Expected Assignment II source-preparation manifest with official test excluded")
    if manifest.get("external_images", 0) > manifest.get("original_source_train_images", 0) * 0.5:
        raise ValueError("External distinct images exceed the 1.5x allowance")
    if set(manifest.get("train_board_groups", [])) & set(manifest.get("val_board_groups", [])):
        raise ValueError("Physical PCB identity leakage between train and validation")

    split_id = hashlib.sha256(
        json.dumps(
            {
                "seed": manifest["seed"],
                "train": sorted(manifest["train_board_groups"]),
                "val": sorted(manifest["val_board_groups"]),
                "archive": manifest.get("source_archive_sha256"),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    train_ds = SourceTileDataset(data, "train", augment=True, seed=args.seed)
    val_ds = SourceTileDataset(data, "val", augment=False, seed=args.seed)
    train_groups = set(manifest.get("train_board_groups", []))
    val_groups = set(manifest.get("val_board_groups", []))
    if any(r.get("board_id") not in train_groups for r in train_ds.rows):
        raise ValueError("Train tile board identity is not in the declared train split")
    if any(r.get("board_id") not in val_groups for r in val_ds.rows):
        raise ValueError("Validation tile board identity is not in the declared validation split")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=list_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=list_collate,
        pin_memory=(device.type == "cuda"),
    )

    model = CompactMask2Former(
        num_queries=args.queries,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.layers,
        pretrained_backbone=args.imagenet_pretrained,
        masked_attention=not args.disable_masked_attention,
        normalize_imagenet=args.imagenet_pretrained,
        memory_size=args.memory_size,
    ).to(device)

    counts = parameter_counts(model)
    if counts["params_inference"] > args.parameter_ceiling:
        raise RuntimeError(
            f"Model has {counts['params_inference']:,} parameters, over ceiling {args.parameter_ceiling:,}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = []
    best_sem = -1.0
    best_inst = -1.0
    best_sem_row = None
    best_inst_row = None
    threshold_grid = [i / 10 for i in range(1, 11)]
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def epoch(loader, training):
        model.train(training)
        loss_sum = 0.0
        terms = {}
        forward_seconds = 0.0
        confusions = torch.zeros((len(threshold_grid), 26, 26), dtype=torch.int64)
        inst_counts = torch.zeros((len(threshold_grid), 3), dtype=torch.int64)
        inst_class_counts = torch.zeros((len(threshold_grid), 25, 3), dtype=torch.int64)
        inst_overlap = torch.zeros((len(threshold_grid), 2), dtype=torch.float64)
        ap_records = []
        gt_per_class = torch.zeros(25, dtype=torch.int64)

        for rows in loader:
            images = torch.stack([r["image"] for r in rows]).to(device).float() / 255.0
            targets = []
            for r in rows:
                labels = r["labels"].to(device)
                masks = r["masks"].to(device)
                targets.append({"labels": labels, "masks": masks})
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=(device.type == "cuda")):
                fstart = time.perf_counter()
                outputs = model(images)
                if not training:
                    forward_seconds += time.perf_counter() - fstart
                loss, detail = query_loss(outputs, targets, boundary_weight=args.boundary_weight)
            if not training:
                pred_masks = F.interpolate(
                    outputs["pred_masks"].float(), (128, 128), mode="bilinear", align_corners=False
                ).sigmoid()
                cls_prob = outputs["pred_logits"].float().softmax(-1)[:, :, :25]
                class_scores = torch.einsum("bqc,bqhw->bchw", cls_prob, pred_masks)
                max_score, pred_class = class_scores.max(1)

                for bi, target in enumerate(targets):
                    gt_sem = torch.zeros((128, 128), dtype=torch.long, device=device)
                    small = (
                        F.interpolate(target["masks"][:, None].float(), (128, 128), mode="nearest")[:, 0]
                        if len(target["labels"])
                        else torch.zeros((0, 128, 128), device=device)
                    )
                    for mi, label in enumerate(target["labels"]):
                        gt_sem[small[mi] > 0.5] = label + 1
                    for ti, threshold in enumerate(threshold_grid):
                        pr = pred_class[bi].clone() + 1
                        pr[max_score[bi] < threshold] = 0
                        cm = torch.bincount((gt_sem.flatten() * 26 + pr.flatten()).cpu(), minlength=26 * 26).reshape(
                            26, 26
                        )
                        confusions[ti] += cm
                    gt_masks = small.bool()
                    gt_labels = target["labels"]
                    query_conf, query_label = cls_prob[bi].max(-1)
                    query_masks = pred_masks[bi] > 0.5
                    image_class_ious = {}
                    for cid in range(25):
                        gi = torch.where(gt_labels == cid)[0]
                        qi = torch.where(query_label == cid)[0]
                        gt_per_class[cid] += len(gi)
                        if len(qi) and len(gi):
                            pm = query_masks[qi].flatten(1).float()
                            gm = gt_masks[gi].flatten(1).float()
                            inter = pm @ gm.T
                            union = pm.sum(1)[:, None] + gm.sum(1)[None, :] - inter
                            image_class_ious[cid] = (qi, gi, inter / union.clamp_min(1))
                        else:
                            image_class_ious[cid] = (qi, gi, None)

                        # Ranked, greedy one-to-one matching for source-validation AP50.
                        ranked = qi[torch.argsort(query_conf[qi], descending=True)]
                        unmatched = set(range(len(gi)))
                        iou_by_q = (
                            {
                                int(q): image_class_ious[cid][2][int((qi == q).nonzero()[0])]
                                .detach()
                                .cpu()
                                .tolist()
                                for q in qi
                            }
                            if len(qi) and len(gi)
                            else {}
                        )
                        for q in ranked:
                            hit = False
                            if unmatched:
                                row = iou_by_q[int(q)]
                                bestj = max(unmatched, key=lambda j: row[j])
                                if row[bestj] >= 0.5:
                                    unmatched.remove(bestj)
                                    hit = True
                            ap_records.append({"class_id": cid, "score": float(query_conf[q]), "tp": int(hit)})
                    for ti, threshold in enumerate(threshold_grid):
                        qkeep = query_conf >= threshold
                        tp = fp = fn = 0
                        for cid in range(25):
                            all_qi, gi, iou = image_class_ious[cid]
                            qi = all_qi[qkeep[all_qi]]
                            if len(qi) and len(gi):
                                inds = torch.where(qkeep[all_qi])[0]
                                filtered = iou[inds]
                                rr, cc = linear_sum_assignment((-filtered).cpu().numpy())
                                matched_ious = filtered[rr, cc]
                                valid = matched_ious >= 0.5
                                matched = int(valid.sum())
                                tp += matched
                                fp += len(qi) - matched
                                fn += len(gi) - matched
                                selected_iou = matched_ious[valid]
                                inst_overlap[ti, 0] += selected_iou.sum().double().cpu()
                                inst_overlap[ti, 1] += (2 * selected_iou / (1 + selected_iou)).sum().double().cpu()
                            else:
                                fp += len(qi)
                                fn += len(gi)
                            inst_class_counts[ti, cid] += torch.tensor(
                                [
                                    matched if len(qi) and len(gi) else 0,
                                    len(qi) - (matched if len(qi) and len(gi) else 0),
                                    len(gi) - (matched if len(qi) and len(gi) else 0),
                                ]
                            )
                        inst_counts[ti] += torch.tensor([tp, fp, fn])

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.detach())
            for k, v in detail.items():
                terms[k] = terms.get(k, 0.0) + v

        n = max(1, len(loader))
        metrics = {k: v / n for k, v in terms.items()}
        if not training:
            metrics["inference_seconds_per_tile"] = forward_seconds / n
            scored = []
            for ti, threshold in enumerate(threshold_grid):
                cm = confusions[ti].float()
                tp = cm.diag()
                fp = cm.sum(0) - tp
                fn = cm.sum(1) - tp
                supported = cm.sum(1)[1:] > 0
                f1 = (2 * tp[1:] / (2 * tp[1:] + fp[1:] + fn[1:]).clamp_min(1e-8))[supported]
                iou = (tp[1:] / (tp[1:] + fp[1:] + fn[1:]).clamp_min(1e-8))[supported]
                scored.append(
                    {
                        "threshold": threshold,
                        "macro_f1": float(f1.mean()) if f1.numel() else 0.0,
                        "macro_iou": float(iou.mean()) if iou.numel() else 0.0,
                    }
                )
            best_threshold = max(scored, key=lambda x: (x["macro_f1"], x["macro_iou"]))["threshold"]

            # Decoupled per-class threshold calibration:
            class_calibrated_thresholds = {}
            for cid in range(1, 26):
                class_best_f1 = -1.0
                class_best_thresh = best_threshold
                for ti, thresh in enumerate(threshold_grid):
                    cm_t = confusions[ti].float()
                    tp_c = float(cm_t[cid, cid])
                    fp_c = float(cm_t[:, cid].sum() - tp_c)
                    fn_c = float(cm_t[cid, :].sum() - tp_c)
                    f1_c = (2 * tp_c) / max(1e-8, 2 * tp_c + fp_c + fn_c)
                    if f1_c > class_best_f1:
                        class_best_f1 = f1_c
                        class_best_thresh = thresh
                class_calibrated_thresholds[str(cid)] = class_best_thresh

            cm = confusions[threshold_grid.index(best_threshold)].float()
            tp = cm.diag()
            fp = cm.sum(0) - tp
            fn = cm.sum(1) - tp
            classwise = {}
            for cid in range(1, 26):
                if cm[cid].sum() == 0:
                    continue
                classwise[str(cid)] = {
                    "precision": float(tp[cid] / (tp[cid] + fp[cid]).clamp_min(1e-8)),
                    "recall": float(tp[cid] / (tp[cid] + fn[cid]).clamp_min(1e-8)),
                    "f1": float(2 * tp[cid] / (2 * tp[cid] + fp[cid] + fn[cid]).clamp_min(1e-8)),
                    "iou": float(tp[cid] / (tp[cid] + fp[cid] + fn[cid]).clamp_min(1e-8)),
                    "support": int(cm[cid].sum()),
                    "calibrated_threshold": class_calibrated_thresholds.get(str(cid), best_threshold),
                }

            inst_sweep = []
            for ti, threshold in enumerate(threshold_grid):
                tp_val, fp_val, fn_val = map(int, inst_counts[ti])
                inst_sweep.append(
                    {
                        "threshold": threshold,
                        "precision": tp_val / max(1, tp_val + fp_val),
                        "recall": tp_val / max(1, tp_val + fn_val),
                        "f1": 2 * tp_val / max(1, 2 * tp_val + fp_val + fn_val),
                        "tp": tp_val,
                        "fp": fp_val,
                        "fn": fn_val,
                        "matched_mean_iou": float(inst_overlap[ti, 0] / max(1, tp_val)),
                        "matched_mean_dice": float(inst_overlap[ti, 1] / max(1, tp_val)),
                    }
                )
            best_instance = max(inst_sweep, key=lambda x: (x["f1"], -x["threshold"]))
            ii = threshold_grid.index(best_instance["threshold"])
            instance_classwise = {}
            inst_class_calibrated_thresholds = {}
            for cid in range(25):
                best_c_f1 = -1.0
                best_c_thresh = best_instance["threshold"]
                for ti, thresh in enumerate(threshold_grid):
                    tp_ci, fp_ci, fn_ci = map(int, inst_class_counts[ti, cid])
                    f1_ci = (2 * tp_ci) / max(1, 2 * tp_ci + fp_ci + fn_ci)
                    if f1_ci > best_c_f1:
                        best_c_f1 = f1_ci
                        best_c_thresh = thresh
                inst_class_calibrated_thresholds[str(cid + 1)] = best_c_thresh

                tp_ci, fp_ci, fn_ci = map(int, inst_class_counts[ii, cid])
                if tp_ci + fp_ci + fn_ci:
                    instance_classwise[str(cid + 1)] = {
                        "tp": tp_ci,
                        "fp": fp_ci,
                        "fn": fn_ci,
                        "precision": tp_ci / max(1, tp_ci + fp_ci),
                        "recall": tp_ci / max(1, tp_ci + fn_ci),
                        "f1": 2 * tp_ci / max(1, 2 * tp_ci + fp_ci + fn_ci),
                        "calibrated_threshold": best_c_thresh,
                    }

            aps = {}
            for cid in range(25):
                total = int(gt_per_class[cid])
                if not total:
                    continue
                ranked = sorted((r for r in ap_records if r["class_id"] == cid), key=lambda r: r["score"], reverse=True)
                tp_arr = np.cumsum([r["tp"] for r in ranked])
                fp_arr = np.cumsum([1 - r["tp"] for r in ranked])
                rec = tp_arr / max(1, total)
                prec = tp_arr / np.maximum(1, tp_arr + fp_arr)
                aps[str(cid + 1)] = float(
                    np.mean([np.max(prec[rec >= pt], initial=0.0) for pt in np.linspace(0, 1, 101)])
                )

            metrics.update(
                {
                    "semantic_threshold_sweep": scored,
                    "selected_threshold": best_threshold,
                    "class_calibrated_thresholds": class_calibrated_thresholds,
                    "macro_f1": max(x["macro_f1"] for x in scored),
                    "macro_iou": max(x["macro_iou"] for x in scored),
                    "semantic_classwise": classwise,
                    "instance_threshold_sweep": inst_sweep,
                    "selected_instance_threshold": best_instance["threshold"],
                    "instance_class_calibrated_thresholds": inst_class_calibrated_thresholds,
                    "instance_f1": best_instance["f1"],
                    "instance_classwise": instance_classwise,
                    "instance_ap50_by_class": aps,
                    "instance_ap50": float(np.mean(list(aps.values()))) if aps else 0.0,
                }
            )
        return loss_sum / n, metrics

    for ep in range(1, args.epochs + 1):
        train_ds.epoch = ep
        tr, _ = epoch(train_loader, True)
        va, val_terms = epoch(val_loader, False)
        row = {"epoch": ep, "train_loss": tr, "val_loss": va, "val_metrics": val_terms}
        history.append(row)
        print(json.dumps(row), flush=True)

        base = {
            "format": "csf425-a2-querymask-v1",
            "model_config": {
                "num_classes": 25,
                "num_queries": args.queries,
                "hidden_dim": args.hidden_dim,
                "decoder_layers": args.layers,
                "pretrained_backbone": False,
                "masked_attention": not args.disable_masked_attention,
                "normalize_imagenet": args.imagenet_pretrained,
                "memory_size": args.memory_size,
            },
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }
        provenance = {
            "seed": args.seed,
            "epoch": ep,
            "data_source": "official source train, source validation only",
            "parameter_counts": counts,
            "validation_loss": va,
            "threshold_source": "source validation only",
            "target_data_used": False,
            "imagenet_pretrained": args.imagenet_pretrained,
        }
        if val_terms["macro_f1"] > best_sem:
            best_sem = val_terms["macro_f1"]
            best_sem_row = row
            torch.save(
                {
                    **base,
                    "metadata": {
                        **provenance,
                        "validation_macro_f1": best_sem,
                        "threshold": val_terms["selected_threshold"],
                        "class_thresholds": val_terms["class_calibrated_thresholds"],
                    },
                },
                out / "best_semantic.ckpt",
            )
        if val_terms["instance_f1"] > best_inst:
            best_inst = val_terms["instance_f1"]
            best_inst_row = row
            torch.save(
                {
                    **base,
                    "metadata": {
                        **provenance,
                        "validation_instance_f1": best_inst,
                        "instance_threshold": val_terms["selected_instance_threshold"],
                        "instance_class_thresholds": val_terms["instance_class_calibrated_thresholds"],
                    },
                },
                out / "best_instance.ckpt",
            )

    runtime = time.perf_counter() - start
    memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    semvm = best_sem_row["val_metrics"]
    instvm = best_inst_row["val_metrics"]
    common = {
        "run_id": out.name,
        "seed": args.seed,
        "model_config": {
            "num_classes": 25,
            "num_queries": args.queries,
            "hidden_dim": args.hidden_dim,
            "decoder_layers": args.layers,
            "imagenet_pretrained": args.imagenet_pretrained,
            "masked_attention": not args.disable_masked_attention,
            "memory_size": args.memory_size,
        },
        **counts,
        "runtime_seconds": runtime,
        "peak_gpu_memory_bytes": memory,
        "data_source": "official training boards, physical-board validation split",
        "checkpoint": "task-specific best checkpoint",
        "history": history,
        "teacher_version": None,
        "pseudo_metrics": None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
    }
    reference = "source validation masks; instances derived by 8-connected components per exact class color"
    common["split_id"] = split_id
    sem = {
        **common,
        "task": "semantic",
        "reference_type": reference,
        "train_loss": best_sem_row["train_loss"],
        "val_loss": best_sem_row["val_loss"],
        "best_epoch": best_sem_row["epoch"],
        "threshold": semvm["selected_threshold"],
        "class_thresholds": semvm["class_calibrated_thresholds"],
        "val_metrics": {
            "macro_f1": semvm["macro_f1"],
            "macro_iou": semvm["macro_iou"],
            "threshold_sweep": semvm["semantic_threshold_sweep"],
            "classwise": semvm["semantic_classwise"],
            "inference_seconds_per_tile": semvm["inference_seconds_per_tile"],
        },
    }
    inst = {
        **common,
        "task": "instance",
        "reference_type": reference,
        "train_loss": best_inst_row["train_loss"],
        "val_loss": best_inst_row["val_loss"],
        "best_epoch": best_inst_row["epoch"],
        "threshold": instvm["selected_instance_threshold"],
        "instance_class_thresholds": instvm["instance_class_calibrated_thresholds"],
        "val_metrics": {
            "f1_iou50": instvm["instance_f1"],
            "threshold_sweep": instvm["instance_threshold_sweep"],
            "classwise": instvm["instance_classwise"],
            "ap50_by_class": instvm["instance_ap50_by_class"],
            "ap50": instvm["instance_ap50"],
            "inference_seconds_per_tile": instvm["inference_seconds_per_tile"],
        },
    }
    (out / "semantic_stats.json").write_text(json.dumps(sem, indent=2, allow_nan=False))
    (out / "instance_stats.json").write_text(json.dumps(inst, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Assignment II prepared source tiles")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--memory-size", type=int, default=32)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--imagenet-pretrained",
        action="store_true",
        help="Initialize ResNet-18 from torchvision ImageNet weights; downloaded/cached weights must be cited",
    )
    p.add_argument(
        "--disable-masked-attention",
        action="store_true",
        help="Controlled ablation: use unmasked query cross-attention",
    )
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--parameter-ceiling", type=int, default=23600000)
    run(p.parse_args())
