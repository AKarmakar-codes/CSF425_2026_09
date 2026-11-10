"""Plot source-only training, validation, and candidate ablation evidence."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main(a):
    runs = [Path(x) for x in a.runs]
    docs = [json.loads((p / "semantic_stats.json").read_text()) for p in runs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, d in zip(runs, docs):
        hist = d["history"]
        ax.plot([r["epoch"] for r in hist], [r["train_loss"] for r in hist], label=f"{p.name} train")
        ax.plot([r["epoch"] for r in hist], [r["val_loss"] for r in hist], linestyle="--", label=f"{p.name} val")
    ax.set(xlabel="Epoch", ylabel="Hungarian query loss", title="Source-only training and validation loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=200)
    plt.close(fig)

    # 2. Source validation Macro-F1 curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for p, d in zip(runs, docs):
        ax.plot([r["epoch"] for r in d["history"]], [r["val_metrics"]["macro_f1"] for r in d["history"]], label=p.name)
    ax.set(xlabel="Epoch", ylabel="Source validation semantic macro-F1", title="Source validation model selection")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "source_macro_f1.png", dpi=200)
    plt.close(fig)

    # 3. Threshold sweeps
    best_i = max(range(len(docs)), key=lambda i: docs[i]["val_metrics"]["macro_f1"])
    best = docs[best_i]["val_metrics"]
    ibest = json.loads((runs[best_i] / "instance_stats.json").read_text())["val_metrics"]
    fig, ax = plt.subplots(figsize=(8, 5))
    s = best["threshold_sweep"]
    ax.plot([r["threshold"] for r in s], [r["macro_f1"] for r in s], marker="o", label="Semantic macro-F1")
    ax.plot([r["threshold"] for r in ibest["threshold_sweep"]], [r["f1_iou50"] for r in ibest["threshold_sweep"]], marker="s", label="Instance F1 @ IoU 0.50")
    ax.set(xlabel="Confidence threshold", ylabel="Score", title=f"Source validation threshold sweeps ({runs[best_i].name})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "threshold_sweeps.png", dpi=200)
    plt.close(fig)

    # 4. Parameter vs. Performance Capacity Ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    params = [d["params_inference"] / 1e6 for d in docs]
    scores = [d["val_metrics"]["macro_f1"] for d in docs]
    ax.scatter(params, scores, color="crimson", s=80, zorder=5)
    for p, d in zip(runs, docs):
        ax.annotate(p.name, (d["params_inference"] / 1e6, d["val_metrics"]["macro_f1"]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    ax.axvline(23.6, color="black", linestyle="--", alpha=0.7, label="P_max Cap (23.6M)")
    ax.axvline(19.504, color="forestgreen", linestyle=":", alpha=0.7, label="1.10x Bonus Sweet Spot (<=19.5M)")
    ax.set(xlabel="Inference parameters (M)", ylabel="Source validation macro-F1", title="Model Capacity vs Parameter Budget")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "parameter_ablation.png", dpi=200)
    plt.close(fig)

    # 5. Assignment I baseline comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Assignment 1: YOLOv8m-seg (Over-budget)", "Assignment 2: CompactMask2Former (Ours)"]
    param_counts = [27.25, docs[best_i]["params_inference"] / 1e6]
    bars = ax.bar(models, param_counts, color=["#b0bec5", "#1976d2"], width=0.4)
    ax.axhline(23.6, color="red", linestyle="--", label="Permitted Cap (23.6M)")
    ax.axhline(19.504, color="green", linestyle=":", label="1.10x Multiplier Ceiling (19.5M)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.2f}M", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Inference Parameters (M)")
    ax.set_title("Architecture Parameter Compliance vs Assignment I")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "assignment1_vs_assignment2_comparison.png", dpi=200)
    plt.close(fig)

    print(f"All 5 ablation plots saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
