"""Plot only measurements present in training/evaluation artifacts.

Run histories and checkpoints are accepted as inputs. Threshold and per-class
plots require explicit JSON files containing measured records. Missing data is
reported in the corresponding figure and in ``plot_manifest.json``; no curves,
metrics, or ablation values are synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import pickle
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    from_cwd = Path.cwd() / path
    if from_cwd.exists():
        return from_cwd.resolve()
    from_repo = REPO_ROOT / path
    if from_repo.exists():
        return from_repo.resolve()
    return from_cwd.resolve()


def number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def run_name_from_path(path: Path) -> str:
    name = path.stem
    for prefix in ("best_", "latest_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith("_history"):
        name = name[:-len("_history")]
    if name.endswith("_stats"):
        name = name[:-len("_stats")]
    return name


def add_record(runs: dict[str, dict[str, Any]], name: str, row: dict[str, Any]) -> None:
    epoch = number(row.get("epoch"))
    if epoch is None:
        return
    run = runs.setdefault(name, {"records": {}, "params_inference": None, "sources": set()})
    record = run["records"].setdefault(int(epoch), {})
    # Merge records from separate training chunks/checkpoints. Existing measured
    # values are retained unless the incoming row has a concrete replacement.
    for key, value in row.items():
        if key == "epoch" or value is None:
            continue
        if number(value) is not None:
            record[key] = number(value)
        elif isinstance(value, dict):
            for metric, metric_value in value.items():
                parsed = number(metric_value)
                if parsed is not None:
                    record[f"val_{metric}"] = parsed


def add_checkpoint_summary(runs: dict[str, dict[str, Any]], name: str, data: dict[str, Any]) -> None:
    run = runs.setdefault(name, {"records": {}, "params_inference": None, "sources": set()})
    params = number(data.get("params_inference"))
    if params is not None:
        run["params_inference"] = params

    epoch = number(data.get("epoch"))
    metrics = data.get("val_metrics")
    if epoch is not None and isinstance(metrics, dict):
        add_record(runs, name, {"epoch": int(epoch), **{f"val_{k}": v for k, v in metrics.items()}})

    utility = number(data.get("best_utility_C1"))
    if utility is not None and epoch is not None:
        run["records"].setdefault(int(epoch), {})["val_utility_C1"] = utility


class CheckpointMetadataUnpickler(pickle.Unpickler):
    """Read scalar metadata/history from a torch zip checkpoint without torch.

    Tensor rebuild calls are replaced with inert placeholders. Only the globals
    used by the checked-in checkpoint format are allowed.
    """

    def persistent_load(self, _persistent_id: Any) -> None:
        return None

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return lambda *args, **kwargs: None
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        raise pickle.UnpicklingError(f"Checkpoint metadata references unsupported global {module}.{name}")


def load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Extract checkpoint scalar/history data without deserializing tensor bytes."""
    with zipfile.ZipFile(path) as archive:
        pickle_members = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(pickle_members) != 1:
            raise ValueError(f"Expected one data.pkl in checkpoint archive, found {len(pickle_members)}")
        with archive.open(pickle_members[0]) as stream:
            data = CheckpointMetadataUnpickler(stream).load()
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported checkpoint structure in {path}")
    return data


def load_json_source(path: Path, runs: dict[str, dict[str, Any]]) -> str:
    data = json.loads(path.read_text())
    name = run_name_from_path(path)
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                add_record(runs, name, row)
        return name

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported JSON structure in {path}")

    name = str(data.get("run_name") or data.get("run_id") or name)
    history = data.get("history")
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                add_record(runs, name, row)
    if isinstance(data.get("val_metrics"), dict) or data.get("params_inference") is not None:
        add_checkpoint_summary(runs, name, data)
    if data.get("best_epoch") is not None:
        row = {
            "epoch": data.get("best_epoch"),
            "val_macro_f1": data.get("best_macro_f1"),
            "val_mIoU": data.get("best_mIoU"),
            "val_utility_C1": data.get("best_utility_C1"),
        }
        add_record(runs, name, row)
    return name


def load_checkpoint_source(path: Path, runs: dict[str, dict[str, Any]]) -> str:
    # Checkpoints are local training artifacts supplied by the user. Loading on
    # CPU avoids allocating a GPU just to read metadata/history.
    try:
        import torch
    except ModuleNotFoundError:
        data = load_checkpoint_metadata(path)
    else:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Compatibility with PyTorch versions lacking weights_only.
            data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported checkpoint structure in {path}")

    name = str(data.get("run_name") or data.get("run_id") or run_name_from_path(path))
    history = data.get("history")
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                add_record(runs, name, row)
    add_checkpoint_summary(runs, name, data)
    return name


def load_run_sources(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    runs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            if path.is_dir():
                children = sorted(path.glob("*_history.json")) + sorted(path.glob("best_run*.ckpt"))
                for child in children:
                    if child.suffix == ".json":
                        name = load_json_source(child, runs)
                    else:
                        name = load_checkpoint_source(child, runs)
                    runs[name]["sources"].add(str(child))
            elif path.suffix.lower() == ".json":
                name = load_json_source(path, runs)
                runs[name]["sources"].add(str(path))
            elif path.suffix.lower() in {".ckpt", ".pt", ".pth"}:
                name = load_checkpoint_source(path, runs)
                runs[name]["sources"].add(str(path))
            else:
                errors.append(f"Skipped unsupported input: {path}")
        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")
    return runs, errors


def ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high == low:
        pad = max(abs(low) * 0.05, 1.0)
        low, high = low - pad, high + pad
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def format_tick(value: float) -> str:
    return f"{value:.3g}"


def save_svg_and_png(out_dir: Path, filename: str, svg: str) -> None:
    svg_path = out_dir / Path(filename).with_suffix(".svg")
    png_path = out_dir / filename
    svg_path.write_text(svg)
    converter = shutil.which("magick") or shutil.which("convert")
    if converter is None:
        raise RuntimeError("Neither ImageMagick 'magick' nor 'convert' is available to render PNGs.")
    subprocess.run([converter, str(svg_path), str(png_path)], check=True, capture_output=True, text=True)


def render_chart(
    out_dir: Path,
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    *,
    x_ticks: list[tuple[float, str]] | None = None,
) -> None:
    """Render measured series to SVG, then convert to the assignment PNG format."""
    width = 1200
    legend_rows = max(1, math.ceil(len(series) / 3))
    height = 680 + max(0, legend_rows - 1) * 26
    left, right, top, bottom = 115, width - 45, 85, 500
    all_points = [point for item in series for segment in item["segments"] for point in segment]
    if not all_points:
        plot_unavailable(out_dir, filename, title, "No measured points were supplied.")
        return

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_ticks:
        x_min = min(x_min, min(t[0] for t in x_ticks))
        x_max = max(x_max, max(t[0] for t in x_ticks))
    if x_max == x_min:
        pad = max(abs(x_min) * 0.05, 1.0)
        x_min, x_max = x_min - pad, x_max + pad
    else:
        x_pad = (x_max - x_min) * 0.025
        x_min, x_max = x_min - x_pad, x_max + x_pad
    if y_max == y_min:
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        y_pad = (y_max - y_min) * 0.08
        y_min, y_max = y_min - y_pad, y_max + y_pad

    def px(x: float) -> float:
        return left + (x - x_min) * (right - left) / (x_max - x_min)

    def py(y: float) -> float:
        return bottom - (y - y_min) * (bottom - top) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="23" font-weight="bold">{html.escape(title)}</text>',
    ]
    if x_ticks is not None:
        tick_values = x_ticks
    else:
        data_low, data_high = min(xs), max(xs)
        raw_ticks = ticks(data_low, data_high)
        if all(abs(value - round(value)) < 1e-9 for value in xs):
            discrete = sorted({round(value) for value in raw_ticks})
            tick_values = [(value, str(value)) for value in discrete]
        else:
            tick_values = [(value, format_tick(value)) for value in raw_ticks]
    for value, label in tick_values:
        x = px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#dddddd" stroke-dasharray="3,5"/>')
        parts.append(f'<text x="{x:.2f}" y="{bottom + 24}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(str(label))}</text>')
    for value in ticks(y_min, y_max):
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#dddddd" stroke-dasharray="3,5"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{format_tick(value)}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<text x="{(left + right) / 2}" y="{bottom + 57}" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(x_label)}</text>')
    parts.append(f'<text x="30" y="{(top + bottom) / 2}" text-anchor="middle" transform="rotate(-90 30 {(top + bottom) / 2})" font-family="sans-serif" font-size="16">{html.escape(y_label)}</text>')

    for index, item in enumerate(series):
        color = item.get("color") or PALETTE[index % len(PALETTE)]
        for segment in item["segments"]:
            if not segment:
                continue
            d = " ".join(("M" if i == 0 else "L") + f" {px(x):.2f} {py(y):.2f}" for i, (x, y) in enumerate(segment))
            parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in segment:
                parts.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}"/>')

    legend_top = bottom + 100
    for index, item in enumerate(series):
        row, col = divmod(index, 3)
        x = left + col * ((right - left) / 3)
        y = legend_top + row * 25
        color = item.get("color") or PALETTE[index % len(PALETTE)]
        parts.append(f'<line x1="{x:.2f}" y1="{y - 5:.2f}" x2="{x + 28:.2f}" y2="{y - 5:.2f}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="13">{html.escape(item["label"])}</text>')
    parts.append("</svg>")
    save_svg_and_png(out_dir, filename, "\n".join(parts))


def plot_unavailable(out_dir: Path, filename: str, title: str, reason: str) -> None:
    """Replace any previous illustrative image with a clear no-data notice."""
    width, height = 1200, 680
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="120" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold">{html.escape(title)}</text>',
        f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" font-family="sans-serif" font-size="20">Not plotted from measured data</text>',
        f'<text x="{width / 2}" y="{height / 2 + 36}" text-anchor="middle" font-family="sans-serif" font-size="16">{html.escape(reason)}</text>',
        '</svg>',
    ])
    save_svg_and_png(out_dir, filename, svg)


def plot_training_loss(out_dir: Path, runs: dict[str, dict[str, Any]]) -> bool:
    series = []
    for name, run in sorted(runs.items()):
        records = run["records"]
        metric = "train_loss_total" if any("train_loss_total" in row for row in records.values()) else "train_loss"
        points = [(epoch, number(row.get(metric))) for epoch, row in sorted(records.items())]
        points = [(epoch, value) for epoch, value in points if value is not None]
        if not points:
            continue
        segments: list[list[tuple[float, float]]] = []
        for epoch, value in points:
            if not segments or epoch != segments[-1][-1][0] + 1:
                segments.append([])
            segments[-1].append((epoch, value))
        series.append({"label": name, "segments": segments})
    if series:
        render_chart(out_dir, "loss_curves.png", "Training loss from saved run records", "Recorded epoch", "Training loss", series)
        return True
    else:
        plot_unavailable(out_dir, "loss_curves.png", "Training loss", "No recorded training-loss values were supplied.")
        return False


def plot_validation_metrics(out_dir: Path, runs: dict[str, dict[str, Any]]) -> bool:
    metrics = (
        ("val_macro_f1", "Macro F1"),
        ("val_mIoU", "mIoU"),
        ("val_utility_C1", "C1 utility"),
    )
    series = []
    for name, run in sorted(runs.items()):
        records = run["records"]
        for key, label in metrics:
            points = [(epoch, number(row.get(key))) for epoch, row in sorted(records.items())]
            points = [(epoch, value) for epoch, value in points if value is not None]
            if not points:
                continue
            segments: list[list[tuple[float, float]]] = []
            for epoch, value in points:
                if not segments or epoch != segments[-1][-1][0] + 1:
                    segments.append([])
                segments[-1].append((epoch, value))
            series.append({"label": f"{name}: {label}", "segments": segments})
    if series:
        render_chart(out_dir, "validation_metrics.png", "Source validation metrics from saved run records", "Recorded epoch", "Validation score", series)
        return True
    else:
        plot_unavailable(out_dir, "validation_metrics.png", "Validation metrics", "No recorded validation metrics were supplied.")
        return False


def load_measurement_rows(path: Path, collection_key: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get(collection_key), list):
        rows = data[collection_key]
    elif isinstance(data, dict) and isinstance(data.get("val_metrics"), dict):
        rows = data["val_metrics"].get(collection_key, [])
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def plot_threshold_sweep(out_dir: Path, data_paths: list[Path]) -> bool:
    series = []
    for path in data_paths:
        try:
            rows = load_measurement_rows(path, "threshold_sweep")
        except Exception:
            continue
        rows = sorted((r for r in rows if number(r.get("threshold")) is not None), key=lambda r: float(r["threshold"]))
        for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("macro_f1", "Macro F1"), ("mIoU", "mIoU"), ("AP50", "AP50"), ("f1_iou50", "Instance F1")):
            points = [(number(r.get("threshold")), number(r.get(key))) for r in rows]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if points:
                series.append({"label": f"{path.stem}: {label}", "segments": [points]})
    if series:
        render_chart(out_dir, "threshold_sweeps.png", "Measured threshold sweep", "Threshold", "Measured score", series)
        return True
    else:
        plot_unavailable(out_dir, "threshold_sweeps.png", "Threshold sweep", "No measured threshold-sweep records were supplied.")
        return False


def plot_class_metrics(out_dir: Path, data_paths: list[Path]) -> bool:
    series = []
    labels_seen: list[str] = []
    for path in data_paths:
        try:
            rows = load_measurement_rows(path, "per_class")
        except Exception:
            continue
        rows = [r for r in rows if number(r.get("f1", r.get("macro_f1"))) is not None]
        if not rows:
            continue
        labels = [str(r.get("class_code", r.get("class", r.get("class_id", "?")))) for r in rows]
        values = [number(r.get("f1", r.get("macro_f1"))) for r in rows]
        labels_seen = labels
        series.append({"label": path.stem, "segments": [[(i, value) for i, value in enumerate(values) if value is not None]]})
    if series:
        tick_labels = [(i, label) for i, label in enumerate(labels_seen)]
        render_chart(out_dir, "class_metrics.png", "Per-class F1 from measured validation records", "Class", "Measured F1", series, x_ticks=tick_labels)
        return True
    else:
        plot_unavailable(out_dir, "class_metrics.png", "Per-class metrics", "No measured per-class records were supplied.")
        return False


def plot_ablation_tradeoff(out_dir: Path, runs: dict[str, dict[str, Any]]) -> bool:
    candidates = []
    for name, run in sorted(runs.items()):
        params = number(run.get("params_inference"))
        utility = None
        for row in run["records"].values():
            utility = number(row.get("val_utility_C1"))
            if utility is not None:
                break
        if params is not None and utility is not None:
            candidates.append((name, params, utility))
    if len(candidates) < 2:
        plot_unavailable(
            out_dir,
            "ablation_tradeoff.png",
            "Ablation trade-off",
            "At least two runs with recorded parameter counts and validation utility are needed.",
        )
        return False

    series = [{"label": name, "segments": [[(params, utility)]]} for name, params, utility in candidates]
    render_chart(out_dir, "ablation_tradeoff.png", "Ablation trade-off from saved run checkpoints", "Inference parameters", "Recorded validation C1 utility", series)
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot only recorded training/evaluation measurements.")
    parser.add_argument("--out", default="results/plots", help="Output directory for plot images and manifest")
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Run history JSONs, checkpoints, or directories containing them. Defaults to recorded Run 1–3 artifacts under checkpoints/.",
    )
    parser.add_argument(
        "--threshold-data",
        nargs="*",
        default=[],
        help="JSON files with measured threshold_sweep rows (each row needs threshold and measured metrics).",
    )
    parser.add_argument(
        "--class-data",
        nargs="*",
        default=[],
        help="JSON files with measured per_class rows (each row needs class_code/class and f1).",
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.runs is None:
        checkpoint_dir = REPO_ROOT / "checkpoints"
        run_paths = sorted(checkpoint_dir.glob("*_history.json")) + sorted(checkpoint_dir.glob("best_run*.ckpt"))
    else:
        run_paths = [resolve_path(value) for value in args.runs]
    threshold_paths = [resolve_path(value) for value in args.threshold_data]
    class_paths = [resolve_path(value) for value in args.class_data]

    runs, input_errors = load_run_sources(run_paths)
    generated = {
        "loss_curves.png": plot_training_loss(out_dir, runs),
        "validation_metrics.png": plot_validation_metrics(out_dir, runs),
        "threshold_sweeps.png": plot_threshold_sweep(out_dir, threshold_paths),
        "class_metrics.png": plot_class_metrics(out_dir, class_paths),
        "ablation_tradeoff.png": plot_ablation_tradeoff(out_dir, runs),
    }

    input_paths = sorted({*run_paths, *threshold_paths, *class_paths}, key=str)
    manifest = {
        "input_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in input_paths
            if path.is_file()
        ],
        "runs": {
            name: {
                "epochs_with_records": sorted(run["records"]),
                "params_inference": run["params_inference"],
                "sources": sorted(run["sources"]),
            }
            for name, run in sorted(runs.items())
        },
        "generated_from_measurements": generated,
        "input_errors": input_errors,
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Plots written to: {out_dir}")
    for filename, has_data in generated.items():
        print(f"{'Plotted measured data' if has_data else 'No measured data; wrote notice'}: {filename}")
    for error in input_errors:
        print(f"Input note: {error}")


if __name__ == "__main__":
    main()
