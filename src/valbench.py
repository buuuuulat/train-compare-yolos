from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from train import build_run_name


def resolve_weights(
    model: str,
    train_cfg: Mapping[str, Any],
    run_dir: Path,
    name_keys: Iterable[str],
) -> str | Path:
    """Pick the best trained weight for a model, falling back to the base weight string."""
    run_name = build_run_name(model, train_cfg, name_keys)
    candidate = run_dir / run_name / "weights" / "best.pt"
    return candidate if candidate.exists() else model


def evaluate_model(weights: str | Path, val_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Run validation for a single set of weights and extract key metrics."""
    runner = YOLO(weights)
    metrics = runner.val(**val_cfg)

    box_metrics = getattr(metrics, "box", None)
    map50 = float(getattr(box_metrics, "map50", 0.0)) if box_metrics else None
    map5095 = float(getattr(box_metrics, "map", 0.0)) if box_metrics else None
    speed_ms = None
    if getattr(metrics, "speed", None):
        speed_ms = (
            metrics.speed.get("inference")
            or metrics.speed.get("inference_speed")
            or metrics.speed.get("inference_ms")
        )
        if speed_ms is not None:
            speed_ms = float(speed_ms)

    return {
        "map50": map50,
        "map50_95": map5095,
        "inference_ms": speed_ms,
    }


def export_tensorrt(weights: str | Path, export_cfg: Mapping[str, Any]) -> Path | None:
    """Export weights to TensorRT engine if enabled, returning the exported path."""
    if not export_cfg.get("enabled"):
        return None

    export_kwargs = {k: v for k, v in export_cfg.items() if k != "enabled" and v is not None}
    try:
        exported = YOLO(weights).export(**export_kwargs)
    except Exception as exc:  # pragma: no cover - runtime-dependent
        print(f"[warn] TensorRT export failed for {weights}: {exc}")
        return None
    return Path(exported)


def write_json(records: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "variant",
        "run_name",
        "map50",
        "map50_95",
        "inference_ms",
        "weights",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_bar(records: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    filtered = [r for r in records if r.get(metric) is not None]
    if not filtered:
        return
    labels = [f"{r['model']}-{r['variant']}" for r in filtered]
    values = [r[metric] for r in filtered]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scatter_speed_accuracy(
    records: list[dict[str, Any]],
    accuracy_key: str,
    speed_key: str,
    path: Path,
) -> None:
    filtered = [
        r for r in records if r.get(accuracy_key) is not None and r.get(speed_key) is not None
    ]
    if not filtered:
        return

    xs = [r[speed_key] for r in filtered]
    ys = [r[accuracy_key] for r in filtered]
    labels = [f"{r['model']}-{r['variant']}" for r in filtered]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, color="#55A868")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)
    ax.set_xlabel("Inference time (ms / image)")
    ax.set_ylabel("mAP@0.50")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def benchmark_all(
    models: Iterable[str],
    train_cfg: Mapping[str, Any],
    val_cfg: Mapping[str, Any],
    run_dir: Path,
    bench_dir: Path,
    name_keys: Iterable[str],
    export_cfg: Mapping[str, Any],
    weights_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate/benchmark models and save plots + tables.

    If weights_dir is provided, benchmarks all compatible files there (.pt/.engine/.onnx),
    otherwise uses trained weights resolved from models/train_cfg/run_dir.
    """
    bench_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = bench_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    export_variant = export_cfg.get("format", "engine")
    allowed_suffixes = {".pt", ".engine", ".onnx"}

    if weights_dir:
        weights_dir = Path(weights_dir)
        for path in sorted(weights_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            variant = path.suffix.lower().lstrip(".") or "weights"
            run_name = path.stem
            entries.append(
                {
                    "model": path.stem,
                    "variant": variant,
                    "run_name": run_name,
                    "weights": path,
                }
            )
            if (
                export_cfg.get("enabled")
                and variant != export_variant
                and path.suffix.lower() == ".pt"
            ):
                trt_path = export_tensorrt(path, export_cfg)
                if trt_path:
                    entries.append(
                        {
                            "model": path.stem,
                            "variant": export_variant,
                            "run_name": run_name,
                            "weights": trt_path,
                        }
                    )
    else:
        for model in models:
            weight_path = resolve_weights(model, train_cfg, run_dir, name_keys)
            run_name = build_run_name(model, train_cfg, name_keys)
            entries.append(
                {
                    "model": Path(model).stem,
                    "variant": "pt",
                    "run_name": run_name,
                    "weights": weight_path,
                }
            )

            trt_path = export_tensorrt(weight_path, export_cfg)
            if trt_path:
                entries.append(
                    {
                        "model": Path(model).stem,
                        "variant": export_variant,
                        "run_name": run_name,
                        "weights": trt_path,
                    }
                )

    records: list[dict[str, Any]] = []
    for entry in entries:
        print(f"Validating {entry['model']} [{entry['variant']}] (weights: {entry['weights']})")
        metrics = evaluate_model(entry["weights"], val_cfg)
        records.append({**entry, **metrics, "weights": str(entry["weights"])})

    write_json(records, bench_dir / "metrics.json")
    write_csv(records, bench_dir / "metrics.csv")
    plot_bar(records, "map50", "mAP@0.50", plots_dir / "accuracy_map50.png")
    plot_bar(records, "inference_ms", "Inference (ms / image)", plots_dir / "speed.png")
    plot_scatter_speed_accuracy(records, "map50", "inference_ms", plots_dir / "speed_vs_acc.png")

    return records
