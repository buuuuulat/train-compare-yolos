from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ultralytics import YOLO


def build_run_name(model: str, cfg: Mapping[str, Any], name_keys: Iterable[str]) -> str:
    """Compose a folder-friendly name from model and selected hyperparameters."""
    stem = Path(model).stem
    parts = [
        f"{key}-{cfg[key]}"
        for key in name_keys
        if key in cfg
    ]
    params_id = "_".join(parts) if parts else "default"
    return f"{stem}_{params_id}"


def train_model(
    model: str,
    cfg: Mapping[str, Any],
    run_dir: Path,
    name_keys: Iterable[str],
) -> Path:
    """Train a single model and return the path to its best weight."""
    run_name = build_run_name(model, cfg, name_keys)
    print(f"Training {model} -> {run_dir / run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = YOLO(model)
    runner.train(project=run_dir, name=run_name, **cfg)
    return run_dir / run_name / "weights" / "best.pt"


def train_all(
    models: Iterable[str],
    cfg: Mapping[str, Any],
    run_dir: Path,
    name_keys: Iterable[str],
) -> list[Path]:
    """Train all configured models sequentially."""
    best_weights: list[Path] = []
    for model in models:
        best_weights.append(train_model(model, cfg, run_dir, name_keys))
    return best_weights


if __name__ == "__main__":
    from config import MODEL_WEIGHTS, RUN_NAME_KEYS, RUNS_DIR, TRAIN_CONFIG

    train_all(MODEL_WEIGHTS, TRAIN_CONFIG, RUNS_DIR, RUN_NAME_KEYS)

