from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    BENCH_DIR,
    EXPORT_CONFIG,
    MODEL_WEIGHTS,
    RUN_NAME_KEYS,
    RUNS_DIR,
    TRAIN_CONFIG,
    VAL_CONFIG,
)
from train import train_all
from valbench import benchmark_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO models and benchmark them.")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["all", "train", "bench"],
        default="all",
        help="What to run: train, bench (validation), or all (default).",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Optional directory with weights (.pt/.engine/.onnx) to benchmark instead of training runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode in ("all", "train"):
        train_all(MODEL_WEIGHTS, TRAIN_CONFIG, RUNS_DIR, RUN_NAME_KEYS)

    if args.mode in ("all", "bench"):
        benchmark_all(
            MODEL_WEIGHTS,
            TRAIN_CONFIG,
            VAL_CONFIG,
            RUNS_DIR,
            BENCH_DIR,
            RUN_NAME_KEYS,
            EXPORT_CONFIG,
            weights_dir=args.weights_dir,
        )


if __name__ == "__main__":
    main()
