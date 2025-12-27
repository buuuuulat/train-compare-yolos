from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_WEIGHTS = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8s.pt",
    "yolo12n.pt",
]

RUN_NAME_KEYS = ["epochs", "imgsz", "batch", "lr0"]

TRAIN_CONFIG: dict[str, Any] = {
    "data": "./FSOCO-1/data.yaml",
    "epochs": 100,
    "imgsz": 800,
    "batch": 32,
    "workers": 8,
    "device": 0,
    "dropout": 0.1,
    "amp": True,
    "plots": True,
    "cache": True,
    "cos_lr": True,
    "patience": 20,
}

VAL_CONFIG: dict[str, Any] = {
    "data": TRAIN_CONFIG["data"],
    "batch": 1,
    "imgsz": TRAIN_CONFIG["imgsz"],
    "device": 0,
}

# TensorRT export options. Set enabled=False to skip export/bench of TRT engines.
EXPORT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "format": "engine",  # TensorRT
    "half": True,
    "device": 0,
    "imgsz": 800,
    "batch": 1,
}

RUNS_DIR = Path("train_runs")
BENCH_DIR = Path("bench_runs")
