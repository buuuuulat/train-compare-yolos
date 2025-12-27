# train-compare-yolos

Small utility to train multiple YOLO models, optionally export to TensorRT, then validate/benchmark them and save metrics/plots.

## Setup
- Python 3.13+.
- Install deps (pick one):
  - `pip install -e .`
  - or `pip install -r requirements.txt`
- TensorRT export requires TensorRT installed and a compatible `torch`.

## Configuration (`src/config.py`)
- `MODEL_WEIGHTS`: models/checkpoints to train and (by default) benchmark.
- `TRAIN_CONFIG`: passed to `YOLO(...).train(...)`.
- `VAL_CONFIG`: passed to `YOLO(...).val(...)` (keep `batch=1` for per-frame latency).
- `RUN_NAME_KEYS`: which hyperparams appear in the run folder name.
- `RUNS_DIR`: training outputs (default `train_runs/`).
- `BENCH_DIR`: benchmark outputs (default `bench_runs/`).
- `EXPORT_CONFIG`: TensorRT export control (`enabled`, `format="engine"`, `half`, optional `device`).

## Running (from repo root)
- Train + bench:  
  `python src/main.py` (or `python src/main.py all`)
- Train only:  
  `python src/main.py train`
- Bench only using training runs:  
  `python src/main.py bench`
- Bench a directory of ready weights (no training):  
  `python src/main.py bench --weights-dir path/to/weights_dir`  
  Benchmarks all `.pt/.engine/.onnx` files there. If export is enabled, `.pt` files are also exported to `EXPORT_CONFIG["format"]` and evaluated.

## Outputs
- Training: `train_runs/<model>_<params>/weights/best.pt`.
- Benchmark: `bench_runs/metrics.json`, `bench_runs/metrics.csv`, plots in `bench_runs/plots/`:
  - `accuracy_map50.png` — mAP@0.5 bars.
  - `speed.png` — inference time (ms/image) bars.
  - `speed_vs_acc.png` — speed vs accuracy scatter.
  Labels use `model-variant`, where `variant` is `pt` or the export format (e.g., `engine`).

## Notes
- For latency-focused eval, keep `VAL_CONFIG["batch"]=1`.
- If TensorRT export fails or is unavailable, you'll see a warning and the bench runs on the original weights only.
