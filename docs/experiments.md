[← Architecture](architecture.md) · [Back to README](../README.md) · [MLflow Stack →](mlflow-stack.md)

# Experiments

Phase 1 compares four SlowFast variants on the AVA action-anticipation task. Each variant is a self-contained Python module under `experiments/experiment_0N/`; they share the same data pipeline, loss, and metrics via `experiments/core/`.

## Variants Registry

| ID | Name | Architecture | Model file | Best-weights file | Technical objective |
|----|------|--------------|------------|-------------------|---------------------|
| 01 | Baseline | SlowFast R50 (`pytorchvideo.models.slowfast.create_slowfast`) | `experiments/experiment_01/train.py` (inline `get_model`) | `best_baseline.pth` | Reference for Top-1 / mAP / TTA / FPS. |
| 02 | Attention | SlowFast + Non-Local / Self-Attention | `experiments/experiment_02/model.py` | `best_attention.pth` | Evaluate global correlations via Self-Attention. |
| 03 | ROI Guidance | SlowFast + YOLOv8 ROI mask (Hadamard on Fast pathway) | `experiments/experiment_03/model.py` | `best_roi.pth` | Reduce redundancy by masking background on Fast pathway. |
| 04 | Hybrid | ROI + local attention (attention only on ROI tokens) | `experiments/experiment_04/model.py` | `best_hybrid.pth` | Pareto-optimize accuracy vs latency. |

All four write runs to the MLflow experiment `SlowFast_Anticipation` under run names `Experiment_0N_<Variant>`.

## Running an Experiment

Assuming the stack is up (`docker compose -f compose.experiments.yml up -d`) and videos are in `AVA/`:

```bash
python experiments/experiment_01/train.py      # Baseline
python experiments/experiment_02/train.py      # Attention
python experiments/experiment_03/train.py      # ROI
python experiments/experiment_04/train.py      # Hybrid
```

Each run is resumable — if `checkpoint.pth` exists in the experiment folder, model / optimizer / scheduler / epoch / best-val-acc are restored and training continues.

### Environment overrides

Every `train.py` accepts the same env-var overrides:

| Var | Purpose | Default |
|-----|---------|---------|
| `VIDEO_DIR` | Directory of AVA `.mp4` files. | `<experiment_dir>/data/videos` |
| `TRAIN_CSV` | Training split CSV. | `<experiment_dir>/train.csv` |
| `VAL_CSV` | Validation split CSV. | `<experiment_dir>/test.csv` |
| `CKPT_PATH` | Resume checkpoint path. | `<experiment_dir>/checkpoint.pth` |
| `MLFLOW_TRACKING_URI` | MLflow server. | `http://localhost:5000` |

Example — run experiment 02 on a shared video directory:

```bash
VIDEO_DIR=$(pwd)/AVA python experiments/experiment_02/train.py
```

## Shared Kernel

Every variant reuses the same primitives — never fork them.

| Module | Public API | Purpose |
|--------|------------|---------|
| `experiments/core/data_loader.py` | `build_label_map`, `make_ava_dataset`, `PackPathway`, `AvaAnticipationDataset`, `_slowfast_collate` | Dataset, transforms (`ShortSideScale`, `UniformTemporalSubsample`, `Normalize`, `PackPathway`), DataLoader factory, slow/fast pathway collate. |
| `experiments/core/loss.py` | `ActionAnticipationLoss(alpha, T_start, T_scale)` | Multi-label BCE + exponential temporal penalty `α · exp((t − T_start) / T_scale)`. |
| `experiments/core/metrics.py` | `multilabel_accuracy`, `topk_accuracy`, `AverageMeter`, `calculate_tta` | Per-label accuracy, Top-k accuracy, running averages, Time-to-Action. |

### Data contract

CSVs are AVA v2.2 format:

```
video_id, ts, x1, y1, x2, y2, action, person_id
```

One sample per `(video_id, ts)`; all persons' actions are pooled into a multi-hot label. The dataset subtracts `time_offset = 900.0` seconds to convert AVA absolute timestamps to local file time, then applies `anticipation_gap` to form the observation window.

### Label-map invariant

Every `train.py` builds the label map **once** from the training CSV before constructing either DataLoader:

```python
label_map = build_label_map(TRAIN_CSV)
train_loader, _ = make_ava_dataset(TRAIN_CSV, VIDEO_DIR, mode="train", label_map=label_map, ...)
val_loader,   _ = make_ava_dataset(VAL_CSV,   VIDEO_DIR, mode="val",   label_map=label_map, ...)
```

This is a hard invariant — class indices must be identical across splits for metrics to mean anything.

## Hyperparameters (current defaults)

| Variant | Epochs | Batch size | LR | Optimizer | Scheduler | Loss α | T_start | T_scale |
|---------|--------|------------|----|-----------|-----------|--------|---------|---------|
| 01 Baseline | 10 | 8 | 0.01 | SGD mom=0.9, wd=1e-4 | Cosine T_max=10 | 0.5 | 900.0 | 100.0 |

Variants 02–04 follow the same skeleton; see each `train.py` for variant-specific deltas.

## Adding a New Variant

1. **Create the folder:** `experiments/experiment_05/` with `__init__.py`, `train.py`, and (if it has a custom architecture) `model.py`.
2. **Start from the skeleton:** copy `experiment_01/train.py` and adapt `get_model`, run name, and best-weights filename.
3. **Use the kernel:** import `build_label_map`, `make_ava_dataset`, `ActionAnticipationLoss`, `AverageMeter`, `multilabel_accuracy` from `experiments.core`. Do NOT import from another variant.
4. **If a helper is only for this variant**, keep it inside the variant folder. Promote to `experiments/core/` once a second variant needs it.
5. **Register it** in this page's Variants Registry and in `experiments/experiments.md`.
6. **Generate splits** with `experiments/dataset/build_splits.py` (or copy the existing CSVs for a fair cross-variant comparison).
7. **Log to MLflow** under experiment `SlowFast_Anticipation`, run name `Experiment_05_<YourName>`.
8. **Update `verify_setup.py`** so the new variant is included in the smoke test.

## Protected Files

The following files are expensive to regenerate — do NOT delete or rewrite:

- `experiments/experiment_0N/checkpoint.pth` — training resume state.
- `experiments/experiment_0N/best_*.pth` — best-val weights.
- `experiments/experiment_0N/train.csv` / `test.csv` — dataset splits (seeded, reproducible, but re-downloading the AVA videos is slow).

See [MLflow Stack → Volume Safety](mlflow-stack.md#volume-safety) for the equivalent rule on the MLflow backend and artifact store.

## See Also

- [Architecture](architecture.md) — module boundaries and dependency rules.
- [MLflow Stack](mlflow-stack.md) — where params, metrics, and artifacts actually live.
- [Getting Started](getting-started.md) — end-to-end setup walkthrough.
