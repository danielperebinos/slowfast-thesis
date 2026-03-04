# SlowFast Action Anticipation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorchVideo](https://img.shields.io/badge/PyTorchVideo-0.1.5-EE4C2C)](https://pytorchvideo.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=ultralytics&logoColor=black)](https://ultralytics.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![uv](https://img.shields.io/badge/uv-package%20manager-5C4EFA)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Research project exploring SlowFast R50 architectural variants for **human action recognition
> and anticipation** on the AVA dataset, with a Gradio dashboard for live video inference.

---

## Overview

This project investigates the Pareto trade-off between **accuracy** (mAP / Top-1) and
**anticipation speed** (Time-to-Action / latency) through four SlowFast variants trained
with an action anticipation loss. A Phase 2 interactive dashboard lets you run any trained
model on a video file and see person detections, action predictions, and latency metrics
in real time.

### Key Contributions

- **4 SlowFast variants** ranging from a clean baseline to a hybrid ROI + attention model
- **Action anticipation loss** — exponential penalty for late decisions
- **Full MLflow tracking** — every run stores params, metrics, curves, and checkpoints
- **Gradio dashboard** — hot-swap models, simulate from video, observe metrics live

---

## Architecture Variants

| Variant | Model | Core Idea |
| :--- | :--- | :--- |
| **Baseline** | SlowFast R50 | Standard two-pathway network; CE loss only |
| **Attention** | SlowFast R50 + Non-Local | Self-attention injected into Fast pathway `res_4` / `res_5` |
| **ROI Guidance** | SlowFast R50 + YOLOv8 | Binary person-ROI mask applied (Hadamard) to Fast pathway features |
| **Hybrid** | ROI + Local Attention | Non-Local attention restricted to ROI tokens — accuracy of Attention at speed of ROI |

### Anticipation Loss

$$L_{\text{total}} = L_{\text{cls}} + \alpha \cdot \exp\!\left(\frac{t - T_{\text{start}}}{T}\right)$$

Where $t$ is the clip end time (AVA absolute seconds), $T_{\text{start}}$ is the boundary
after which the penalty grows, and $\alpha$ weights its contribution.

---

## Project Structure

```
slowfast/
├── experiments/
│   ├── core/
│   │   ├── data_loader.py      # AvaAnticipationDataset, PackPathway, transforms
│   │   ├── loss.py             # ActionAnticipationLoss
│   │   └── metrics.py          # mAP, Top-k, TTA
│   ├── dataset/
│   │   ├── fetch.py            # Video downloader (yt-dlp)
│   │   ├── download_all.py     # Batch downloader
│   │   └── build_splits.py     # Train/val CSV splits
│   ├── experiment_01/train.py  # Baseline
│   ├── experiment_02/
│   │   ├── model.py            # AttentionSlowFast
│   │   └── train.py
│   ├── experiment_03/
│   │   ├── model.py            # RoiGuidanceSlowFast
│   │   └── train.py
│   ├── experiment_04/
│   │   ├── model.py            # HybridSlowFast
│   │   └── train.py
│   ├── yolov8n.pt              # YOLOv8n weights
│   └── experiments.md          # Experiment registry
├── dashboard/
│   ├── app.py                  # Gradio UI entry point
│   ├── inference/
│   │   ├── buffer.py           # FrameBuffer (deque, maxlen=32)
│   │   ├── model_loader.py     # load_model(): checkpoint → eval model
│   │   └── pipeline.py        # InferencePipeline (YOLO + SlowFast)
│   ├── labels/
│   │   ├── ava_actions.txt     # 80 AVA action labels
│   │   └── kinetics400.txt     # 400 Kinetics labels (pretrained fallback)
│   ├── videos/                 # Drop .mp4 files here for simulation
│   └── plans.md                # Phase 2 implementation plan
├── deployment/
│   └── docker/
│       └── Dockerfile.mlflow
├── volumes/                    # Postgres + MinIO data (git-ignored)
├── compose.experiments.yml     # MLflow infrastructure (Postgres + MinIO)
├── pyproject.toml
└── uv.lock
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — `pip install uv`
- NVIDIA GPU with CUDA 11.8 (CPU fallback works but is slow)
- Docker + Docker Compose (for MLflow infrastructure)

### Install

```bash
git clone <repo-url>
cd slowfast
uv sync
```

### Phase 1 — Train

**1. Start MLflow infrastructure:**

```bash
# Copy and fill in the required env vars
cp .env.example .env          # set POSTGRES_USER, POSTGRES_PASSWORD, etc.
docker compose -f compose.experiments.yml up -d
# MLflow UI → http://localhost:5000
# MinIO UI  → http://localhost:9001
```

**2. Download AVA clips:**

```bash
uv run python experiments/dataset/build_splits.py
uv run python experiments/dataset/download_all.py
```

**3. Run experiments:**

```bash
uv run python experiments/experiment_01/train.py   # Baseline
uv run python experiments/experiment_02/train.py   # Attention
uv run python experiments/experiment_03/train.py   # ROI Guidance
uv run python experiments/experiment_04/train.py   # Hybrid
```

All metrics, parameters, and checkpoints are logged to MLflow automatically.

### Phase 2 — Dashboard

```bash
uv run python dashboard/app.py
# → http://localhost:7860
```

1. Drop a `.mp4` file into `dashboard/videos/`.
2. Select a **Model Variant** and click **Load Model**.
   - If a checkpoint exists in `volumes/`, it is loaded automatically.
   - Otherwise the app falls back to pretrained Kinetics-400 weights.
3. Pick the video from the dropdown (or upload directly).
4. Click **Simulate** — annotated frames stream to the output panel.
5. Use **Stop** to halt, then switch variants and repeat.

---

## Configuration

Each training script has a configuration block at the top (no external config file needed):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 4 | Samples per batch |
| `LR` | 1e-4 | Initial learning rate (AdamW) |
| `ALPHA` | 0.1 | Anticipation loss weight |
| `CLIP_DURATION` | 2.0 s | Observation window length |
| `ANTICIPATION_GAP` | 1.0 s | Gap between clip end and action timestamp |
| `NUM_FRAMES` | 32 | Fast pathway frames |
| `SLOWFAST_ALPHA` | 8 | Slow/Fast frame ratio |

Dashboard controls (adjustable at runtime):

| Control | Range | Default | Description |
| :--- | :--- | :--- | :--- |
| Confidence Threshold | 0.1 – 0.9 | 0.5 | YOLO detection confidence |
| Inference Every N Frames | 1 – 32 | 8 | SlowFast inference cadence |

---

## MLflow Tracking

Every training run logs:

- **Params:** model variant, dataset split, seed, LR, batch size, alpha, epochs, crop size
- **Metrics (per epoch):** `train/loss`, `val/loss`, `val/top1`, `val/map`
- **Artefacts:** `checkpoint.pth`, loss curve PNG
- **Tags:** `num_classes`, `label_map`

Access the UI at `http://localhost:5000` after starting the Docker stack.

---

## Inference Pipeline

```
Video File
    │
    ▼
FrameBuffer (deque, maxlen=32, BGR frames)
    │
    ├── Every frame ──► YOLOv8 person detection → boxes
    │
    └── Every K frames ──► PackPathway
                               ├── Slow: 4 uniformly sampled frames  (1, 3,  4, 224, 224)
                               └── Fast: all 32 frames               (1, 3, 32, 224, 224)
                                       │
                                       ▼
                                 SlowFast forward
                                       │
                                  softmax → Top-3 actions
                                       │
                               annotate frame + overlay metrics
```

Preprocessing matches training: short-side scale 256 → center crop 224 ×  224,
normalised with mean `[0.45, 0.45, 0.45]` and std `[0.225, 0.225, 0.225]`.

---

## Dependencies

| Package | Version | Role |
| :--- | :--- | :--- |
| `torch` | 2.0.1+cu118 | Deep learning framework |
| `torchvision` | 0.15.2+cu118 | Video transforms |
| `pytorchvideo` | 0.1.5 | SlowFast model, `PackPathway`, transforms |
| `ultralytics` | ≥ 8.4 | YOLOv8 person detector |
| `mlflow` | ≥ 3.9 | Experiment tracking |
| `gradio` | ≥ 4.0 (6.8) | Dashboard UI |
| `opencv-python` | ≥ 4.8 (4.11) | Video I/O, frame annotation |
| `pandas` | ≥ 2.3 | CSV / label-map handling |
| `numpy` | < 2.0 | Tensor utilities |
| `yt-dlp` | ≥ 2026.2 | AVA video downloading |

---

## Roadmap

- [x] Phase 1 — Baseline SlowFast R50 training on AVA
- [x] Phase 1 — Attention variant (Non-Local blocks)
- [x] Phase 1 — ROI Guidance variant (YOLOv8 mask)
- [x] Phase 1 — Hybrid variant (ROI + Local Attention)
- [x] Phase 2 — Gradio dashboard with video file simulation
- [ ] Phase 2 — Comparative metrics table in dashboard (mAP, TTA, FPS across variants)
- [ ] Phase 2 — Grad-CAM / activation map overlay
- [ ] Phase 2 — RTSP / webcam live stream support
- [ ] Phase 2 — Export annotated video to file

---

## License

MIT — see [LICENSE](LICENSE).
