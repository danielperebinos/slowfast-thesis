# Experiments Registry — SlowFast Action Anticipation

## Phase 1: Training

| ID | Name | Architecture | Technical Objective |
| :--- | :--- | :--- | :--- |
| **01** | Baseline | SlowFast R50 | Reference mAP, TTA and FPS. Standard CE loss. |
| **02** | Attention | SlowFast R50 + Non-Local blocks | Global self-attention on Fast pathway (`res_4`, `res_5`). Hypothesis: higher mAP on complex actions at cost of latency. |
| **03** | ROI Guidance | SlowFast R50 + YOLOv8 mask | Binary ROI mask via YOLOv8 applied (Hadamard) to Fast pathway features. Hypothesis: earlier TTA by suppressing background. |
| **04** | Hybrid | ROI + Local Attention | Non-Local attention applied on ROI-masked features only. Pareto target: accuracy near exp02, speed near exp03. |

### Core Modules (`core/`)

| Module | Description |
| :--- | :--- |
| `loss.py` | `ActionAnticipationLoss`: $L_{total} = L_{cls} + \alpha \cdot \exp\!\left(\frac{t - T_{start}}{T}\right)$ — penalises late decisions. |
| `metrics.py` | mAP, Top-k accuracy, TTA (Time-To-Action). |
| `data_loader.py` | `AvaAnticipationDataset`, `PackPathway`, `get_train_transform`, `get_val_transform`. |

### How to Run

```bash
# Start MLflow infrastructure first
docker compose -f compose.experiments.yml up -d

# Then run each experiment (from project root)
uv run python experiments/experiment_01/train.py   # Baseline
uv run python experiments/experiment_02/train.py   # Attention
uv run python experiments/experiment_03/train.py   # ROI Guidance
uv run python experiments/experiment_04/train.py   # Hybrid
```

All runs are logged to **MLflow** (params, metrics, artefacts, checkpoints).
Checkpoints are saved to `volumes/<experiment>/checkpoint.pth`.

---

## Phase 2: Live Inference Dashboard

Built on top of Phase 1 trained models. Runs any of the 4 variants on a video file and
displays person detections, predicted actions, and performance metrics in real time.

| Component | Location | Description |
| :--- | :--- | :--- |
| Gradio app | `dashboard/app.py` | UI entry point — video upload + Simulate loop |
| Pipeline | `dashboard/inference/pipeline.py` | YOLO detection + SlowFast inference per frame |
| Frame buffer | `dashboard/inference/buffer.py` | `FrameBuffer(maxlen=32)` → slow/fast pathway tensors |
| Model loader | `dashboard/inference/model_loader.py` | Loads checkpoint or falls back to pretrained hub weights |
| Action labels | `dashboard/labels/` | `ava_actions.txt` (80) / `kinetics400.txt` (400) |
| Test videos | `dashboard/videos/` | Drop `.mp4` files here |

### How to Run

```bash
uv run python dashboard/app.py
# → http://localhost:7860
```

1. Select a model variant and click **Load Model**.
2. Upload a video or pick one from `dashboard/videos/`.
3. Click **Simulate** — annotated frames stream into the output panel.
4. Adjust **Confidence Threshold** and **Inference Every N Frames** live.
5. Click **Stop** to halt the simulation.

See `dashboard/plans.md` for the full Phase 2 implementation plan and design decisions.
