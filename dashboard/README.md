# SlowFast Dashboard

Inference-only Streamlit demo for the four trained SlowFast variants from this thesis. Pick a video, pick a model, watch live predictions with honest latency and Time-to-Action numbers. Intended for the thesis defense presentation.

## What it does

- Loads one of the four trained checkpoints (`best_baseline.pth`, `best_attention.pth`, `best_roi.pth`, `best_hybrid.pth`) on demand.
- Plays an AVA `.mp4` from `../AVA/` and runs inference every `INFERENCE_STRIDE_FRAMES` raw frames on a rolling `NUM_FRAMES`-frame buffer.
- Shows the top-k predicted actions, the raw probabilities, forward-pass latency (p50 / p95 / p99, CUDA-sync'd), FPS, and TTA against AVA ground truth when it's available for the selected clip.

## Prerequisites

- Docker + NVIDIA Container Toolkit **(recommended for defense day)**, or a host Python 3.10 with CUDA 11.8.
- GPU strongly recommended. CPU forward passes for SlowFast are ~1–3 s per clip — the app still works on CPU, it just won't feel live.
- The four trained checkpoints must already exist under `../experiments/experiment_0{1..4}/best_*.pth`. The dashboard never retrains anything.

## Quick start (Docker)

```bash
cd dashboard
make up                       # docker compose -f compose.yml up -d --build
open http://localhost:8501    # or just navigate your browser there
make logs                     # tail the container logs
make down                     # stop (no volumes to clean, by design)
```

Port override:

```bash
DASHBOARD_PORT=8600 make up
```

## Quick start (local host GPU)

```bash
cd dashboard
make install                  # uv sync — installs streamlit + torch-cu118 + ...
make run                      # streamlit run app.py on port 8501
```

## Running the smoke tests

```bash
cd dashboard
make smoke                                   # all tests, including the slow per-variant forward
PYTEST_ARGS='-m "not slow"' make smoke       # fast subset (preprocess/latency/label-map)
```

## Protected assets

This subproject **never** writes to shared training state:

- `../volumes/postgres` and `../volumes/minio` — the MLflow backend DB and artifact store. Docker-wise we never touch them; the dashboard's compose project name is `slowfast-dashboard`, fully namespaced away from `compose.experiments.yml`.
- `../experiments/**/best_*.pth` and `../experiments/**/checkpoint.pth` — bind-mounted into the container **read-only** (see `compose.yml`).
- `../AVA/*.mp4` — same: read-only bind mount.
- The `Makefile` has no `clean-volumes` / `clean-checkpoints` / `down -v` target, and `compose.yml` has no named `volumes:` section pointing at `../volumes/`.

If a future change adds any of those, treat it as a bug. See `.ai-factory/RULES.md`.

## Architecture

```
AVA .mp4
    │
    ▼
cv2.VideoCapture ─► rolling frame buffer (≥ NUM_FRAMES frames)
    │
    ▼
ClipPreprocessor ─► UniformTemporalSubsample(32) ─► /255 ─► Normalize(0.45, 0.225)
                    ShortSideScale(256) ─► CenterCrop(224) ─► PackPathway
    │                                                             │
    ▼                                                             ▼
                                [slow (1,3,4,224,224), fast (1,3,32,224,224)]
                                               │
                                               ▼
                InferenceEngine ─► variant model (frozen, eval, read-only checkpoint)
                    │                   ├── CUDA-sync bracketed forward timing
                    │                   └── sigmoid → top-k
                    ▼
                InferenceResult ─► LatencyTracker (p50/p95/p99) + TTAComputer (vs AVA CSV)
                    │
                    ▼
                Streamlit panel: frame preview + metrics + bar chart + raw probs
```

Preprocessing constants (`NUM_FRAMES=32`, `CROP=224`, `SHORT_SIDE=256`, `MEAN=0.45`, `STD=0.225`, `SLOWFAST_ALPHA=8`, `SLOW_FRAMES=4`) are defined in `config.py` and **must** match `experiments/core/data_loader.py::get_val_transform` line-for-line. A parity smoke test (`test_label_map_matches_core_implementation`) guards against label-map drift.

## Metrics shown

| Metric | Source | Unit | Notes |
|---|---|---|---|
| Forward | `torch.cuda.synchronize` bracketed perf_counter | ms | Pure model forward pass |
| FPS (p50) | `1000 / p50(forward_ms)` | Hz | Over the last 120 samples |
| TTA | `(clip_end_sec + anticipation_gap) − gt_ts` | s | Shown only when an AVA annotation matches a top-k prediction; "—" otherwise. Matches `experiments/core/metrics.py::calculate_tta` semantics. |

## Limitations

- Only `../AVA/*.mp4` streams are supported this iteration. Webcam, RTSP, and file upload are out of scope for the defense demo — their failure modes (driver conflicts, network flakiness) are too dangerous on presentation day.
- TTA requires matching rows in `../experiments/experiment_01/test.csv`. Clips outside AVA's annotation set display "—".
- The dashboard does **not** write to MLflow. The MLflow tracking URI is wired in for forward compatibility but unused.
- Streamlit reruns on every widget interaction — heavy state (loaded model, latency ring buffer) is cached in `st.session_state` and rebuilt only on variant change.

## File layout

```
dashboard/
├── app.py                 # Streamlit entry
├── config.py              # paths, variants, preprocessing constants
├── logging_setup.py       # verbose logger factory
├── inference/
│   ├── preprocess.py      # ClipPreprocessor (mirror of get_val_transform)
│   ├── model_loader.py    # load_variant / load_label_map (read-only checkpoint)
│   ├── engine.py          # InferenceEngine + InferenceResult
│   └── metrics.py         # LatencyTracker + TTAComputer
├── ui/
│   ├── state.py           # st.session_state keys
│   └── components.py      # variant/stream pickers, results panel
├── tests/test_smoke.py    # preprocess shapes, label-map parity, per-variant forward (slow)
├── pyproject.toml         # streamlit, torch-cu118, pytorchvideo, ultralytics, opencv-headless
├── Dockerfile             # CUDA 11.8 + Python 3.10 + Streamlit on 8501
├── compose.yml            # slowfast-dashboard project, read-only bind mounts, GPU
├── Makefile               # install / run / smoke / up / down / logs / ...
├── .dockerignore
└── .env.example
```
