[← MLflow Stack](mlflow-stack.md) · [Back to README](../README.md)

# Dashboard

Self-contained Streamlit demo under `dashboard/` that turns the four trained SlowFast variants into a live inference tool for the thesis defense. Pick an AVA clip, pick a variant, watch predicted actions, forward latency, FPS, and Time-to-Action update in real time.

## Summary

| | |
|---|---|
| Location | `dashboard/` (sibling of `experiments/`) |
| Entrypoint | `dashboard/app.py` (Streamlit, port **8501**) |
| Scope | **Inference only.** Reads `experiments/**/best_*.pth` read-only; never writes to them. |
| Stream sources | AVA `.mp4` files under `../AVA/`. Webcam / RTSP / upload are out of scope this iteration. |
| Supported variants | `01_baseline`, `02_attention`, `03_roi`, `04_hybrid` (one checkbox each) |
| Shipping artifacts | Own `pyproject.toml`, `Dockerfile`, `compose.yml`, `Makefile`, `tests/`, `.dockerignore`, `.env.example` |
| Internal README | [`dashboard/README.md`](../dashboard/README.md) (task-level usage) |

## Why it exists

The training repo already produces four checkpoints with different accuracy / latency / TTA trade-offs. For the defense itself we need something the thesis committee can *see* — a UI that makes the Pareto story visible in a few clicks. Streamlit keeps the code small enough to audit in one sitting and avoids the JS/HTML ceremony that would distract from the actual research contribution.

## Quick start (Docker, recommended)

```bash
cd dashboard
make up                     # docker compose -f compose.yml up -d --build
open http://localhost:8501  # pick variant + AVA clip, press Start
make logs                   # tail the container's stdout
make down                   # stop (no -v, never any -v)
```

Override the host port:

```bash
DASHBOARD_PORT=8600 make up
```

## Quick start (local host GPU)

```bash
cd dashboard
make install        # uv sync reading dashboard/pyproject.toml
make run            # streamlit run app.py on port 8501
```

`uv` fetches `torch 2.0.1 + cu118` wheels from the same index the training repo uses — there is no risk of pulling a different CUDA build.

## How it works

```
AVA .mp4
    │
    ▼
cv2.VideoCapture ─► rolling frame buffer (≥ NUM_FRAMES = 32 frames)
    │
    ▼
ClipPreprocessor ─► UniformTemporalSubsample(32) ─► /255 ─► Normalize(0.45, 0.225)
                    ShortSideScale(256) ─► CenterCrop(224) ─► PackPathway
    │                                                             │
    ▼                                                             ▼
                                [slow (1,3,4,224,224), fast (1,3,32,224,224)]
                                               │
                                               ▼
                InferenceEngine ─► variant model (frozen, eval, RO checkpoint)
                    │                   ├── torch.cuda.synchronize() bracketing
                    │                   ├── sigmoid (multi-label, BCE-trained)
                    │                   └── top-k
                    ▼
                InferenceResult ─► LatencyTracker (p50/p95/p99) + TTAComputer (AVA lookup)
                    │
                    ▼
                Streamlit panel: frame preview + metrics + top-k bar chart + raw probs
```

Preprocessing constants live in `dashboard/config.py` and **must match** `experiments/core/data_loader.py::get_val_transform` line-for-line. A parity smoke test (`test_label_map_matches_core_implementation`) guards against drift between the dashboard's standalone `load_label_map` and the canonical `experiments.core.data_loader.build_label_map`.

## Metrics shown

| Metric | Source | Unit | Notes |
|---|---|---|---|
| **Forward** | `torch.cuda.synchronize` bracketed `perf_counter` | ms | Pure model forward pass. CPU wall-clock would lie for async CUDA — we sync both sides. |
| **Preprocess** | `perf_counter` around `ClipPreprocessor.prepare()` | ms | Exposed on `InferenceResult` for debugging; not in the main metrics row. |
| **FPS (p50)** | `1000 / p50(forward_ms)` | Hz | Median over last `LATENCY_HISTORY` (120) samples. |
| **TTA** | `(clip_end_sec + anticipation_gap) − gt_ts` | s | Shown only when a top-k prediction matches an AVA annotation for the current video. "—" otherwise. Matches `experiments/core/metrics.py::calculate_tta`. |

Sign convention for TTA follows training: **positive = late**, **negative = early** (desired).

## Variant registry

Defined in `dashboard/config.py` as a frozen `dict[str, VariantSpec]`. Adding a new variant requires three things:

1. A new `experiment_05/model.py` with a constructor compatible with `num_classes=<int>`.
2. A trained `best_<name>.pth` file (state_dict only, no optimizer state) next to it.
3. A new `VariantSpec` entry in `VARIANTS` **plus** a matching branch in `inference/model_loader.py::_BUILDERS`.

The checkpoint is always loaded with `torch.load(path, map_location=device)` followed by `load_state_dict(state, strict=True)`; a `strict=False` fallback is kept only to survive future key-renames, logged as `WARN`.

## Protected assets

This subproject strictly honors the project's data-safety rules. The `compose.yml` bind-mounts `../experiments` and `../AVA` as `:ro`, so even a rogue file write inside the container would fail at the kernel level. There is no `down -v` target in the `Makefile`, and `compose.yml` declares no named volumes.

- `../volumes/postgres`, `../volumes/minio` — never referenced. Dashboard stack is a different compose project (`slowfast-dashboard`).
- `../experiments/**/best_*.pth`, `../experiments/**/checkpoint.pth` — opened with `torch.load(..., map_location=device)` only. No `torch.save`, no file writes.
- `../AVA/*.mp4` — read through `cv2.VideoCapture`.

See [MLflow Stack → Volume Safety](mlflow-stack.md#volume-safety) for the matching protections on the training side, and `.ai-factory/RULES.md` for the top-level axioms.

## Configuration

All knobs are environment variables. Defaults in parentheses.

| Variable | Default | Purpose |
|---|---|---|
| `DASHBOARD_PORT` | `8501` | Host port for the Streamlit server. Keep clear of the MLflow stack (5000, 5432, 9000, 9001). |
| `DASHBOARD_LOG_LEVEL` | `DEBUG` | Logger level. DEBUG is verbose but intended — silence `ultralytics`/`matplotlib`/`PIL` internally. |
| `PROJECT_ROOT` | `/workspace` in container, autodetected on host | Used to resolve `MODEL_ROOT` / `VIDEO_ROOT` when they are not set explicitly. |
| `MODEL_ROOT` | `$PROJECT_ROOT/experiments` | Where the four `experiment_0N/` trees live (read-only). |
| `VIDEO_ROOT` | `$PROJECT_ROOT/AVA` | Where the AVA `.mp4` clips live (read-only). |
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` | Informational only — the dashboard never writes to MLflow. |

An `.env.example` is provided for convenience; copy to `.env` if you want `docker compose` to pick values up automatically.

## Testing

Smoke tests live under `dashboard/tests/` and are safe to run anywhere (CPU-only, network-free).

```bash
cd dashboard
make smoke                                     # all tests, incl. slow per-variant forward
PYTEST_ARGS='-m "not slow"' make smoke         # fast subset (preprocess + label-map + latency)
```

What's covered:

- Every `VariantSpec.checkpoint` file exists (skips cleanly if `experiments/` is not mounted).
- `ClipPreprocessor` produces the exact `(1,3,4,224,224)` / `(1,3,32,224,224)` pathway pair from arbitrary-length RGB input.
- Empty-input rejection.
- Label-map parity vs `experiments.core.data_loader.build_label_map`.
- One CPU forward pass per variant, asserting finite probs and non-zero `forward_ms` (`slow`-marked).
- `LatencyTracker.summary()` returns monotonic p50 ≤ p95 ≤ p99 ≤ max over a 100-sample dataset.

## Limitations

- **GPU strongly recommended.** CPU SlowFast forwards are ~1–3 s per clip, which breaks the live-demo illusion. The numbers remain honest either way.
- **TTA requires matching AVA annotations.** If the selected clip has no rows in `experiments/experiment_01/test.csv`, the TTA column stays `—`.
- **No streaming ingest.** Webcam, RTSP, and file upload are out of scope for the defense iteration — their failure modes (driver conflicts, network flakiness) are too risky on presentation day. The architecture leaves room to add them later; see `dashboard/app.py` playback loop for the extension point.
- **No MLflow writes.** The tracking URI is passed through but unused; the dashboard is a one-way consumer of training artifacts.

## File layout

```
dashboard/
├── app.py                 # Streamlit entry
├── config.py              # VARIANTS registry + preprocessing constants
├── logging_setup.py       # verbose logger factory
├── inference/
│   ├── preprocess.py      # ClipPreprocessor (mirror of get_val_transform)
│   ├── model_loader.py    # load_variant / load_label_map (read-only checkpoint)
│   ├── engine.py          # InferenceEngine + InferenceResult (CUDA-sync timing)
│   └── metrics.py         # LatencyTracker + TTAComputer + format helpers
├── ui/
│   ├── state.py           # st.session_state keys + init/reset helpers
│   └── components.py      # variant/stream pickers, results panel, frame preview
├── tests/test_smoke.py    # CPU-only smoke tests (slow-marked per-variant forward)
├── pyproject.toml         # streamlit, torch-cu118, pytorchvideo, ultralytics, opencv-headless
├── Dockerfile             # CUDA 11.8 + Python 3.10 + Streamlit on 8501
├── compose.yml            # slowfast-dashboard project, RO bind mounts, GPU reservation
├── Makefile               # install / run / smoke / up / down / logs / status / ...
├── .dockerignore
└── .env.example
```

## See Also

- [MLflow Stack](mlflow-stack.md) — Postgres + MinIO + MLflow server the dashboard's checkpoints originated from.
- [Experiments](experiments.md) — Variant registry and training-time hyperparameters the dashboard must mirror at inference.
- [Architecture](architecture.md) — Module boundaries; explains why the dashboard imports `experiments/experiment_0{2,3,4}/model.py` but never `train.py`.
