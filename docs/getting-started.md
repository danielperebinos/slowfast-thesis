[Back to README](../README.md) · [Architecture →](architecture.md)

# Getting Started

How to set up the repo, bring up the MLflow stack, prepare data, and launch your first experiment.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | **3.10** | pinned in `.python-version`; `uv` reads this file. |
| `uv` | latest | package manager; see [install docs](https://docs.astral.sh/uv/getting-started/installation/). |
| Docker + Docker Compose | 24+ | for the MLflow stack (postgres + minio + server). |
| NVIDIA driver + CUDA 11.8 | — | PyTorch wheels pull from the `pytorch-cu118` index. CPU-only works for smoke tests. |
| `ffmpeg` + `yt-dlp` | latest | required by `experiments/dataset/download_all.py` if you need to (re-)download AVA clips. |

## 1. Clone & Install

```bash
git clone https://github.com/danielperebinos/slowfast-thesis.git
cd slowfast-thesis
uv sync
```

`uv sync` resolves `pyproject.toml` + `uv.lock` into `.venv/`. PyTorch comes from the CUDA 11.8 index declared in `pyproject.toml`.

Activate the venv if you need to run ad-hoc commands:

```bash
source .venv/bin/activate
```

## 2. Configure Environment

Create or verify `.env` at the repo root (the file is git-ignored). Minimal values:

```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password
POSTGRES_DB=mlflow
MINIO_ROOT_USER=mlflow
MINIO_ROOT_PASSWORD=mlflow_password
MLFLOW_BUCKET=mlflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

See [MLflow Stack](mlflow-stack.md) for the full list and what each service does.

## 3. Bring Up the MLflow Stack

```bash
docker compose -f compose.experiments.yml up -d
```

This starts:

- **postgres** on port `5432` — MLflow backend store.
- **minio** on ports `9000` (API) and `9001` (console) — artifact store.
- **mc** — one-shot bucket creator (exits after creating `${MLFLOW_BUCKET}`).
- **mlflow** on port `5000` — tracking server.

Verify the UI is up: open `http://localhost:5000`.

> ⚠️ **Never stop the stack with `docker compose down -v`**. The `-v` flag deletes the named volumes under `./volumes/`, which hold the tracking DB and all trained-model artifacts. Use `docker compose -f compose.experiments.yml down` (no `-v`) or just `stop`. See [MLflow Stack → Volume Safety](mlflow-stack.md#volume-safety).

## 4. Prepare the Dataset

### 4.1 — Download AVA clips

The raw AVA annotation CSV is expected at `experiments/dataset/data/ava/ava_train_v2.2.csv`.

```bash
export AVA_VIDEO_DIR=$(pwd)/AVA
python experiments/dataset/download_all.py \
    --csv experiments/dataset/data/ava/ava_train_v2.2.csv \
    --workers 4
```

The script downloads seconds `900–1800` of each YouTube video with `yt-dlp` + `ffmpeg`, re-encodes at 30 FPS / CRF 23, and writes `${AVA_VIDEO_DIR}/{video_id}.mp4`.

### 4.2 — Build experiment splits

Generates `train.csv` / `test.csv` for each experiment (video-id-level split, no row leakage):

```bash
AVA_VIDEO_DIR=$(pwd)/AVA OUT_DIR=$(pwd)/experiments/experiment_01 \
    python experiments/dataset/build_splits.py
```

Repeat for `experiment_02`, `experiment_03`, `experiment_04` if you want independent splits (or copy the same CSVs across variants for a strict fairness comparison).

## 5. Run Your First Experiment

```bash
python experiments/experiment_01/train.py
```

What happens:

1. Loads `train.csv` / `test.csv` and builds the shared label map once.
2. Creates the SlowFast R50 model.
3. Restores `checkpoint.pth` if one is present (training is resumable).
4. Runs 10 epochs, logs params + per-epoch metrics to MLflow under experiment `SlowFast_Anticipation`, run name `Experiment_01_Baseline`.
5. Writes `best_baseline.pth` whenever val-accuracy improves and uploads it as an MLflow artifact.

Open `http://localhost:5000` and select the `SlowFast_Anticipation` experiment to watch the run live.

## 6. Verify Your Environment (optional)

A dummy-data forward/backward pass for all four variants:

```bash
cd experiments
python verify_setup.py
```

This exercises `baseline`, `attention`, `roi_guidance`, and `hybrid` with random tensors. Useful to confirm CUDA + PyTorchVideo + local modules import correctly before launching a long training run.

## Next Steps

- Learn the code layout and module rules → [Architecture](architecture.md).
- Add a new variant or reconfigure an existing one → [Experiments](experiments.md).
- Tune the tracking stack → [MLflow Stack](mlflow-stack.md).

## See Also

- [Architecture](architecture.md) — module boundaries and dependency rules.
- [Experiments](experiments.md) — how to configure and compare the four variants.
- [MLflow Stack](mlflow-stack.md) — postgres + minio + mlflow server, env vars, volume safety.
