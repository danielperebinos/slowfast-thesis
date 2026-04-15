[← Experiments](experiments.md) · [Back to README](../README.md) · [Dashboard →](dashboard.md)

# MLflow Stack

How the tracking infrastructure is wired, what each service does, how to talk to it, and — critically — how **not** to wipe its data.

## Topology

```
┌─────────────────────────────────────────────────────────┐
│                    Training scripts                      │
│           (experiments/experiment_0N/train.py)          │
└────────────────────────────┬────────────────────────────┘
                             │ mlflow.log_* → HTTP :5000
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    MLflow server (:5000)                 │
│                    deployment/docker/                   │
│                    Dockerfile.mlflow                    │
└───────────┬────────────────────────────┬────────────────┘
            │ params / metrics            │ artifacts (models, files)
            ▼ SQL                         ▼ S3 API
┌────────────────────┐          ┌──────────────────────────┐
│   postgres:15      │          │      minio (:9000)       │
│  (MLflow backend)  │          │   bucket: ${MLFLOW_BUCKET}│
│                    │          │   console: :9001         │
│  volume:           │          │   volume:                │
│  ./volumes/postgres│          │   ./volumes/minio        │
└────────────────────┘          └──────────────────────────┘
```

All four services are defined in [`compose.experiments.yml`](../compose.experiments.yml).

## Services

| Service | Image | Purpose | Exposed ports | Volume (host path) |
|---------|-------|---------|---------------|---------------------|
| `postgres` | `postgres:15` | MLflow backend store — params, metrics, run metadata, tags. | `5432` | `./volumes/postgres` |
| `minio` | `minio/minio:RELEASE.2024-11-07T00-52-20Z` | S3-compatible artifact store — model checkpoints, logged files. | `9000` (API), `9001` (console) | `./volumes/minio` |
| `mc` | `minio/mc` | One-shot bucket initializer — creates `${MLFLOW_BUCKET}` on startup, then exits. | — | — |
| `mlflow` | built from `deployment/docker/Dockerfile.mlflow` (base `mlflow:v2.14.1` + `psycopg2-binary` + `boto3`) | MLflow tracking server. Talks to postgres + minio; training scripts talk to it. | `5000` | — |

## Environment Variables

All configured via `.env` at the project root (git-ignored).

| Var | Used by | Example | Notes |
|-----|---------|---------|-------|
| `POSTGRES_USER` | postgres, mlflow | `mlflow` | Postgres role for the MLflow backend DB. |
| `POSTGRES_PASSWORD` | postgres, mlflow | `mlflow_password` | — |
| `POSTGRES_DB` | postgres, mlflow | `mlflow` | DB name, created by the postgres image on first boot. |
| `MINIO_ROOT_USER` | minio, mc, mlflow | `mlflow` | MinIO access key; reused as `AWS_ACCESS_KEY_ID` inside the MLflow container. |
| `MINIO_ROOT_PASSWORD` | minio, mc, mlflow | `mlflow_password` | Reused as `AWS_SECRET_ACCESS_KEY`. |
| `MLFLOW_BUCKET` | mc, mlflow | `mlflow` | Bucket name in MinIO for artifact storage. |
| `MLFLOW_TRACKING_URI` | training scripts | `http://localhost:5000` | Read by `mlflow.set_tracking_uri(...)` in each `train.py`. |

The `mlflow` service boots with:

```
mlflow server
  --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
  --artifacts-destination s3://${MLFLOW_BUCKET}/
  --host 0.0.0.0 --port 5000
```

`MLFLOW_S3_ENDPOINT_URL=http://minio:9000` is set in the container env so the boto3 client inside MLflow targets MinIO instead of AWS.

## Lifecycle

### Start the stack

```bash
docker compose -f compose.experiments.yml up -d
```

Healthchecks: postgres uses `pg_isready`; minio pings `/minio/health/live`; `mc` waits for minio to be healthy, creates the bucket, and exits successfully. `mlflow` only starts after postgres is healthy, minio is healthy, and `mc` has completed — so a fresh boot can take ~15 seconds before `http://localhost:5000` answers.

### Check status

```bash
docker compose -f compose.experiments.yml ps
```

### View logs

```bash
docker compose -f compose.experiments.yml logs -f mlflow
docker compose -f compose.experiments.yml logs -f postgres
docker compose -f compose.experiments.yml logs -f minio
```

### Safe shutdown

```bash
docker compose -f compose.experiments.yml down
```

Stops containers. Volumes under `./volumes/` are preserved.

### Pause without stopping

```bash
docker compose -f compose.experiments.yml stop
```

## Volume Safety

Docker volumes in this project store **irreplaceable** training artifacts:

- `./volumes/postgres` — the MLflow backend DB. Losing it means losing every run's params/metrics/history.
- `./volumes/minio` — all logged MLflow artifacts, including `best_*.pth` model weights uploaded during training.

### Never run these commands

| Command | Why it's dangerous |
|---------|-------------------|
| `docker compose down -v` | `-v` deletes all named volumes — wipes both postgres and minio state. |
| `docker compose -f compose.experiments.yml down -v` | Same — `-v` affects volumes regardless of which compose file you invoke. |
| `docker volume rm ...` | Directly removes a named volume. |
| `docker system prune --volumes` / `-a --volumes` | Mass removal — will hit our volumes. |
| `rm -rf volumes/` | Deletes the host-mounted directory — same effect as volume removal. |
| Deleting `volumes.zip` | It is the archived snapshot of the above; keep it until you know you can reproduce everything from scratch. |

If a workflow (test reset, clean rebuild, CI wipe) appears to require volume removal: **stop and confirm with the user first**. There is almost always a less destructive path (rename the MLflow experiment, run `mc mb` for a fresh bucket, etc.).

### Backup before destructive maintenance

If you must touch the volumes — e.g., upgrade postgres across a major version — snapshot first:

```bash
# Postgres dump
docker compose -f compose.experiments.yml exec postgres \
    pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > mlflow_backup.sql

# MinIO bucket copy (via mc alias)
docker run --rm --network host -v $(pwd)/backup:/backup minio/mc \
    sh -c "mc alias set src http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD} && \
           mc mirror src/${MLFLOW_BUCKET} /backup/${MLFLOW_BUCKET}"
```

## Accessing MLflow Programmatically

From a training script:

```python
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("SlowFast_Anticipation")
with mlflow.start_run(run_name="Experiment_01_Baseline"):
    mlflow.log_params({...})
    mlflow.log_metrics({...}, step=epoch)
    mlflow.log_artifact("best_baseline.pth")
```

From the MinIO console (`http://localhost:9001`): browse artifacts under the `${MLFLOW_BUCKET}` bucket. Login with `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`.

## Common Operations

### Wipe a single run without touching volumes

Use the MLflow UI or `mlflow runs delete <run_id>` — this soft-deletes in postgres; artifacts remain in minio until garbage collection (`mlflow gc`). Volumes stay intact.

### Start fresh without losing history

Register a new MLflow experiment (e.g., `SlowFast_Anticipation_v2`) and set it in each `train.py`. All prior runs stay where they are, under the original experiment.

### Upgrade the MLflow server image

Bump the tag in `deployment/docker/Dockerfile.mlflow`, rebuild with:

```bash
docker compose -f compose.experiments.yml build mlflow
docker compose -f compose.experiments.yml up -d mlflow
```

Volumes are untouched; runs continue against the same postgres DB.

## See Also

- [Getting Started](getting-started.md) — full install + stack-up walkthrough.
- [Experiments](experiments.md) — what each variant logs to MLflow.
- [Architecture](architecture.md) — where MLflow calls belong in the code (hint: inside `main()`, never at import time).
