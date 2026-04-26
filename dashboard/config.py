"""Single source of truth for dashboard paths, variant registry, and
preprocessing constants.

**Do not put any I/O here.** This module must stay import-safe so tests and
unit checks can pick it up without touching the filesystem or loading a
model. All path defaults are derived from env vars; override via:

- ``PROJECT_ROOT``  — absolute path to the slowfast-thesis checkout
- ``MODEL_ROOT``    — where the four ``experiment_0N/`` folders live
- ``VIDEO_ROOT``    — where AVA ``*.mp4`` clips live

Preprocessing constants mirror ``experiments/core/data_loader.py``
``get_val_transform`` exactly. Drift here invalidates the trained weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from logging_setup import get_logger

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

# dashboard/ sits at <PROJECT_ROOT>/dashboard/; the default project root is
# the parent of this file's directory.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT") or _DEFAULT_PROJECT_ROOT)
EXPERIMENTS_DIR: Path = Path(os.getenv("MODEL_ROOT") or (PROJECT_ROOT / "experiments"))
VIDEO_DIR: Path = Path(os.getenv("VIDEO_ROOT") or (PROJECT_ROOT / "AVA"))

# YOLO weights used by variants 03 and 04 live inside the experiments tree.
YOLO_WEIGHTS: Path = EXPERIMENTS_DIR / "yolov8n.pt"

# CSV used to rebuild the same label map the four experiments trained against.
# All four experiments share the same CSVs after `make data-splits-all`, so
# experiment_01's copy is authoritative.
LABEL_MAP_CSV: Path = EXPERIMENTS_DIR / "experiment_01" / "train.csv"

# AVA ground-truth annotation CSVs used for live TTA lookup.
# Both train and test splits are loaded so TTA works for all local videos.
TTA_ANNOTATIONS_CSVS: tuple[Path, ...] = (
    EXPERIMENTS_DIR / "experiment_01" / "train.csv",
    EXPERIMENTS_DIR / "experiment_01" / "test.csv",
)

# ── Preprocessing constants — must match experiments/core/data_loader.py ─────

NUM_FRAMES: int = 32
SHORT_SIDE: int = 256
CROP_SIZE: int = 224
MEAN: tuple[float, float, float] = (0.45, 0.45, 0.45)
STD: tuple[float, float, float] = (0.225, 0.225, 0.225)
SLOWFAST_ALPHA: int = 8
SLOW_FRAMES: int = NUM_FRAMES // SLOWFAST_ALPHA  # 4
CLIP_DURATION_SEC: float = 2.0
ANTICIPATION_GAP_SEC: float = 1.0  # same default as AvaAnticipationDataset
TIME_OFFSET_SEC: float = 900.0  # AVA file-local time offset

# ── Dashboard runtime knobs ──────────────────────────────────────────────────

# Target playback FPS. The playback loop fast-skips intervening video frames
# (via cv2.VideoCapture.grab — no decode) to match this cadence. Paired with
# JPEG-encoded frames (see JPEG_QUALITY / JPEG_MAX_WIDTH below), the Streamlit
# wire payload drops ~10–20× vs raw ndarray and 20 FPS is sustainable on a
# modest host. CPU-only hosts should dial this back via DASHBOARD_PLAYBACK_FPS.
#
# Overridable via env var: DASHBOARD_PLAYBACK_FPS=<int>. Clamped to [1, 60].
PLAYBACK_FPS_CAP: int = max(1, min(60, int(os.getenv("DASHBOARD_PLAYBACK_FPS", "20"))))

# How often (in *kept* video frames, post frame-skip) to trigger an inference
# pass during playback. Inference runs on the last NUM_FRAMES frames of the
# rolling buffer every INFERENCE_STRIDE_FRAMES kept frames.
INFERENCE_STRIDE_FRAMES: int = 8

# Rolling buffer size — must be at least NUM_FRAMES so the preprocessor can
# temporally subsample a full clip per step. Buffer holds only *kept* frames,
# so the clip it represents spans FRAME_BUFFER_SIZE / PLAYBACK_FPS_CAP seconds
# of real video (default 64 / 13 ≈ 4.9s; UniformTemporalSubsample then picks
# 32 evenly-spaced frames, landing close to the 2.0s training distribution).
FRAME_BUFFER_SIZE: int = NUM_FRAMES * 2

# Default top-k for the results panel.
DEFAULT_TOPK: int = 5

# Latency history length (entries) for the p50/p95/p99 summary.
LATENCY_HISTORY: int = 120

# JPEG encode settings for video playback. Baked-HUD frames are encoded
# before handing them to st.image() so the wire payload drops ~10–20×.
# Lower quality → smaller payload but more artifacts; the HUD text stays
# readable down to ~60. max_width downscales larger frames before encoding
# (aspect preserved, never upscales).
JPEG_QUALITY: int = max(1, min(100, int(os.getenv("DASHBOARD_JPEG_QUALITY", "78"))))
JPEG_MAX_WIDTH: int = max(240, min(1920, int(os.getenv("DASHBOARD_JPEG_MAX_WIDTH", "720"))))


# ── Variant registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VariantSpec:
    """Static metadata about a trained SlowFast variant."""

    key: str
    label: str
    checkpoint: Path
    yolo_required: bool


VARIANTS: dict[str, VariantSpec] = {
    "01_baseline": VariantSpec(
        key="01_baseline",
        label="01 · Baseline (SlowFast R50)",
        checkpoint=EXPERIMENTS_DIR / "experiment_01" / "best_baseline.pth",
        yolo_required=False,
    ),
    "02_attention": VariantSpec(
        key="02_attention",
        label="02 · Attention (Non-Local)",
        checkpoint=EXPERIMENTS_DIR / "experiment_02" / "best_attention.pth",
        yolo_required=False,
    ),
    "03_roi": VariantSpec(
        key="03_roi",
        label="03 · ROI Guidance (YOLO)",
        checkpoint=EXPERIMENTS_DIR / "experiment_03" / "best_roi.pth",
        yolo_required=True,
    ),
    "04_hybrid": VariantSpec(
        key="04_hybrid",
        label="04 · Hybrid (ROI + Attention)",
        checkpoint=EXPERIMENTS_DIR / "experiment_04" / "best_hybrid.pth",
        yolo_required=True,
    ),
}

DEFAULT_VARIANT_KEY: str = "01_baseline"


logger.debug(
    "config loaded: project_root=%s experiments=%s video_dir=%s variants=%d playback_fps_cap=%d tta_csvs=%s",
    PROJECT_ROOT,
    EXPERIMENTS_DIR,
    VIDEO_DIR,
    len(VARIANTS),
    PLAYBACK_FPS_CAP,
    [str(p) for p in TTA_ANNOTATIONS_CSVS],
)
