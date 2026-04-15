"""SlowFast Dashboard — Streamlit entry point.

Run locally:
    uv run streamlit run dashboard/app.py

Run via Docker:
    cd dashboard && make up

GPU is STRONGLY recommended. CPU forward passes for SlowFast typically take
1–3 s per clip, which defeats the purpose of a live demo. Everything still
works on CPU — the latency numbers will just be honestly large.

This app is inference-only. It reads ``best_*.pth`` checkpoints read-only
and never writes to any file under ``../experiments`` or ``../volumes``
(see project RULES.md).
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import (
    CLIP_DURATION_SEC,
    DEFAULT_TOPK,
    EXPERIMENTS_DIR,
    FRAME_BUFFER_SIZE,
    INFERENCE_STRIDE_FRAMES,
    NUM_FRAMES,
    TTA_ANNOTATIONS_CSV,
    VARIANTS,
    VIDEO_DIR,
)
from inference.engine import InferenceEngine
from inference.metrics import LatencyTracker, TTAComputer
from inference.model_loader import load_label_map, load_variant
from inference.preprocess import ClipPreprocessor
from logging_setup import get_logger, log_cuda_environment
from ui import components, state

logger = get_logger(__name__)

# Emit a one-shot CUDA-environment diagnostic at module import time. Runs
# exactly once per fresh Streamlit process (idempotent); subsequent Streamlit
# reruns are no-ops.
log_cuda_environment()


# ── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="SlowFast Dashboard",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_engine(variant_key: str, label_map: dict[str, int], device: torch.device) -> InferenceEngine:
    """Build (or retrieve cached) inference engine for the selected variant."""
    model = load_variant(variant_key, num_classes=len(label_map), device=device)
    preprocessor = ClipPreprocessor(device=device)
    engine = InferenceEngine(
        model=model,
        preprocessor=preprocessor,
        device=device,
        label_map=label_map,
    )
    return engine


def _ensure_engine_ready(variant_key: str, device: torch.device) -> InferenceEngine | None:
    """Lazy-load the engine; show a spinner on the first call or after variant change."""
    engine = state.get_engine()
    if engine is not None:
        return engine

    label_map = state.get_label_map()
    if label_map is None:
        try:
            label_map = load_label_map()
        except FileNotFoundError as err:
            st.error(f"Could not build label map: {err}")
            logger.error("label_map load failed: %s", err)
            return None
        state.set_label_map(label_map)

    try:
        with st.spinner(f"Loading {VARIANTS[variant_key].label} …"):
            engine = _build_engine(variant_key, label_map, device)
            engine.warmup(n=2)
    except FileNotFoundError as err:
        st.error(str(err))
        logger.error("variant load failed: %s", err)
        return None
    except Exception as err:  # noqa: BLE001 — surface everything to the user
        st.error(f"Failed to load variant: {err}")
        logger.exception("unexpected variant load error: %s", err)
        return None

    state.set_engine(engine)
    if state.get_latency_tracker() is None:
        state.set_latency_tracker(LatencyTracker())
    logger.info("engine ready: variant=%s device=%s", variant_key, device)
    return engine


def _build_tta_computer(video_path: Path) -> TTAComputer | None:
    """Look up AVA annotations for this clip and wrap them in a TTAComputer.

    Returns None if the annotations CSV is missing or contains no rows for
    the currently selected video.
    """
    if not TTA_ANNOTATIONS_CSV.exists():
        logger.debug("TTA CSV missing: %s", TTA_ANNOTATIONS_CSV)
        return None
    try:
        df = pd.read_csv(TTA_ANNOTATIONS_CSV)
    except Exception as err:  # noqa: BLE001
        logger.warning("failed to read TTA CSV %s: %s", TTA_ANNOTATIONS_CSV, err)
        return None
    vid_id = video_path.stem
    subset = df[df["video_id"] == vid_id]
    if subset.empty:
        logger.debug("no AVA annotations for video_id=%s", vid_id)
        return None
    return TTAComputer(subset)


def _read_frame_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame_bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ── App body ────────────────────────────────────────────────────────────────


state.init_state()
device = _device()

with st.sidebar:
    st.title("🎞️ SlowFast Dashboard")
    st.caption(
        f"device: **{device.type.upper()}** · experiments: `{EXPERIMENTS_DIR.name}` · videos: `{VIDEO_DIR.name}`"
    )

    variant_key = components.variant_picker()
    video_path = components.stream_picker()
    topk = components.topk_picker() or DEFAULT_TOPK

    components.gpu_status_panel(state.get_engine())

    st.divider()
    cols = st.columns(3)
    start_clicked = cols[0].button("▶️ Start", use_container_width=True, disabled=video_path is None)
    stop_clicked = cols[1].button("⏹️ Stop", use_container_width=True)
    step_clicked = cols[2].button("⏭️ Step", use_container_width=True, disabled=video_path is None)

    st.divider()
    st.caption(
        "GPU strongly recommended — CPU SlowFast forward passes are ~1–3 s "
        "and break the live illusion. TTA shows `—` when no AVA annotations "
        "exist for the selected clip."
    )

if stop_clicked:
    # Stop just stops. Do NOT wipe the global model cache here — the session
    # still holds a reference to the engine, so torch.cuda.empty_cache() would
    # be a no-op on live tensors and would only penalize the next variant
    # switch with a cold reload.
    st.session_state[state.PLAYING] = False
    logger.info("stop clicked")

engine = _ensure_engine_ready(variant_key, device) if video_path is not None else None

# Rebuild TTA computer when the video changes.
prev_video = st.session_state.get("dashboard._prev_video_path")
if video_path is not None and str(video_path) != prev_video:
    state.set_tta_computer(_build_tta_computer(video_path))
    state.set_last_tta(None)
    st.session_state["dashboard._prev_video_path"] = str(video_path)

# Main layout.
left, right = st.columns([3, 4])

with left:
    st.subheader("Stream")
    placeholder_frame = st.empty()
    placeholder_progress = st.empty()

with right:
    st.subheader("Predictions")
    placeholder_metrics = st.container()


def _render_idle() -> None:
    with placeholder_frame.container():
        components.video_frame_preview(None, None)
    with placeholder_metrics:
        components.results_panel(state.get_last_result(), state.get_latency_tracker(), state.get_last_tta())


def _do_playback(single_step: bool) -> None:  # noqa: C901, PLR0912 — main loop
    if engine is None or video_path is None:
        _render_idle()
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error(f"Failed to open video: {video_path}")
        logger.warning("cv2.VideoCapture failed for %s", video_path)
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    buffer: deque[np.ndarray] = deque(maxlen=max(FRAME_BUFFER_SIZE, NUM_FRAMES))
    tta = state.get_tta_computer()
    latency = state.get_latency_tracker()

    frame_idx = 0
    try:
        while True:
            frame_rgb = _read_frame_rgb(cap)
            if frame_rgb is None:
                logger.info("video ended at frame=%d", frame_idx)
                break

            buffer.append(frame_rgb)
            current_sec = frame_idx / video_fps

            # Refresh the frame preview every raw frame (cheap).
            with placeholder_frame.container():
                components.video_frame_preview(frame_rgb, current_sec)
            if total_frames > 0:
                placeholder_progress.progress(min(1.0, (frame_idx + 1) / total_frames))

            # Run inference every INFERENCE_STRIDE_FRAMES once we have a full buffer.
            ready = len(buffer) >= NUM_FRAMES
            should_infer = ready and (frame_idx % INFERENCE_STRIDE_FRAMES == 0)

            if should_infer:
                try:
                    frames = np.stack(list(buffer)[-FRAME_BUFFER_SIZE:])
                    result = engine.run(frames, k=topk)
                    if latency is not None:
                        latency.record(result)
                    result.clip_end_sec = current_sec - CLIP_DURATION_SEC  # approximate local clip-end time
                    state.set_last_result(result)

                    if tta is not None:
                        predicted_actions = {p.action for p in result.topk}
                        tta_val = tta.step(result.clip_end_sec or current_sec, predicted_actions)
                        state.set_last_tta(tta_val)
                    else:
                        state.set_last_tta(None)
                except Exception as err:  # noqa: BLE001
                    logger.exception("inference error at frame=%d: %s", frame_idx, err)
                    st.error(f"Inference error: {err}")

                with placeholder_metrics:
                    components.results_panel(
                        state.get_last_result(),
                        state.get_latency_tracker(),
                        state.get_last_tta(),
                    )

            frame_idx += 1

            if single_step:
                break

            # Respect Stop clicks: re-check the flag after each frame.
            if not st.session_state.get(state.PLAYING, False):
                logger.debug("playback stopped by user at frame=%d", frame_idx)
                break

            # Be nice to Streamlit's event loop.
            time.sleep(max(0.0, 1.0 / video_fps))
    finally:
        cap.release()
        # Do NOT clear_cache() here — we want the engine warm for the next Start.


# Entry-point routing.
if start_clicked and video_path is not None:
    st.session_state[state.PLAYING] = True
    logger.info(
        "playback start: variant=%s video=%s device=%s topk=%d",
        variant_key,
        video_path,
        device,
        topk,
    )
    _do_playback(single_step=False)
    st.session_state[state.PLAYING] = False
elif step_clicked and video_path is not None:
    logger.debug("single-step playback: video=%s", video_path)
    _do_playback(single_step=True)
else:
    _render_idle()


logger.info(
    "dashboard render: cuda=%s experiments=%s video_dir=%s",
    torch.cuda.is_available(),
    EXPERIMENTS_DIR,
    VIDEO_DIR,
)
