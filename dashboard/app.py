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
    DEFAULT_TOPK,
    EXPERIMENTS_DIR,
    FRAME_BUFFER_SIZE,
    INFERENCE_STRIDE_FRAMES,
    JPEG_MAX_WIDTH,
    JPEG_QUALITY,
    NUM_FRAMES,
    PLAYBACK_FPS_CAP,
    TTA_ANNOTATIONS_CSVS,
    VARIANTS,
    VIDEO_DIR,
)
from inference.engine import InferenceEngine
from inference.metrics import LatencyTracker, TTAComputer
from inference.model_loader import load_label_map, load_variant
from inference.preprocess import ClipPreprocessor
from logging_setup import get_logger, log_cuda_environment
from ui import components, state
from ui.overlay import HudData, draw_hud
from ui.render import frame_to_jpeg_bytes

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

    Loads annotations from all configured CSVs (train + test splits) and
    concatenates them so TTA works for every local video, not just the
    test split.  Returns None if no CSV could be read or the video has no
    matching annotations.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in TTA_ANNOTATIONS_CSVS:
        if not csv_path.exists():
            logger.warning("TTA CSV missing, skipping: %s", csv_path)
            continue
        try:
            frames.append(pd.read_csv(csv_path))
            logger.debug("TTA CSV loaded: %s (%d rows)", csv_path, len(frames[-1]))
        except Exception as err:  # noqa: BLE001
            logger.warning("failed to read TTA CSV %s: %s", csv_path, err)
    if not frames:
        logger.debug("no TTA CSVs could be loaded")
        return None
    df = pd.concat(frames, ignore_index=True)
    logger.debug("TTA annotations merged: %d total rows from %d CSVs", len(df), len(frames))
    vid_id = video_path.stem
    subset = df[df["video_id"] == vid_id]
    if subset.empty:
        logger.debug("no AVA annotations for video_id=%s", vid_id)
        return None
    logger.debug("TTA annotations for video_id=%s: %d rows", vid_id, len(subset))
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

# Main layout — single column. Video + HUD is the primary visual; details go
# into a collapsed expander below.
st.subheader("Stream")
placeholder_frame = st.empty()
placeholder_progress = st.empty()

details_expander = st.expander("Details (top-k, probs, latency history)", expanded=False)
with details_expander:
    placeholder_metrics = st.container()


# Minimum wall-clock interval between `Details` expander refreshes. Playback
# ticks at PLAYBACK_FPS_CAP (default 13 FPS); refreshing the bar chart + line
# chart every frame is wasteful, so we throttle it to ~1 Hz.
_DETAILS_REFRESH_INTERVAL_SEC = 1.0


def _build_hud(result, latency: LatencyTracker | None, tta_val: float | None) -> HudData:
    """Assemble the HUD payload from current session state."""
    action = result.topk[0].action if (result is not None and result.topk) else None
    score = result.topk[0].score if (result is not None and result.topk) else None
    # Latest forward-pass reading — the honest "is this real?" number.
    latency_last = float(result.forward_ms) if result is not None else None
    # Rolling median for context / trend.
    latency_p50 = None
    if latency is not None:
        summary = latency.summary()
        if summary.samples > 0:
            latency_p50 = summary.p50
    return HudData(
        action=action,
        score=score,
        latency_last_ms=latency_last,
        latency_p50_ms=latency_p50,
        tta_sec=tta_val,
    )


def _render_idle() -> None:
    placeholder_frame.info("Press Start to begin playback.")
    with placeholder_metrics:
        components.results_panel(state.get_last_result(), state.get_latency_tracker(), state.get_last_tta())


def _do_playback(single_step: bool) -> None:  # noqa: C901, PLR0912, PLR0915 — main loop
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

    # Frame-skip stride: how many raw-video frames advance per "kept" frame.
    stride = max(1, int(round(video_fps / PLAYBACK_FPS_CAP)))
    target_frame_dt = 1.0 / PLAYBACK_FPS_CAP
    buffer_sec = FRAME_BUFFER_SIZE / PLAYBACK_FPS_CAP

    logger.debug(
        "playback: video_fps=%.1f target_fps=%d stride=%d buffer_sec=%.2f "
        "jpeg_quality=%d jpeg_max_width=%d",
        video_fps,
        PLAYBACK_FPS_CAP,
        stride,
        buffer_sec,
        JPEG_QUALITY,
        JPEG_MAX_WIDTH,
    )

    frame_idx = 0  # real-video frame index (advances by `stride` per iteration)
    kept_idx = 0   # number of frames we have decoded and displayed
    frames_decoded = 0
    last_details_refresh = 0.0
    started_at = time.perf_counter()

    try:
        while True:
            iter_start = time.perf_counter()

            # Decode the "kept" frame. `frame_idx` is the real-video index of
            # *this* frame, so compute `current_sec` from it BEFORE we advance.
            frame_rgb = _read_frame_rgb(cap)
            if frame_rgb is None:
                logger.info("video ended at kept_frame=%d", kept_idx)
                break
            frames_decoded += 1
            current_sec = frame_idx / video_fps
            kept_idx += 1

            buffer.append(frame_rgb)

            # Run inference every INFERENCE_STRIDE_FRAMES *kept* frames once
            # the rolling buffer is warm. This cadence now scales with the
            # playback target (PLAYBACK_FPS_CAP), not the raw video FPS —
            # deliberate: demo-facing inference rate stays steady regardless
            # of the source clip's native FPS.
            ready = len(buffer) >= NUM_FRAMES
            should_infer = ready and (kept_idx % INFERENCE_STRIDE_FRAMES == 0)

            if should_infer:
                try:
                    frames = np.stack(list(buffer)[-FRAME_BUFFER_SIZE:])
                    result = engine.run(frames, k=topk)
                    if latency is not None:
                        latency.record(result)
                    # The observation window ENDS at current_sec (that's the
                    # last frame we just fed the model). Matches the
                    # calculate_tta semantics in experiments/core/metrics.py.
                    result.clip_end_sec = current_sec
                    state.set_last_result(result)

                    if tta is not None:
                        predicted_actions = {p.action for p in result.topk}
                        tta_val = tta.step(current_sec, predicted_actions)
                        state.set_last_tta(tta_val)
                    else:
                        state.set_last_tta(None)
                except Exception as err:  # noqa: BLE001
                    logger.exception("inference error at kept_frame=%d: %s", kept_idx, err)
                    st.error(f"Inference error: {err}")

            # Fast-skip the next (stride - 1) frames without decoding. Do this
            # AFTER we've finished with the current frame (display, inference)
            # so `current_sec` above reflected the frame we actually showed.
            end_of_stream = False
            for _ in range(stride - 1):
                if not cap.grab():
                    end_of_stream = True
                    break
            frame_idx += stride

            # Bake the HUD onto the frame and render with a single st.image.
            # JPEG-encode before st.image so Streamlit's websocket carries
            # ~80KB per frame instead of ~700KB of raw RGB — the biggest
            # single FPS win available without a streaming-video rewrite.
            hud = _build_hud(
                state.get_last_result(),
                latency,
                state.get_last_tta(),
            )
            frame_with_hud = draw_hud(frame_rgb, hud)
            try:
                jpeg_payload = frame_to_jpeg_bytes(
                    frame_with_hud,
                    quality=JPEG_QUALITY,
                    max_width=JPEG_MAX_WIDTH,
                )
                placeholder_frame.image(
                    jpeg_payload,
                    caption=f"t = {current_sec:.2f}s",
                    use_container_width=True,
                )
                if should_infer:
                    # Tie the size log to inference ticks to avoid per-frame spam.
                    logger.debug(
                        "jpeg frame: h=%d w=%d bytes=%d",
                        frame_with_hud.shape[0],
                        frame_with_hud.shape[1],
                        len(jpeg_payload),
                    )
            except Exception as err:  # noqa: BLE001 — never let encode kill playback
                logger.error("jpeg encode failed at kept_frame=%d: %s", kept_idx, err)
                placeholder_frame.image(
                    frame_with_hud,
                    caption=f"t = {current_sec:.2f}s",
                    use_container_width=True,
                )

            if total_frames > 0:
                placeholder_progress.progress(min(1.0, frame_idx / total_frames))

            # Throttle the Details expander refresh to ~1 Hz of wall clock.
            now = time.perf_counter()
            if now - last_details_refresh >= _DETAILS_REFRESH_INTERVAL_SEC:
                with placeholder_metrics:
                    components.results_panel(
                        state.get_last_result(),
                        state.get_latency_tracker(),
                        state.get_last_tta(),
                    )
                last_details_refresh = now

            if single_step or end_of_stream:
                if end_of_stream:
                    logger.info("video ended mid-skip at kept_frame=%d", kept_idx)
                break

            # Respect Stop clicks.
            if not st.session_state.get(state.PLAYING, False):
                logger.debug("playback stopped by user at kept_frame=%d", kept_idx)
                break

            # Pace the loop to hit the target cadence.
            elapsed = time.perf_counter() - iter_start
            time.sleep(max(0.0, target_frame_dt - elapsed))
    finally:
        cap.release()
        total_elapsed = time.perf_counter() - started_at
        logger.info(
            "playback end: frames_shown=%d frames_decoded=%d elapsed=%.1fs",
            kept_idx,
            frames_decoded,
            total_elapsed,
        )
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
