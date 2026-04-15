"""Reusable Streamlit components for the dashboard.

Each function takes explicit arguments and returns a selection — no global
mutation beyond the ``state.py`` helpers. Keeping side effects narrow makes
it easier to reason about Streamlit's rerun model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import VARIANTS, VIDEO_DIR
from inference.engine import InferenceEngine, InferenceResult
from inference.metrics import LatencyTracker, format_ms, format_tta
from logging_setup import get_logger
from ui import state

logger = get_logger(__name__)

_BYTES_PER_GIB = 1024 ** 3


# ── Pickers ──────────────────────────────────────────────────────────────────


def variant_picker() -> str:
    """Render the variant selector; resets the engine on change."""
    keys = list(VARIANTS.keys())
    labels = {k: VARIANTS[k].label for k in keys}
    current = state.get_variant_key()
    current_idx = keys.index(current) if current in keys else 0

    choice = st.selectbox(
        "Model variant",
        options=keys,
        index=current_idx,
        format_func=lambda k: labels[k],
        help="Which trained SlowFast variant to run inference with.",
    )

    if choice != current:
        logger.debug("variant_picker: selected=%s (was=%s)", choice, current)
        st.session_state[state.VARIANT_KEY] = choice
        state.reset_engine()

    return choice


@st.cache_data(ttl=5)
def _list_videos(video_dir: str) -> list[str]:
    """List ``*.mp4`` under ``video_dir``, cached for 5 s."""
    root = Path(video_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*.mp4"))


def stream_picker() -> Path | None:
    """Render the stream selector. Returns the chosen Path (or None)."""
    videos = _list_videos(str(VIDEO_DIR))
    if not videos:
        st.warning(
            f"No videos found in {VIDEO_DIR}. Place AVA `.mp4` clips there "
            "(the directory is bind-mounted read-only in Docker)."
        )
        return None

    labels = {v: Path(v).name for v in videos}
    current = st.session_state.get(state.VIDEO_PATH)
    default_idx = videos.index(current) if current in videos else 0

    chosen = st.selectbox(
        "Video stream",
        options=videos,
        index=default_idx,
        format_func=lambda v: labels[v],
        help="AVA .mp4 clip to play and run inference on.",
    )

    if chosen != current:
        logger.debug("stream_picker: selected=%s", chosen)
        st.session_state[state.VIDEO_PATH] = chosen

    return Path(chosen)


def topk_picker() -> int:
    value = st.slider(
        "Top-k predictions",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get(state.TOPK, 5)),
        step=1,
    )
    st.session_state[state.TOPK] = value
    return value


# ── Displays ─────────────────────────────────────────────────────────────────


def video_frame_preview(frame_rgb: np.ndarray | None, current_sec: float | None) -> None:
    """Show the current video frame with a second-counter caption."""
    if frame_rgb is None:
        st.info("Press Start to begin playback.")
        return
    caption = f"t = {current_sec:.2f}s" if current_sec is not None else ""
    st.image(frame_rgb, caption=caption, use_container_width=True)


def results_panel(
    result: InferenceResult | None,
    latency: LatencyTracker | None,
    tta_value: float | None,
) -> None:
    """Three-column metrics row + top-k bar chart + expandable raw probs."""
    col_forward, col_fps, col_tta = st.columns(3)

    if result is None:
        col_forward.metric("Forward", "—")
        col_fps.metric("FPS (p50)", "—")
        col_tta.metric("TTA", "—")
        st.caption("Metrics appear after the first inference pass.")
        return

    summary = latency.summary() if latency is not None else None
    fps_label = f"{summary.fps:.1f}" if (summary and summary.fps > 0) else "—"
    col_forward.metric("Forward", format_ms(result.forward_ms))
    col_fps.metric("FPS (p50)", fps_label)
    col_tta.metric("TTA", format_tta(tta_value))

    if summary is not None:
        st.caption(
            f"latency: p50={format_ms(summary.p50)} · p95={format_ms(summary.p95)} · "
            f"p99={format_ms(summary.p99)} · max={format_ms(summary.max)} · n={summary.samples}"
        )

    # Top-k bar chart: action -> score.
    chart_df = pd.DataFrame(
        {
            "action": [p.action for p in result.topk],
            "score": [p.score for p in result.topk],
        }
    ).set_index("action")
    st.bar_chart(chart_df, horizontal=True, height=24 * max(1, len(result.topk)) + 40)

    with st.expander("Raw probabilities"):
        raw = pd.DataFrame(
            {
                "class_idx": list(range(len(result.probs))),
                "prob": result.probs,
            }
        ).sort_values("prob", ascending=False)
        st.dataframe(raw, height=260, use_container_width=True)


# ── GPU / device status ──────────────────────────────────────────────────────


def _cudnn_version() -> str:
    try:
        v = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        return "N/A"
    return str(v) if v is not None else "N/A"


def _model_param_device(engine: InferenceEngine | None) -> str:
    if engine is None:
        return "—"
    try:
        return str(next(engine.model.parameters()).device)
    except StopIteration:
        return "<no-params>"
    except Exception as err:  # noqa: BLE001
        return f"<err: {err}>"


def gpu_status_panel(engine: InferenceEngine | None) -> None:
    """Sidebar expander showing live CUDA / CPU state.

    Reads torch state on every Streamlit rerun. No caching — the underlying
    calls (`torch.cuda.is_available`, `mem_get_info`, `get_device_name`) are
    cheap driver queries.
    """
    with st.expander("🖥️ GPU status", expanded=False):
        cuda_available = torch.cuda.is_available()
        cuda_ver = getattr(torch.version, "cuda", None) or "N/A"

        if not cuda_available:
            st.markdown(f"**CPU** · Torch `{torch.__version__}`")
            st.caption(
                "No CUDA runtime visible. SlowFast forward passes on CPU are "
                "typically 1–3 s per clip; expect the latency metric to stay in "
                "the hundreds-of-ms-plus range. GPU is strongly recommended "
                "for the defense demo."
            )
            return

        device_count = torch.cuda.device_count()
        st.markdown(
            f"**CUDA** · Torch `{torch.__version__}` · CUDA `{cuda_ver}` · "
            f"cuDNN `{_cudnn_version()}`"
        )
        st.caption(f"Visible GPUs: {device_count}")

        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_gib = props.total_memory / _BYTES_PER_GIB
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(i)
                    free_gib = free_bytes / _BYTES_PER_GIB
                    mem_line = f"{free_gib:.1f} / {total_gib:.1f} GiB free"
                except (RuntimeError, AttributeError):
                    mem_line = f"{total_gib:.1f} GiB total · free query N/A"
                st.markdown(f"- GPU `{i}`: **{name}** — {mem_line}")
            except Exception as err:  # noqa: BLE001, PERF203 — up to 8 GPUs, per-rerun UI
                st.caption(f"GPU {i}: diagnostics failed ({err})")

        st.markdown(f"**Model on:** `{_model_param_device(engine)}`")
        st.caption(
            f"cudnn.benchmark = {torch.backends.cudnn.benchmark} · "
            f"cudnn.enabled = {torch.backends.cudnn.enabled}"
        )
