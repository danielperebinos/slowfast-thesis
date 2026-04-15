"""Centralize Streamlit ``st.session_state`` keys.

Streamlit reruns the script on every widget interaction. Keeping all state
keys in one module — instead of sprinkling raw strings across the UI —
prevents typo-driven bugs and lets us reset related values together when
the user switches variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from config import DEFAULT_VARIANT_KEY
from logging_setup import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from inference.engine import InferenceEngine, InferenceResult
    from inference.metrics import LatencyTracker, TTAComputer

logger = get_logger(__name__)

# Canonical session-state keys.
VARIANT_KEY = "dashboard.variant_key"
VIDEO_PATH = "dashboard.video_path"
PLAYING = "dashboard.playing"
LATENCY_TRACKER = "dashboard.latency_tracker"
TTA_COMPUTER = "dashboard.tta_computer"
LAST_RESULT = "dashboard.last_result"
LABEL_MAP = "dashboard.label_map"
ENGINE = "dashboard.inference_engine"
LAST_TTA = "dashboard.last_tta"
TOPK = "dashboard.topk"


def init_state() -> None:
    """Create every session-state key with its default value, once."""
    defaults: dict[str, object] = {
        VARIANT_KEY: DEFAULT_VARIANT_KEY,
        VIDEO_PATH: None,
        PLAYING: False,
        LATENCY_TRACKER: None,
        TTA_COMPUTER: None,
        LAST_RESULT: None,
        LABEL_MAP: None,
        ENGINE: None,
        LAST_TTA: None,
        TOPK: 5,
    }
    created = 0
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            created += 1
    if created:
        logger.debug("init_state: created %d/%d session keys", created, len(defaults))


def reset_engine() -> None:
    """Clear cached inference engine + dependent derived state.

    Call this whenever the user switches variants so we rebuild the engine
    (and free GPU memory from the previous variant).
    """
    logger.debug("reset_engine: clearing engine/result/latency/TTA")
    st.session_state[ENGINE] = None
    st.session_state[LAST_RESULT] = None
    st.session_state[LAST_TTA] = None
    tracker: LatencyTracker | None = st.session_state.get(LATENCY_TRACKER)
    if tracker is not None:
        tracker.reset()


# ── Typed getters (IDE-friendly, optional) ──────────────────────────────────


def get_variant_key() -> str:
    return st.session_state[VARIANT_KEY]


def get_engine() -> InferenceEngine | None:
    return st.session_state[ENGINE]


def set_engine(engine: InferenceEngine | None) -> None:
    st.session_state[ENGINE] = engine


def get_latency_tracker() -> LatencyTracker | None:
    return st.session_state[LATENCY_TRACKER]


def set_latency_tracker(tracker: LatencyTracker) -> None:
    st.session_state[LATENCY_TRACKER] = tracker


def get_tta_computer() -> TTAComputer | None:
    return st.session_state[TTA_COMPUTER]


def set_tta_computer(computer: TTAComputer | None) -> None:
    st.session_state[TTA_COMPUTER] = computer


def get_label_map() -> dict[str, int] | None:
    return st.session_state[LABEL_MAP]


def set_label_map(label_map: dict[str, int]) -> None:
    st.session_state[LABEL_MAP] = label_map


def get_last_result() -> InferenceResult | None:
    return st.session_state[LAST_RESULT]


def set_last_result(result: InferenceResult) -> None:
    st.session_state[LAST_RESULT] = result


def get_last_tta() -> float | None:
    return st.session_state[LAST_TTA]


def set_last_tta(value: float | None) -> None:
    st.session_state[LAST_TTA] = value
