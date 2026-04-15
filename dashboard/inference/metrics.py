"""In-memory latency stats and live Time-to-Action computation.

TTA semantics match ``experiments/core/metrics.py::calculate_tta``:

    TTA = prediction_time - action_start_time

    positive → prediction arrived after the action started (Late)
    negative → prediction arrived before the action started (Early, desired)

For a dashboard demo we treat the "prediction_time" as the clip's end
timestamp plus the anticipation gap, and we look up the action onset from
AVA's annotated ground-truth CSV for the currently playing video.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ANTICIPATION_GAP_SEC, LATENCY_HISTORY, TIME_OFFSET_SEC
from inference.engine import InferenceResult
from logging_setup import get_logger

logger = get_logger(__name__)


# ── Latency tracker ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatencySummary:
    p50: float
    p95: float
    p99: float
    mean: float
    max: float
    fps: float
    samples: int


class LatencyTracker:
    """Ring buffer of forward-pass latencies (ms) with percentile stats.

    Thread-safety: not needed — Streamlit runs the playback loop in a single
    script thread; each rerun instantiates a fresh tracker via session state.
    """

    def __init__(self, capacity: int = LATENCY_HISTORY) -> None:
        self._buf: deque[float] = deque(maxlen=max(1, capacity))

    def record(self, result: InferenceResult) -> None:
        self._buf.append(float(result.forward_ms))

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:  # pragma: no cover — trivial
        return len(self._buf)

    def recent(self, n: int = 60) -> list[float]:
        """Return the most recent ``n`` forward_ms samples, oldest → newest.

        Used by the details expander to show a compact latency line-chart.
        Empty list when the buffer is empty.
        """
        if not self._buf:
            return []
        n = max(1, n)
        return list(self._buf)[-n:]

    def summary(self) -> LatencySummary:
        if not self._buf:
            return LatencySummary(
                p50=0.0, p95=0.0, p99=0.0, mean=0.0, max=0.0, fps=0.0, samples=0
            )
        arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean = float(arr.mean())
        mx = float(arr.max())
        fps = (1000.0 / p50) if p50 > 0 else 0.0
        return LatencySummary(p50=p50, p95=p95, p99=p99, mean=mean, max=mx, fps=fps, samples=len(arr))


# ── TTA computer ─────────────────────────────────────────────────────────────


class TTAComputer:
    """Compute Time-to-Action against AVA ground-truth for one video.

    ``annotations`` is expected to be a DataFrame filtered to rows where
    ``video_id`` matches the currently playing clip. We keep only the
    columns we need and sort by ``ts`` ascending for the forward lookup.
    """

    def __init__(self, annotations: pd.DataFrame) -> None:
        required = {"video_id", "ts", "action"}
        missing = required - set(annotations.columns)
        if missing:
            raise ValueError(f"TTAComputer: annotations missing columns: {sorted(missing)}")
        # Keep only rows for a single video to avoid ambiguity.
        self._frame = annotations[["ts", "action"]].copy().sort_values("ts", kind="mergesort").reset_index(drop=True)
        logger.debug("TTAComputer: %d annotation rows", len(self._frame))

    def __len__(self) -> int:  # pragma: no cover
        return len(self._frame)

    def step(self, clip_end_local_sec: float, predicted_actions: set[str]) -> float | None:
        """Return TTA (seconds) if any predicted action matches the next
        annotated action onset; ``None`` otherwise.

        ``clip_end_local_sec`` is the *local* video time (matches the mp4
        timeline) at which the observation window ended. We convert to the
        AVA absolute timeline by adding ``TIME_OFFSET_SEC`` before comparing
        against the annotation ``ts`` column (AVA absolute seconds).
        """
        if self._frame.empty:
            return None

        prediction_ts_ava = clip_end_local_sec + TIME_OFFSET_SEC + ANTICIPATION_GAP_SEC
        # Find the smallest ts >= prediction_ts_ava minus a small epsilon so a
        # prediction exactly at onset still counts.
        eps = 0.05
        candidates = self._frame[self._frame["ts"] >= prediction_ts_ava - eps]
        if candidates.empty:
            logger.debug(
                "tta: clip_end=%.2f pred_ts_ava=%.2f gt=none match=False tta=None",
                clip_end_local_sec,
                prediction_ts_ava,
            )
            return None

        next_row = candidates.iloc[0]
        gt_ts = float(next_row["ts"])
        gt_action = str(next_row["action"])
        match = gt_action in predicted_actions

        if not match:
            logger.debug(
                "tta: clip_end=%.2f gt_ts=%.2f gt_action=%s match=False tta=None",
                clip_end_local_sec,
                gt_ts,
                gt_action,
            )
            return None

        tta = prediction_ts_ava - gt_ts
        logger.debug(
            "tta: clip_end=%.2f gt_ts=%.2f gt_action=%s match=True tta=%.3f",
            clip_end_local_sec,
            gt_ts,
            gt_action,
            tta,
        )
        return float(tta)


# ── Formatting helpers ───────────────────────────────────────────────────────


def format_ms(ms: float) -> str:
    """Human-friendly duration: ``12.3 ms`` under 1 s, ``1.24 s`` at/above."""
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000.0:.2f} s"


def format_tta(tta_sec: float | None) -> str:
    """Show TTA with a sign so ``early`` vs ``late`` is obvious at a glance."""
    if tta_sec is None:
        return "—"
    sign = "+" if tta_sec >= 0 else "−"
    return f"{sign}{abs(tta_sec):.2f} s"
