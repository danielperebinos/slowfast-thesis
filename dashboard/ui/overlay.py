"""HUD overlay painter — bakes a single 3-line status box onto a video frame.

The overlay is rendered directly onto the frame (not a separate Streamlit
widget) so playback stays at one ``st.image`` call per shown frame. Pure
OpenCV — no Streamlit dependency in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HudData:
    """Three-line HUD payload. Any field may be ``None``; the painter shows
    a dash for missing values and never crashes on all-None input.

    ``latency_last_ms`` is the most recent forward-pass measurement (the
    honest single-sample reading). ``latency_p50_ms`` is the rolling median
    for context. When both are present the HUD shows them side by side.
    """

    action: str | None
    score: float | None  # 0..1
    latency_last_ms: float | None
    latency_p50_ms: float | None
    tta_sec: float | None


# ── internals ────────────────────────────────────────────────────────────────


def _blended_rect(
    frame: np.ndarray,
    tl: tuple[int, int],
    br: tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> None:
    """Alpha-blend a filled rectangle onto ``frame`` in place."""
    x1, y1 = tl
    x2, y2 = br
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _format_line(hud: HudData) -> tuple[str, str, str]:
    """Return the three visible lines, with dashes when fields are missing."""
    if hud.action is None:
        line1 = "— awaiting inference —"
    else:
        score = hud.score if hud.score is not None else 0.0
        line1 = f"{hud.action}  {score:.2f}"

    # Latency line: prefer "last (p50 x)" when both numbers are available;
    # fall back gracefully when one or both are missing. The last-sample
    # reading goes first so the user sees the freshest measurement.
    if hud.latency_last_ms is not None and hud.latency_p50_ms is not None:
        line2 = f"Latency: {hud.latency_last_ms:.1f} ms (p50 {hud.latency_p50_ms:.1f})"
    elif hud.latency_last_ms is not None:
        line2 = f"Latency: {hud.latency_last_ms:.1f} ms"
    elif hud.latency_p50_ms is not None:
        line2 = f"Latency p50: {hud.latency_p50_ms:.1f} ms"
    else:
        line2 = "Latency: — ms"

    if hud.tta_sec is None:
        line3 = "TTA: —"
    else:
        sign = "+" if hud.tta_sec >= 0 else "−"
        line3 = f"TTA: {sign}{abs(hud.tta_sec):.2f} s"

    return line1, line2, line3


# ── public API ───────────────────────────────────────────────────────────────


def draw_hud(frame_rgb: np.ndarray, hud: HudData) -> np.ndarray:
    """Return a copy of ``frame_rgb`` with a 3-line HUD baked in.

    Input shape is (H, W, 3) uint8, RGB. Output has the same shape/dtype.
    The overlay sits in the bottom-left, semi-transparent, with font
    thickness and rectangle size scaled from the frame height.
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"frame_rgb must be (H, W, 3); got {frame_rgb.shape}")

    out = frame_rgb.copy()
    h, w = out.shape[:2]

    # Size knobs — scale with frame height so 224×224 previews and full-res
    # AVA frames both look consistent.
    scale = max(0.35, min(1.1, h / 640.0))
    font_scale = 0.6 * scale
    thickness = max(1, int(round(1.5 * scale)))
    line_spacing = max(18, int(26 * scale))
    pad = max(6, int(10 * scale))

    # Compute rectangle size based on longest formatted line.
    line1, line2, line3 = _format_line(hud)
    font = cv2.FONT_HERSHEY_SIMPLEX
    widest_px = 0
    for text in (line1, line2, line3):
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        widest_px = max(widest_px, tw)

    box_w = min(int(widest_px + 2 * pad), w)
    box_h = line_spacing * 3 + 2 * pad

    x1 = pad
    y2 = h - pad
    x2 = x1 + box_w
    y1 = y2 - box_h

    _blended_rect(out, (x1, y1), (x2, y2), color=(0, 0, 0), alpha=0.55)

    # Draw the three lines from top to bottom, inside the padded rectangle.
    text_x = x1 + pad
    baseline_y = y1 + pad + line_spacing - 4
    for text in (line1, line2, line3):
        cv2.putText(
            out,
            text,
            (text_x, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_spacing

    return out


logger.debug("overlay module loaded")
