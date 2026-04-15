"""Frame-to-JPEG encoder for the Streamlit playback loop.

Streamlit's ``st.image(ndarray)`` serializes frames to base64 JSON over its
websocket — for a 640×360 RGB frame that's ~700 KB per tick, which quickly
saturates the wire. ``st.image(bytes)`` accepts a pre-encoded JPEG payload
directly; typical compression ratio is 10–20× for natural video content,
which is what lets the playback loop hit a higher cadence.

Pure OpenCV + NumPy — no Streamlit dependency.
"""

from __future__ import annotations

import cv2
import numpy as np

from logging_setup import get_logger

logger = get_logger(__name__)


def frame_to_jpeg_bytes(
    frame_rgb: np.ndarray,
    quality: int = 78,
    max_width: int | None = 720,
) -> bytes:
    """Convert an (H, W, 3) uint8 RGB frame to JPEG-encoded bytes.

    Optionally downscales (never upscales) so width <= max_width, preserving
    aspect ratio. Uses OpenCV's INTER_AREA for downsample quality.

    Parameters
    ----------
    frame_rgb : np.ndarray
        (H, W, 3) uint8 RGB frame.
    quality : int, default 78
        JPEG quality 0..100. 78 is a sweet spot for 720p natural content
        (barely visible artifacts, ~80 KB per frame).
    max_width : int | None, default 720
        Downscale when the frame is wider than this. ``None`` disables
        downscaling entirely. Smaller frames are never upscaled.

    Returns
    -------
    bytes
        Raw JPEG payload suitable for ``st.image(bytes)``.

    Raises
    ------
    ValueError
        On malformed input (wrong ndim / channels / dtype) or invalid
        ``quality`` outside the 0..100 range.
    RuntimeError
        If ``cv2.imencode`` returns failure (extremely rare; usually means
        the provided quality was rejected by the build).
    """
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"frame_rgb must be (H, W, 3); got shape={frame_rgb.shape}")
    if frame_rgb.dtype != np.uint8:
        raise ValueError(f"frame_rgb must be uint8; got dtype={frame_rgb.dtype}")
    if not (0 <= quality <= 100):
        raise ValueError(f"quality must be in [0, 100]; got {quality}")

    # cv2.imencode expects BGR.
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Downscale (never upscale) while preserving aspect ratio.
    if max_width is not None and bgr.shape[1] > max_width:
        ratio = max_width / float(bgr.shape[1])
        new_w = max_width
        new_h = max(1, int(round(bgr.shape[0] * ratio)))
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed (quality={quality})")

    return buf.tobytes()


logger.debug("render module loaded")
