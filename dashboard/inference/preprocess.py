"""Live-clip preprocessor mirroring ``experiments/core/data_loader.py::get_val_transform``.

The four trained variants expect identical preprocessing at inference time.
We rebuild the val transform locally (instead of importing it) so the
dashboard stays independent of experiments.* internals — but the sequence,
constants, and output shapes are the same line-for-line.
"""

from __future__ import annotations

import numpy as np
import torch
from pytorchvideo.transforms import Normalize, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import CenterCrop, Compose, Lambda

from config import CROP_SIZE, MEAN, NUM_FRAMES, SHORT_SIDE, SLOW_FRAMES, SLOWFAST_ALPHA, STD
from logging_setup import get_logger

logger = get_logger(__name__)


class _PackPathway(torch.nn.Module):
    """Clone of ``experiments.core.data_loader.PackPathway``.

    Kept local so the dashboard does not have to import from training code.
    """

    def __init__(self, slow_frames: int, fast_frames: int, alpha: int) -> None:
        super().__init__()
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.alpha = alpha

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, self.slow_frames).long(),
        )
        return [slow_pathway, fast_pathway]


def _build_val_pipeline() -> Compose:
    """Return the same composition used at val time in the training code."""
    return Compose(
        [
            UniformTemporalSubsample(NUM_FRAMES),
            Lambda(lambda x: x / 255.0),
            Normalize(MEAN, STD),
            ShortSideScale(SHORT_SIDE),
            CenterCrop(CROP_SIZE),
            _PackPathway(
                slow_frames=SLOW_FRAMES,
                fast_frames=NUM_FRAMES,
                alpha=SLOWFAST_ALPHA,
            ),
        ]
    )


class ClipPreprocessor:
    """Convert a list of raw RGB frames into the ``[slow, fast]`` pathway pair.

    Input  — ``frames_rgb``: list of (H, W, 3) uint8 arrays or a (T, H, W, 3)
             array. Must contain at least one frame.
    Output — two batched float32 tensors shaped
             ``(1, 3, SLOW_FRAMES, CROP_SIZE, CROP_SIZE)`` and
             ``(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE)``.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._pipeline = _build_val_pipeline()
        logger.debug("ClipPreprocessor ready: device=%s", device)

    def prepare(self, frames_rgb: list[np.ndarray] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize input to a (T, H, W, 3) array of uint8.
        if isinstance(frames_rgb, np.ndarray):
            if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
                raise ValueError(f"frames_rgb array must be (T, H, W, 3); got {frames_rgb.shape}")
            arr = frames_rgb
        else:
            if len(frames_rgb) == 0:
                raise ValueError("frames_rgb is empty")
            arr = np.stack(frames_rgb, axis=0)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        t_in = arr.shape[0]
        if t_in < NUM_FRAMES:
            logger.warning(
                "preprocess: only %d input frames (< %d) — UniformTemporalSubsample will repeat frames",
                t_in,
                NUM_FRAMES,
            )

        # torch wants (C, T, H, W) float before the pipeline applies its own scaling.
        video = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous().float()

        out = self._pipeline(video)
        slow, fast = out  # list of two tensors

        slow = slow.unsqueeze(0).to(self.device, non_blocking=True)
        fast = fast.unsqueeze(0).to(self.device, non_blocking=True)

        logger.debug(
            "preprocess: in_frames=%d slow=%s fast=%s device=%s",
            t_in,
            tuple(slow.shape),
            tuple(fast.shape),
            self.device,
        )
        return slow, fast
