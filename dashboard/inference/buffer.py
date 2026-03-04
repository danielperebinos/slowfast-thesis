from collections import deque
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import torch

from experiments.core.data_loader import PackPathway

MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
SIDE_SIZE = 256
CROP_SIZE = 224


class FrameBuffer:
    """Circular buffer that stores raw BGR frames and produces SlowFast pathway tensors."""

    def __init__(self, maxlen: int = 32):
        self._buf: deque = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._pack = PackPathway(slow_frames=4, fast_frames=maxlen)

    def add(self, frame_bgr: np.ndarray) -> None:
        self._buf.append(frame_bgr)

    def is_ready(self) -> bool:
        return len(self._buf) == self._maxlen

    def get_pathways(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (slow_tensor, fast_tensor) with shape (1, C, T, H, W)."""
        frames = list(self._buf)  # list of H×W×3 BGR uint8

        # Convert BGR → RGB, resize short side to 256, stack → (T, H, W, C)
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            if h < w:
                new_h, new_w = SIDE_SIZE, int(w * SIDE_SIZE / h)
            else:
                new_h, new_w = int(h * SIDE_SIZE / w), SIDE_SIZE
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized.append(cv2.resize(f_rgb, (new_w, new_h)))

        # (T, H, W, C) → (C, T, H, W), normalise
        video = torch.from_numpy(np.stack(resized)).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor(MEAN).view(3, 1, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1, 1)
        video = (video - mean) / std  # (C, T, H, W)

        # Center crop
        _, _, H, W = video.shape
        top = (H - CROP_SIZE) // 2
        left = (W - CROP_SIZE) // 2
        video = video[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

        slow, fast = self._pack(video)
        return slow.unsqueeze(0), fast.unsqueeze(0)
