"""End-to-end inference engine with honest latency measurement.

Latency on CUDA is asynchronous — wall clock around a ``model(...)`` call
lies unless we bracket it with ``torch.cuda.synchronize()``. Do that here,
once, and give the rest of the dashboard a single ``InferenceResult`` with
numbers we can trust on defense day.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from inference.preprocess import ClipPreprocessor
from logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    action: str
    score: float
    class_idx: int


@dataclass
class InferenceResult:
    topk: list[Prediction]
    forward_ms: float
    preprocess_ms: float
    total_ms: float
    probs: np.ndarray  # shape (num_classes,)
    # Dashboard-level metadata filled in by the caller (not by the engine).
    clip_end_sec: float | None = None
    meta: dict = field(default_factory=dict)


class InferenceEngine:
    """Owns a loaded model + its preprocessor + label map."""

    def __init__(
        self,
        model: nn.Module,
        preprocessor: ClipPreprocessor,
        device: torch.device,
        label_map: dict[str, int],
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.label_map = label_map
        # One-time device-log flag: set after the first successful forward so
        # we emit exactly one INFO line per engine instance (resets on variant
        # switch because a new engine is constructed).
        self._device_logged: bool = False
        # Reverse map: class_idx -> action name. Label map is guaranteed to
        # be dense (0..N-1) by the training-time construction.
        self._index_to_action: list[str] = [""] * len(label_map)
        for action, idx in label_map.items():
            self._index_to_action[idx] = action
        logger.debug(
            "InferenceEngine ready: device=%s num_classes=%d",
            device,
            len(label_map),
        )

    # ── internal helpers ────────────────────────────────────────────────────

    def _sync(self) -> None:
        """Block until any pending CUDA work on our device is done."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model([slow, fast])

    # ── public API ──────────────────────────────────────────────────────────

    def warmup(self, n: int = 2) -> None:
        """Run a few dummy passes to trigger CUDA kernel JIT + YOLO init.

        A cold first click during the defense demo would show an artificially
        large latency spike — this smooths that out before the user sees any
        numbers.
        """
        from config import CROP_SIZE, NUM_FRAMES

        # Use zero frames shaped to match the video decoder output: (T, H, W, 3)
        dummy = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        last_forward = 0.0
        for i in range(max(1, n)):
            result = self.run(dummy, k=1)
            last_forward = result.forward_ms
            logger.debug("warmup pass %d: forward_ms=%.1f", i + 1, last_forward)
        logger.info("warmup complete: n=%d last_forward_ms=%.1f", n, last_forward)

    def run(self, frames_rgb, k: int = 5) -> InferenceResult:
        """Full preprocess + forward + top-k, with CUDA-sync'd timing."""
        # --- preprocess ---
        t0 = time.perf_counter()
        slow, fast = self.preprocessor.prepare(frames_rgb)
        # prepare() already did .to(device, non_blocking=True); sync so the
        # measured preprocess_ms actually reflects host-to-device transfer.
        self._sync()
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        # --- forward ---
        t1 = time.perf_counter()
        logits = self._forward(slow, fast)
        self._sync()
        forward_ms = (time.perf_counter() - t1) * 1000.0

        # ── First-inference device log (once per engine) ───────────────────
        # Proves GPU usage (or CPU fallback) end-to-end: slow/fast tensors
        # actually on device, logits on device, model parameter on device.
        if not self._device_logged:
            try:
                model_param_device = next(self.model.parameters()).device
            except StopIteration:
                model_param_device = "<no-params>"
            logger.info(
                "first inference: model_device=%s slow=%s fast=%s logits=%s engine_device=%s",
                model_param_device,
                slow.device,
                fast.device,
                logits.device,
                self.device,
            )
            self._device_logged = True

        total_ms = preprocess_ms + forward_ms

        # Multi-label: sigmoid (training loss is BCE-with-logits).
        probs = torch.sigmoid(logits).squeeze(0)
        k = max(1, min(k, probs.numel()))
        top_scores, top_indices = torch.topk(probs, k)

        topk: list[Prediction] = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
            action = self._index_to_action[idx] if 0 <= idx < len(self._index_to_action) else f"cls_{idx}"
            topk.append(Prediction(action=action, score=float(score), class_idx=int(idx)))

        probs_np = probs.detach().cpu().numpy()

        logger.debug(
            "inference: preprocess=%.1fms forward=%.1fms top1=%s (%.3f)",
            preprocess_ms,
            forward_ms,
            topk[0].action,
            topk[0].score,
        )

        return InferenceResult(
            topk=topk,
            forward_ms=forward_ms,
            preprocess_ms=preprocess_ms,
            total_ms=total_ms,
            probs=probs_np,
        )
