"""InferencePipeline: orchestrates YOLO person detection + SlowFast action recognition."""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.inference.buffer import FrameBuffer
from dashboard.inference.model_loader import load_model

_LABEL_DIR = Path(__file__).resolve().parent.parent / "labels"
_YOLO_PATH = Path(__file__).resolve().parents[2] / "experiments" / "yolov8n.pt"

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def _load_labels(label_type: str) -> list[str]:
    path = _LABEL_DIR / ("ava_actions.txt" if label_type == "ava" else "kinetics400.txt")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _annotate(
    frame_bgr: np.ndarray,
    boxes: list,
    predictions: list[tuple[str, float]],
    fps: float,
    latency_ms: float,
) -> np.ndarray:
    """Draw bounding boxes and action overlays onto a BGR frame."""
    out = frame_bgr.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(out, "person", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)

    if predictions:
        y_off = 20
        for label, conf in predictions:
            cv2.putText(out, f"{label}: {conf:.2f}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            y_off += 22

    h = out.shape[0]
    cv2.putText(out, f"FPS: {fps:.1f}", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    cv2.putText(out, f"Latency: {latency_ms:.0f}ms", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    return out


class InferencePipeline:
    def __init__(
        self,
        variant: str = "baseline",
        checkpoint_dir: str | Path = "volumes",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("Loading SlowFast variant='{}' on {}", variant, device)
        self.model, self.num_classes, self.label_type = load_model(variant, checkpoint_dir, device)
        self.labels = _load_labels(self.label_type)

        logger.info("Loading YOLO from {}", _YOLO_PATH)
        self.yolo = YOLO(str(_YOLO_PATH))

        self.buffer = FrameBuffer(maxlen=32)
        self.frame_count = 0
        self.last_predictions: list[tuple[str, float]] = []
        self.last_latency_ms: float = 0.0
        self._ts_window: deque = deque(maxlen=30)

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        confidence_threshold: float = 0.5,
        inference_interval: int = 8,
    ) -> tuple[np.ndarray, list[tuple[str, float]], dict]:
        """
        Process one RGB frame.

        Returns
        -------
        annotated_frame : np.ndarray  RGB
        predictions     : list of (label, confidence)
        metrics         : dict {fps, latency_ms, inf_per_sec}
        """
        t_start = time.perf_counter()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # YOLO person detection
        boxes = []
        for r in self.yolo(frame_bgr, classes=[0], verbose=False):
            if r.boxes is not None:
                for b in r.boxes:
                    if float(b.conf[0]) >= confidence_threshold:
                        boxes.append(b.xyxy[0].cpu().numpy())

        # Frame buffer
        self.buffer.add(frame_bgr)
        self.frame_count += 1

        # SlowFast inference
        if self.buffer.is_ready() and self.frame_count % inference_interval == 0:
            slow, fast = self.buffer.get_pathways()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model([slow.to(self.device), fast.to(self.device)])
            self.last_latency_ms = (time.perf_counter() - t0) * 1000

            probs = F.softmax(logits[0], dim=-1)
            top_vals, top_idxs = torch.topk(probs, min(3, self.num_classes))
            self.last_predictions = [
                (self.labels[i] if i < len(self.labels) else f"class_{i}", float(v))
                for i, v in zip(top_idxs.cpu().tolist(), top_vals.cpu().tolist())
            ]
            logger.debug("Top predictions: {}", self.last_predictions)

        # Rolling FPS
        self._ts_window.append(t_start)
        elapsed = self._ts_window[-1] - self._ts_window[0] if len(self._ts_window) >= 2 else 0.0
        fps = (len(self._ts_window) - 1) / elapsed if elapsed > 0 else 0.0

        inf_per_sec = 1000.0 / self.last_latency_ms if self.last_latency_ms > 0 else 0.0
        metrics = {
            "fps": round(fps, 1),
            "latency_ms": round(self.last_latency_ms, 1),
            "inf_per_sec": round(inf_per_sec, 2),
        }

        annotated = _annotate(frame_bgr, boxes, self.last_predictions, fps, self.last_latency_ms)
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), self.last_predictions, metrics
