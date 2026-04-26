"""Smoke tests for the dashboard subproject.

These are CPU-only, network-free, and safe to run anywhere. The heaviest
test (one forward pass per variant) is marked `slow` so pre-commit flows
can skip it. Run everything with:

    cd dashboard && make smoke                      # all tests (CPU)
    cd dashboard && PYTEST_ARGS='-m "not slow"' make smoke   # fast subset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Tests sit under dashboard/tests/. Put the dashboard/ directory itself on
# sys.path so imports look the same as when Streamlit runs app.py.
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from config import (  # noqa: E402
    CROP_SIZE,
    EXPERIMENTS_DIR,
    LABEL_MAP_CSV,
    NUM_FRAMES,
    SLOW_FRAMES,
    VARIANTS,
)
from inference.engine import InferenceEngine  # noqa: E402
from inference.metrics import LatencyTracker  # noqa: E402
from inference.model_loader import load_label_map, load_variant  # noqa: E402
from inference.preprocess import ClipPreprocessor  # noqa: E402

CPU = torch.device("cpu")


# ── 14.1 ────────────────────────────────────────────────────────────────────


def test_config_variants_have_checkpoints() -> None:
    if not EXPERIMENTS_DIR.exists():
        pytest.skip(f"EXPERIMENTS_DIR not mounted at {EXPERIMENTS_DIR}")
    missing = [spec.key for spec in VARIANTS.values() if not spec.checkpoint.exists()]
    assert not missing, f"Missing trained checkpoints: {missing}"


# ── 14.2 ────────────────────────────────────────────────────────────────────


def test_preprocessor_shapes() -> None:
    preprocessor = ClipPreprocessor(device=CPU)
    # 40 > NUM_FRAMES so UniformTemporalSubsample gets real work to do.
    frames = np.random.randint(0, 256, size=(40, 300, 400, 3), dtype=np.uint8)
    slow, fast = preprocessor.prepare(frames)

    assert slow.shape == (1, 3, SLOW_FRAMES, CROP_SIZE, CROP_SIZE), slow.shape
    assert fast.shape == (1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE), fast.shape
    assert slow.dtype == torch.float32
    assert fast.dtype == torch.float32


def test_preprocessor_rejects_empty_input() -> None:
    preprocessor = ClipPreprocessor(device=CPU)
    with pytest.raises(ValueError, match="empty"):
        preprocessor.prepare([])


# ── 14.3 ────────────────────────────────────────────────────────────────────


def test_label_map_matches_core_implementation() -> None:
    """Dashboard's standalone label map must match experiments.core.build_label_map."""
    if not LABEL_MAP_CSV.exists():
        pytest.skip(f"label-map CSV missing: {LABEL_MAP_CSV}")

    # Put project root on sys.path so we can import the canonical impl.
    project_root = _DASHBOARD_DIR.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.core.data_loader import build_label_map  # noqa: E402

    dashboard_map = load_label_map(LABEL_MAP_CSV)
    core_map = build_label_map(str(LABEL_MAP_CSV))

    # Dashboard uses str keys for TTA compatibility; core uses int keys.
    # Compare values after normalising keys to strings.
    core_map_str = {str(k): v for k, v in core_map.items()}
    assert dashboard_map == core_map_str, "label map drifted from experiments.core.build_label_map"


# ── 14.4 ────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("variant_key", list(VARIANTS.keys()))
def test_load_each_variant_and_forward(variant_key: str) -> None:
    """Load each checkpoint on CPU and run one forward pass.

    Heavy — SlowFast on CPU is ~0.5–2 s per forward. Run with:
        make smoke
    Skip with:
        PYTEST_ARGS='-m "not slow"' make smoke
    """
    if not VARIANTS[variant_key].checkpoint.exists():
        pytest.skip(f"checkpoint missing for {variant_key}")
    if not LABEL_MAP_CSV.exists():
        pytest.skip(f"label-map CSV missing: {LABEL_MAP_CSV}")

    label_map = load_label_map(LABEL_MAP_CSV)
    num_classes = len(label_map)

    model = load_variant(variant_key, num_classes=num_classes, device=CPU)
    preprocessor = ClipPreprocessor(device=CPU)
    engine = InferenceEngine(model=model, preprocessor=preprocessor, device=CPU, label_map=label_map)

    zeros = np.zeros((NUM_FRAMES, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    result = engine.run(zeros, k=5)

    assert result.forward_ms > 0.0
    assert len(result.topk) == 5
    assert result.probs.shape == (num_classes,)
    assert np.all(np.isfinite(result.probs))


# ── 14.5 ────────────────────────────────────────────────────────────────────


def _fake_result(ms: float):
    """Minimal stand-in for InferenceResult — only forward_ms is read."""

    class _R:
        forward_ms = ms

    return _R()


def test_latency_tracker_summary_monotonic() -> None:
    tracker = LatencyTracker(capacity=200)
    rng = np.random.default_rng(0)
    for ms in rng.uniform(5.0, 100.0, size=100):
        tracker.record(_fake_result(float(ms)))

    summary = tracker.summary()
    assert summary.samples == 100
    assert summary.p50 <= summary.p95 <= summary.p99 <= summary.max
    assert summary.fps > 0.0


def test_latency_tracker_empty_summary() -> None:
    tracker = LatencyTracker()
    summary = tracker.summary()
    assert summary.samples == 0
    assert summary.fps == 0.0


# ── 14.6 — GPU parameter placement ─────────────────────────────────────────


# ── 14.8 — JPEG encoder ──────────────────────────────────────────────────────


def test_frame_to_jpeg_bytes_returns_nonempty_jpeg() -> None:
    from ui.render import frame_to_jpeg_bytes

    frame = np.random.randint(0, 256, size=(360, 640, 3), dtype=np.uint8)
    payload = frame_to_jpeg_bytes(frame, quality=78, max_width=720)

    assert isinstance(payload, bytes)
    assert len(payload) > 100  # non-trivial JPEG
    # JPEG SOI marker is 0xFFD8
    assert payload[:2] == b"\xff\xd8"


def test_frame_to_jpeg_bytes_downscales_preserving_aspect() -> None:
    import cv2

    from ui.render import frame_to_jpeg_bytes

    frame = np.random.randint(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    payload = frame_to_jpeg_bytes(frame, quality=78, max_width=720)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape[1] == 720  # width capped
    # Aspect ratio within 1 pixel of original (1920/1080 * 720 ≈ 405)
    assert abs(decoded.shape[0] - 405) <= 1


def test_frame_to_jpeg_bytes_does_not_upscale() -> None:
    import cv2

    from ui.render import frame_to_jpeg_bytes

    frame = np.random.randint(0, 256, size=(240, 320, 3), dtype=np.uint8)
    payload = frame_to_jpeg_bytes(frame, quality=78, max_width=720)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    # max_width=720 > frame width 320, so no resize should happen
    assert decoded.shape[:2] == (240, 320)


# ── 14.7 — label map key types ───────────────────────────────────────────────


def test_label_map_keys_are_strings() -> None:
    """load_label_map must return str keys for TTA compatibility."""
    if not LABEL_MAP_CSV.exists():
        pytest.skip(f"label-map CSV missing: {LABEL_MAP_CSV}")
    label_map = load_label_map(LABEL_MAP_CSV)
    bad_types = {type(k).__name__ for k in label_map if not isinstance(k, str)}
    assert not bad_types, f"expected all str keys, got non-str types: {bad_types}"


# ── 14.4 — GPU parameter placement (slow) ───────────────────────────────────


@pytest.mark.slow
def test_model_parameters_land_on_cuda_when_available() -> None:
    """Proves GPU routing works end-to-end.

    When CUDA is available, every parameter of the baseline model must land
    on a CUDA device after ``load_variant(..., device=torch.device("cuda"))``.
    Skipped on CPU-only hosts so the fast test subset stays green anywhere.

    We only exercise the baseline — the other three variants share the same
    ``.to(device)`` path (plus variant-specific wrappers), and the post-load
    device check in ``model_loader.load_variant`` already raises on mismatch
    for any of the four.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this host")
    if not LABEL_MAP_CSV.exists():
        pytest.skip(f"label-map CSV missing: {LABEL_MAP_CSV}")

    label_map = load_label_map(LABEL_MAP_CSV)
    model = load_variant(
        "01_baseline",
        num_classes=len(label_map),
        device=torch.device("cuda"),
    )

    devices = {p.device.type for p in model.parameters()}
    assert devices == {"cuda"}, f"expected all parameters on cuda, got {devices}"
