"""Load one of the four trained SlowFast variants for inference.

Contract
--------
- The four ``best_*.pth`` files contain only a ``state_dict`` — no optimizer
  or label_map. The full ``checkpoint.pth`` (training state) is intentionally
  ignored here.
- Checkpoints are read-only. This module MUST NEVER write to them
  (project rule: never modify the folders/files with trained models).
- Experiment 01 is rebuilt inline with ``pytorchvideo.create_slowfast`` so we
  do not have to import from ``experiment_01/train.py`` — per
  ARCHITECTURE.md the train.py files are entry points, not modules.
- Experiments 02–04 import their clean ``model.py`` wrappers. Those modules
  have no import-time side effects beyond the usual ``torch`` stack.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from config import LABEL_MAP_CSV, PROJECT_ROOT, VARIANTS, YOLO_WEIGHTS, VariantSpec
from logging_setup import get_logger

logger = get_logger(__name__)

# Cache of loaded models keyed by (variant_key, device_str, num_classes).
_MODEL_CACHE: dict[tuple[str, str, int], nn.Module] = {}
_CACHE_LOCK = threading.Lock()

_PROJECT_ROOT_ADDED = False
_PROJECT_ROOT_LOCK = threading.Lock()


def _ensure_project_on_sys_path() -> None:
    """Idempotently put the slowfast-thesis root on sys.path so we can
    import ``experiments.experiment_0{2,3,4}.model``."""
    global _PROJECT_ROOT_ADDED
    with _PROJECT_ROOT_LOCK:
        if _PROJECT_ROOT_ADDED:
            return
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _PROJECT_ROOT_ADDED = True
        logger.debug("sys.path extended with %s", root)


def _build_experiment_01(num_classes: int) -> nn.Module:
    """Rebuild the Baseline model — identical kwargs to experiment_01/train.py."""
    from pytorchvideo.models.slowfast import create_slowfast

    return create_slowfast(
        model_depth=50,
        model_num_class=num_classes,
        slowfast_channel_reduction_ratio=(8,),
        slowfast_fusion_conv_stride=(8, 1, 1),
        input_channels=(3, 3),
        head_pool=nn.AdaptiveAvgPool3d,
    )


def _build_experiment_02(num_classes: int) -> nn.Module:
    _ensure_project_on_sys_path()
    from experiments.experiment_02.model import AttentionSlowFast

    return AttentionSlowFast(num_classes=num_classes)


def _build_experiment_03(num_classes: int) -> nn.Module:
    _ensure_project_on_sys_path()
    from experiments.experiment_03.model import RoiGuidanceSlowFast

    return RoiGuidanceSlowFast(
        num_classes=num_classes,
        yolo_model=str(YOLO_WEIGHTS),
        alpha_blend=1.0,
    )


def _build_experiment_04(num_classes: int) -> nn.Module:
    _ensure_project_on_sys_path()
    from experiments.experiment_04.model import HybridSlowFast

    return HybridSlowFast(
        num_classes=num_classes,
        yolo_model=str(YOLO_WEIGHTS),
        alpha_blend=1.0,
    )


_BUILDERS: dict[str, callable] = {
    "01_baseline": _build_experiment_01,
    "02_attention": _build_experiment_02,
    "03_roi": _build_experiment_03,
    "04_hybrid": _build_experiment_04,
}


def _summarize_state_dict_mismatch(
    model_keys: set[str], ckpt_keys: set[str]
) -> tuple[int, int, list[str]]:
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    samples = sorted(list(missing)[:3] + list(unexpected)[:3])
    return len(missing), len(unexpected), samples


def load_variant(key: str, num_classes: int, device: torch.device) -> nn.Module:
    """Return a fully-loaded, ``eval()``-ed model for the requested variant.

    Results are cached per ``(key, device, num_classes)``.
    Raises ``FileNotFoundError`` if the checkpoint file is missing.
    """
    spec: VariantSpec | None = VARIANTS.get(key)
    if spec is None:
        raise KeyError(f"Unknown variant key: {key!r}. Known: {sorted(VARIANTS)}")

    cache_key = (key, str(device), num_classes)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("variant cache hit: key=%s device=%s", key, device)
            return cached

    logger.debug("load_variant: key=%s device=%s num_classes=%d", key, device, num_classes)

    if not spec.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {spec.checkpoint}. "
            f"Run experiments/{spec.key.split('_', 1)[0]}_*/train.py to produce it."
        )

    builder = _BUILDERS[key]
    model = builder(num_classes)

    state = torch.load(spec.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        # Defensive — should not happen for best_*.pth, but be lenient if a
        # future variant saves a full checkpoint dict here.
        logger.warning("checkpoint contains a full dict; using its 'model' sub-key")
        state = state["model"]

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as err:
        missing, unexpected, samples = _summarize_state_dict_mismatch(
            set(model.state_dict().keys()), set(state.keys())
        )
        logger.warning(
            "strict load failed for variant=%s: missing=%d unexpected=%d (samples=%s). "
            "retrying with strict=False. original_error=%s",
            key,
            missing,
            unexpected,
            samples,
            err,
        )
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())

    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = model

    logger.info(
        "loaded variant %s (%d params) from %s on %s",
        key,
        num_params,
        spec.checkpoint,
        device,
    )

    # ── Post-`.to()` device verification ────────────────────────────────────
    # Catches the silent failure mode where model.to(cuda) succeeds but a
    # sub-module stays on CPU (happens when a tensor is stored outside the
    # nn.Module hierarchy — e.g. variants 03/04 use `object.__setattr__` for
    # their YOLO wrapper, which is intentional; other unintended cases of
    # the same pattern would be bugs).
    try:
        first_param_device = next(model.parameters()).device
    except StopIteration:
        # Pathological: a model with zero parameters. Nothing to verify.
        logger.warning("variant %s: model has no parameters to verify device placement", key)
        return model

    param_devices = {str(p.device) for p in model.parameters()}
    buffer_devices = {str(b.device) for b in model.buffers()}

    logger.info(
        "variant %s device check: requested=%s first_param=%s all_param_devices=%s buffer_devices=%s",
        key,
        device,
        first_param_device,
        sorted(param_devices),
        sorted(buffer_devices),
    )

    if device.type == "cuda" and first_param_device.type != "cuda":
        msg = (
            f"Model did not move to CUDA for variant {key!r}: first parameter is on "
            f"{first_param_device}. Check nvidia-container-toolkit / driver visibility "
            f"(e.g. `docker info | grep -i runtimes`, `nvidia-smi`)."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if len(param_devices) > 1:
        # Expected only for variants 03/04 where YOLO is stored via
        # object.__setattr__ (see experiments/experiment_03/model.py). Surface
        # as WARN so a future unintended case is noticeable in the logs.
        logger.warning(
            "variant %s has parameters on mixed devices: %s (expected for ROI/Hybrid variants "
            "with external YOLO)",
            key,
            sorted(param_devices),
        )

    return model


def load_label_map(train_csv: Path | str | None = None) -> dict[str, int]:
    """Rebuild the same ``action_id -> class_index`` map used at training time.

    Matches ``experiments.core.data_loader.build_label_map`` line-for-line:
    sorted-unique action IDs mapped to consecutive integers starting at 0.
    A parity test in ``tests/test_smoke.py`` guards against drift.
    """
    path = Path(train_csv) if train_csv is not None else LABEL_MAP_CSV
    if not path.exists():
        raise FileNotFoundError(f"Label-map CSV not found: {path}")
    df = pd.read_csv(path)
    action_ids = sorted(df["action"].unique().tolist())
    label_map = {aid: i for i, aid in enumerate(action_ids)}
    logger.debug("load_label_map: csv=%s classes=%d", path, len(label_map))
    return label_map


def clear_cache() -> None:
    """Release cached models. Useful on variant switch to free GPU memory."""
    with _CACHE_LOCK:
        released = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("clear_cache: released=%d entries", released)
