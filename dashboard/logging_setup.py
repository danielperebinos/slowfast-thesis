"""Shared logger factory for the dashboard.

Keeps verbose DEBUG output during implementation while letting operators
dial it down via the ``DASHBOARD_LOG_LEVEL`` environment variable (the plan
mandates verbose logging for this dashboard).
"""

from __future__ import annotations

import logging
import os
import sys
import threading

_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()

# Third-party loggers that are useful but hyperactive — cap them so the
# dashboard log stream stays readable.
_THIRD_PARTY_CAPS: dict[str, int] = {
    "ultralytics": logging.INFO,
    "matplotlib": logging.WARNING,
    "PIL": logging.WARNING,
    "torch": logging.INFO,
}

_LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _resolve_level() -> int:
    """Resolve the dashboard log level from env, default DEBUG (verbose)."""
    raw = os.getenv("DASHBOARD_LOG_LEVEL", "DEBUG").strip().upper()
    # logging.getLevelName returns the numeric level when given a name
    level = logging.getLevelName(raw)
    if not isinstance(level, int):
        # Fallback to DEBUG if the user set an unknown label
        level = logging.DEBUG
    return level


def _configure_once() -> int:
    """Idempotent one-shot root-logger configuration."""
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return _resolve_level()

        level = _resolve_level()
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

        root = logging.getLogger()
        # Clear pre-existing handlers so Streamlit's reruns do not stack them.
        for existing in list(root.handlers):
            root.removeHandler(existing)
        root.addHandler(handler)
        root.setLevel(level)

        # Apply caps only when the overall level is looser than the cap;
        # if the user explicitly asked for DEBUG we keep third-party DEBUG too.
        for name, cap in _THIRD_PARTY_CAPS.items():
            logging.getLogger(name).setLevel(max(level, cap))

        _CONFIGURED = True
        logging.getLogger(__name__).debug(
            "logger initialized: level=%s", logging.getLevelName(level)
        )
        return level


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger.

    First call triggers the one-time root setup; subsequent calls are cheap.
    """
    _configure_once()
    return logging.getLogger(name)


# ── CUDA environment diagnostics ────────────────────────────────────────────

_CUDA_ENV_LOGGED = False
_CUDA_ENV_LOCK = threading.Lock()

_BYTES_PER_GIB = 1024 ** 3


def log_cuda_environment() -> None:
    """Log a one-shot snapshot of the CUDA environment the dashboard sees.

    Emits:
      - one INFO line with torch / CUDA / cuDNN versions and device count
      - one INFO line per visible GPU with name, capability, and memory

    Idempotent: subsequent calls inside the same Python process are no-ops.
    Never raises; per-device failures degrade to a WARN line.
    """
    global _CUDA_ENV_LOGGED
    with _CUDA_ENV_LOCK:
        if _CUDA_ENV_LOGGED:
            return

        # Import lazily so importing logging_setup has no torch dependency
        # ordering requirement.
        import torch  # noqa: PLC0415

        logger = get_logger(__name__)

        cuda_available = torch.cuda.is_available()
        torch_ver = torch.__version__
        cuda_ver = getattr(torch.version, "cuda", None) or "N/A"
        try:
            cudnn_ver = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            cudnn_ver = None
        cudnn_enabled = bool(getattr(torch.backends.cudnn, "enabled", False))

        if not cuda_available:
            logger.info(
                "cuda_env: torch=%s cuda=N/A cudnn=N/A devices=0 (CPU-only mode)",
                torch_ver,
            )
            _CUDA_ENV_LOGGED = True
            return

        try:
            device_count = torch.cuda.device_count()
        except Exception as err:  # noqa: BLE001
            logger.warning("cuda_env: device_count query failed: %s", err)
            device_count = 0

        logger.info(
            "cuda_env: torch=%s cuda=%s cudnn=%s enabled=%s devices=%d",
            torch_ver,
            cuda_ver,
            cudnn_ver if cudnn_ver is not None else "N/A",
            cudnn_enabled,
            device_count,
        )

        for i in range(device_count):
            try:
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_gib = props.total_memory / _BYTES_PER_GIB
                # mem_get_info exists on recent PyTorch + CUDA combos; fall
                # back gracefully if the call fails.
                try:
                    free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
                    free_gib = free_bytes / _BYTES_PER_GIB
                except (RuntimeError, AttributeError) as err:
                    logger.warning(
                        "cuda_device[%d]: mem_get_info failed (%s) — memory field skipped",
                        i,
                        err,
                    )
                    free_gib = float("nan")

                logger.info(
                    "cuda_device[%d]: name=%s capability=%d.%d memory=%.1fGiB free=%.1fGiB",
                    i,
                    name,
                    props.major,
                    props.minor,
                    total_gib,
                    free_gib,
                )
            except Exception as err:  # noqa: BLE001, PERF203 — boot-time only
                logger.warning("cuda_device[%d]: diagnostics failed: %s", i, err)

        _CUDA_ENV_LOGGED = True
