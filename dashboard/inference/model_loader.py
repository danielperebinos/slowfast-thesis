"""Load a SlowFast variant from a checkpoint or fall back to the pretrained hub model."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pytorchvideo.models.slowfast import create_slowfast

VARIANTS = ["baseline", "attention", "roi", "hybrid"]
_YOLO_PATH = Path(__file__).resolve().parents[2] / "experiments" / "yolov8n.pt"


def _make_baseline(num_classes: int) -> nn.Module:
    return create_slowfast(
        model_depth=50,
        model_num_class=num_classes,
        slowfast_channel_reduction_ratio=(8,),
        slowfast_fusion_conv_stride=(8, 1, 1),
        input_channels=(3, 3),
        head_pool=nn.AdaptiveAvgPool3d,
    )


def _make_attention(num_classes: int) -> nn.Module:
    from experiments.experiment_02.model import AttentionSlowFast
    return AttentionSlowFast(num_classes=num_classes)


def _make_roi(num_classes: int) -> nn.Module:
    from experiments.experiment_03.model import RoiGuidanceSlowFast
    return RoiGuidanceSlowFast(num_classes=num_classes, yolo_model=str(_YOLO_PATH))


def _make_hybrid(num_classes: int) -> nn.Module:
    from experiments.experiment_04.model import HybridSlowFast
    return HybridSlowFast(num_classes=num_classes, yolo_model=str(_YOLO_PATH))


_FACTORIES = {
    "baseline": _make_baseline,
    "attention": _make_attention,
    "roi": _make_roi,
    "hybrid": _make_hybrid,
}


def _find_checkpoint(variant: str, checkpoint_dir: str | Path) -> Path | None:
    """Search for a .pth file matching the variant name in checkpoint_dir."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        return None
    for p in ckpt_dir.iterdir():
        if p.suffix == ".pth" and variant in p.name.lower():
            return p
    # Also try <checkpoint_dir>/<variant>/checkpoint.pth
    candidate = ckpt_dir / variant / "checkpoint.pth"
    if candidate.is_file():
        return candidate
    return None


def load_model(
    variant: str,
    checkpoint_dir: str | Path = "volumes",
    device: str = "cpu",
) -> tuple[nn.Module, int, str]:
    """
    Load a SlowFast variant.

    Returns
    -------
    model       : nn.Module  (eval mode, on device)
    num_classes : int
    label_type  : "kinetics" | "ava"
    """
    variant = variant.lower()
    if variant not in _FACTORIES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}")

    ckpt_path = _find_checkpoint(variant, checkpoint_dir)

    if ckpt_path is not None:
        logger.info("Loading checkpoint: {}", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        num_classes = ckpt.get("num_classes", 80)
        label_type = "ava"
        model = _FACTORIES[variant](num_classes)
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
        model.load_state_dict(ckpt[state_key] if state_key in ckpt else ckpt, strict=False)
    else:
        logger.warning("No checkpoint found for '{}'. Falling back to pretrained Kinetics-400 weights.", variant)
        num_classes = 400
        label_type = "kinetics"
        if variant == "baseline":
            model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        else:
            model = _FACTORIES[variant](num_classes)

    model.eval()
    model.to(device)
    return model, num_classes, label_type
