import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.layers.nonlocal_net import create_nonlocal
from pytorchvideo.models.slowfast import create_slowfast
from ultralytics import YOLO

_MEAN = torch.tensor([0.45, 0.45, 0.45])
_STD = torch.tensor([0.225, 0.225, 0.225])


class AttentiveBlockWrapper(nn.Module):
    def __init__(self, original_block, attention_module, pathway=1):
        super().__init__()
        self.original_block = original_block
        self.attention_module = attention_module
        self.pathway = pathway

    def forward(self, x):
        out = self.original_block(x)
        if isinstance(out, list) and len(out) > self.pathway:
            fast_feature = out[self.pathway]
            fast_feature = self.attention_module(fast_feature)
            out[self.pathway] = fast_feature
        return out


class HybridSlowFast(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        model_depth: int = 50,
        slowfast_alpha: int = 8,
        yolo_model: str = "yolov8n.pt",
        alpha_blend: float = 1.0,
    ):
        super().__init__()
        self.model = create_slowfast(
            model_depth=model_depth,
            model_num_class=num_classes,
            slowfast_channel_reduction_ratio=(slowfast_alpha,),
            slowfast_fusion_conv_stride=(slowfast_alpha, 1, 1),
            input_channels=(3, 3),
            head_pool=nn.AdaptiveAvgPool3d,
        )

        # 1. ROI Components — stored outside nn.Module hierarchy to avoid
        # ultralytics YOLO.train() conflicting with PyTorch's mode-setter.
        self.alpha_blend = alpha_blend
        yolo = YOLO(yolo_model)
        for param in yolo.parameters():
            param.requires_grad = False
        object.__setattr__(self, "_yolo", yolo)

        # 2. Attention Components
        self._inject_nonlocal()

    def _inject_nonlocal(self):
        self.nonlocal_res4 = create_nonlocal(dim_in=128, dim_inner=64)
        self.nonlocal_res5 = create_nonlocal(dim_in=256, dim_inner=128)

        if hasattr(self.model, "blocks"):
            self.model.blocks[3] = AttentiveBlockWrapper(self.model.blocks[3], self.nonlocal_res4, pathway=1)
            self.model.blocks[4] = AttentiveBlockWrapper(self.model.blocks[4], self.nonlocal_res5, pathway=1)

    def get_roi_mask(self, x_fast: torch.Tensor) -> torch.Tensor:
        """Build (B, 1, T, H, W) binary masks from YOLO detections on strided frames."""
        B, C, T, H, W = x_fast.shape
        masks = torch.zeros((B, 1, T, H, W), device=x_fast.device)

        mean = _MEAN.to(x_fast.device).view(3, 1, 1)
        std = _STD.to(x_fast.device).view(3, 1, 1)
        stride = max(1, T // 8)

        with torch.no_grad():
            for b in range(B):
                t = 0
                while t < T:
                    frame = (x_fast[b, :, t] * std + mean).clamp(0.0, 1.0)
                    frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    results = self._yolo(frame_np, verbose=False)
                    boxes = results[0].boxes

                    t_end = min(t + stride, T)
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes.xyxy.cpu():
                            x1, y1, x2, y2 = (
                                int(box[0].item()),
                                int(box[1].item()),
                                int(box[2].item()),
                                int(box[3].item()),
                            )
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)
                            masks[b, 0, t:t_end, y1:y2, x1:x2] = 1.0
                    else:
                        # fallback: no suppression when nothing detected
                        masks[b, 0, t:t_end] = 1.0

                    t += stride

        return masks

    def forward(self, x):
        slow, fast = x

        mask = self.get_roi_mask(fast)
        fast = fast * mask * self.alpha_blend + fast * (1.0 - self.alpha_blend)

        return self.model([slow, fast])
