import torch
import torch.nn as nn
from pytorchvideo.layers.nonlocal_net import create_nonlocal
from pytorchvideo.models.slowfast import create_slowfast
from ultralytics import YOLO


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

        # 1. ROI Components
        self.yolo = YOLO(yolo_model)
        self.alpha_blend = alpha_blend
        for param in self.yolo.parameters():
            param.requires_grad = False

        # 2. Attention Components
        self._inject_nonlocal()

    def _inject_nonlocal(self):
        self.nonlocal_res4 = create_nonlocal(dim_in=128, dim_inner=64)
        self.nonlocal_res5 = create_nonlocal(dim_in=256, dim_inner=128)

        if hasattr(self.model, "blocks"):
            self.model.blocks[3] = AttentiveBlockWrapper(self.model.blocks[3], self.nonlocal_res4, pathway=1)
            self.model.blocks[4] = AttentiveBlockWrapper(self.model.blocks[4], self.nonlocal_res5, pathway=1)

    def get_roi_mask(self, x_fast):
        B, C, T, H, W = x_fast.shape
        masks = torch.ones((B, 1, T, H, W), device=x_fast.device)
        # Placeholder for YOLO logic
        return masks

    def forward(self, x):
        slow, fast = x
        # ROI Logic
        # mask = self.get_roi_mask(fast)
        # fast = fast * mask * self.alpha_blend + fast * (1 - self.alpha_blend)

        return self.model([slow, fast])
