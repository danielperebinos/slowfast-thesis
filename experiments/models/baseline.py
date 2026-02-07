import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast

class BaselineSlowFast(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        model_depth: int = 50,
        slowfast_alpha: int = 8,
        slow_pathway_input_channels: int = 3,
        fast_pathway_input_channels: int = 3,
    ):
        super().__init__()
        self.model = create_slowfast(
            model_depth=model_depth,
            model_num_class=num_classes,
            slowfast_channel_reduction_ratio=(slowfast_alpha,),
            slowfast_fusion_conv_stride=(slowfast_alpha, 1, 1),
            input_channels=(slow_pathway_input_channels, fast_pathway_input_channels),
            head_pool=nn.AdaptiveAvgPool3d,
        )

    def forward(self, x):
        """
        Args:
            x (list): [slow_pathway_input, fast_pathway_input]
        """
        return self.model(x)
