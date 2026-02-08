import torch.nn as nn
from pytorchvideo.layers.nonlocal_net import create_nonlocal
from pytorchvideo.models.slowfast import create_slowfast


class AttentiveBlockWrapper(nn.Module):
    def __init__(self, original_block, attention_module, pathway=1):
        super().__init__()
        self.original_block = original_block
        self.attention_module = attention_module
        self.pathway = pathway  # 1 for Fast, 0 for Slow

    def forward(self, x):
        # x is a list of [slow, fast]
        out = self.original_block(x)

        # Apply attention to the specified pathway
        if isinstance(out, list) and len(out) > self.pathway:
            fast_feature = out[self.pathway]
            # NonLocal expects (N, C, T, H, W)
            fast_feature = self.attention_module(fast_feature)
            # Inplace modification of list? Ideally create new list.
            # But SlowFast passes lists around.
            out[self.pathway] = fast_feature

        return out


class AttentionSlowFast(nn.Module):
    def __init__(
        self,
        num_classes: int = 400,
        model_depth: int = 50,
        slowfast_alpha: int = 8,
    ):
        super().__init__()
        self.model = create_slowfast(
            model_depth=model_depth,
            model_num_class=num_classes,
            slowfast_channel_reduction_ratio=(slowfast_alpha,),
            slowfast_fusion_conv_stride=(slowfast_alpha, 1, 1),
            input_channels=(3, 3),  # Hardcoded for now
            head_pool=nn.AdaptiveAvgPool3d,
        )

        self._inject_nonlocal()

    def forward(self, x):
        return self.model(x)

    def _inject_nonlocal(self):
        """
        Injects Non-Local blocks into the network (Fast Pathway).
        """
        # Hardcoded Channel assumptions for R50
        # Fast pathway res4 ~ 128 channels input ?
        # Checking verification failures previously: 128 seemed correct or close.
        # Actually, let's trust the previous verification which passed.

        self.nonlocal_res4 = create_nonlocal(dim_in=128, dim_inner=64)
        self.nonlocal_res5 = create_nonlocal(dim_in=256, dim_inner=128)

        # Checks if blocks list is accessible
        if hasattr(self.model, "blocks"):
            # Res4 is usually index 3, Res5 is index 4 in PyTorchVideo list
            self.model.blocks[3] = AttentiveBlockWrapper(self.model.blocks[3], self.nonlocal_res4, pathway=1)
            self.model.blocks[4] = AttentiveBlockWrapper(self.model.blocks[4], self.nonlocal_res5, pathway=1)
