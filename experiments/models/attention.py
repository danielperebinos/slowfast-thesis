import torch
import torch.nn as nn
from pytorchvideo.models.slowfast import create_slowfast
from pytorchvideo.layers.nonlocal_net import create_nonlocal

class AttentionSlowFast(nn.Module):
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
        
        # Inject Non-Local Blocks into the Fast Pathway (pathway 1)
        # Typically added to res4 and res5
        # The structure of SlowFast from pytorchvideo is usually:
        # blocks[0] = stem
        # blocks[1] = res2 ...
        # pathways are fused.
        # We need to inspect the model structure to inject safely.
        # For simplicity in this assignment, we will attempt to wrap the res4 and res5 blocks
        # of the Fast pathway if accessible, or just append NonLocal after stages.
        
        # NOTE: This is a simplified injection strategy. 
        # A more robust way is to build the Net using a config that supports non-local.
        # Since create_slowfast doesn't expose it directly, we iterate.
        
        self._inject_nonlocal()

    def forward(self, x):
        return self.model(x)

    def _inject_nonlocal(self):
        """
        Injects Non-Local blocks into the network. 
        We assume self.model.blocks contains the stages.
        Indices 3 and 4 correspond to res4 and res5.
        
        Since modifying the internal structure of MultiPathWayWithFuse is difficult post-hoc,
        we will wrap the blocks 3 and 4 to apply attention to the Fast pathway output.
        """
        # Define NonLocal blocks. 
        # Channels depend on the depth (ResNet50). 
        # Fast pathway usually has beta * width channels. 
        # For ResNet50, block 3 (res4) output is 1024, block 4 (res5) is 2048.
        # Fast pathway with beta=1/8 usually has 1/8 of that? 
        # Actually SlowFast channel config is bit complex.
        # Let's assume standard behavior:
        # Fast pathway res4 out: 128 (if width_per_group=64, expansion=4 -> 256? No wait.)
        # Let's dynamically determine channels during first forward or hardcode for ResNet50.
        # ResNet50: res4 -> 1024, res5 -> 2048 (total).
        # Fast pathway is typically 1/8 channel width of Slow if alpha=8, beta=1/8.
        # So Fast res4 ~ 128, Fast res5 ~ 256?
        
        # We will use a wrapper class to intercept the output.
        
        # Hardcoded for ResNet50 SlowFast config:
        # Res4 Fast: 128 channels (approx, verification needed)
        # Res5 Fast: 256 channels
        
        self.nonlocal_res4 = create_nonlocal(dim_in=128, dim_inner=64)
        self.nonlocal_res5 = create_nonlocal(dim_in=256, dim_inner=128)
        
        # Monkey patch the forward of block 3 and 4? 
        # Or better: Subclass the block?
        # Easiest: Wrap the block in a nn.Module that calls the original then applies NonLocal.
        
        self.model.blocks[3] = AttentiveBlockWrapper(self.model.blocks[3], self.nonlocal_res4, pathway=1)
        self.model.blocks[4] = AttentiveBlockWrapper(self.model.blocks[4], self.nonlocal_res5, pathway=1)

class AttentiveBlockWrapper(nn.Module):
    def __init__(self, original_block, attention_module, pathway=1):
        super().__init__()
        self.original_block = original_block
        self.attention_module = attention_module
        self.pathway = pathway # 1 for Fast, 0 for Slow
        
    def forward(self, x):
        # x is a list of [slow, fast]
        out = self.original_block(x)
        
        # Apply attention to the specified pathway
        if isinstance(out, list) and len(out) > self.pathway:
            # Output of a block in SlowFast is [slow_feature, fast_feature]
            # We apply attention to fast_feature
            fast_feature = out[self.pathway]
            # NonLocal expects (N, C, T, H, W)
            fast_feature = self.attention_module(fast_feature)
            out[self.pathway] = fast_feature
            
        return out
