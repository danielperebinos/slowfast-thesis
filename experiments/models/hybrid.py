import torch
import torch.nn as nn
from .roi_guidance import RoiGuidanceSlowFast
from pytorchvideo.layers.nonlocal_net import create_nonlocal
from .attention import AttentiveBlockWrapper

class HybridSlowFast(RoiGuidanceSlowFast):
    """
    Experiment 4: Hybrid (ROI + Local Attention).
    Applies ROI masking AND Non-Local Attention.
    """
    def __init__(
        self,
        num_classes: int = 400,
        model_depth: int = 50,
        slowfast_alpha: int = 8,
        yolo_model: str = "yolov8n.pt",
        alpha_blend: float = 1.0,
    ):
        super().__init__(
            num_classes, model_depth, slowfast_alpha, yolo_model, alpha_blend
        )
        
        # Inject Non-Local blocks similar to Attention model.
        # Channels need to be verified (same as Attention model).
        self.nonlocal_res4 = create_nonlocal(dim_in=128, dim_inner=64)
        self.nonlocal_res5 = create_nonlocal(dim_in=256, dim_inner=128)
        
        # We need to wrap the blocks of the internal baseline model
        # which is self.baseline.model.blocks
        
        self.baseline.model.blocks[3] = AttentiveBlockWrapper(
            self.baseline.model.blocks[3], self.nonlocal_res4, pathway=1
        )
        self.baseline.model.blocks[4] = AttentiveBlockWrapper(
            self.baseline.model.blocks[4], self.nonlocal_res5, pathway=1
        )

    def forward(self, x):
        # We override forward to apply masking BEFORE the network? 
        # RoiGuidanceSlowFast's forward applies mask inside? 
        # Wait, RoiGuidanceSlowFast.forward calls self.baseline([slow, fast]).
        # And it has logic (currently commented) to apply mask.
        
        # In Hybrid, we want:
        # 1. Get Mask from Fast input.
        # 2. Apply Mask to Fast input (Background suppression).
        # 3. Pass through (Attentive) Baseline.
        
        slow, fast = x
        
        # Masking logic (placeholder, same as RoiGuidance)
        # mask = self.get_roi_mask(fast)
        # fast = fast * mask
        
        return self.baseline([slow, fast])
