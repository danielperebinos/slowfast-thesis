import torch
import torch.nn as nn
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    Normalize,
)
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self, slow_frames: int, fast_frames: int, alpha: int = None):
        super().__init__()
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        # default alpha is typically fast // slow, e.g. 32 // 4 = 8
        if alpha is None:
            self.alpha = fast_frames // slow_frames
        else:
            self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        """
        Args:
            frames (torch.Tensor): Video tensor with shape (C, T, H, W).
        Returns:
            list: [slow_pathway, fast_pathway]
        """
        fast_pathway = frames
        # Perform temporal subsampling for the slow pathway
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, self.slow_frames
            ).long(),
        )
        return [slow_pathway, fast_pathway]

def get_train_transform(
    side_size: int = 256,
    crop_size: int = 224,
    mean: list = [0.45, 0.45, 0.45],
    std: list = [0.225, 0.225, 0.225],
    slow_frames: int = 4,
    fast_frames: int = 32,
) -> Compose:
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(fast_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        ShortSideScale(side_size),
                        RandomCrop(crop_size),
                        RandomHorizontalFlip(p=0.5),
                        PackPathway(slow_frames=slow_frames, fast_frames=fast_frames),
                    ]
                ),
            ),
        ]
    )

def get_val_transform(
    side_size: int = 256,
    crop_size: int = 224,
    mean: list = [0.45, 0.45, 0.45],
    std: list = [0.225, 0.225, 0.225],
    slow_frames: int = 4,
    fast_frames: int = 32,
) -> Compose:
    return Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(fast_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        ShortSideScale(side_size),
                        CenterCrop(crop_size),
                        PackPathway(slow_frames=slow_frames, fast_frames=fast_frames),
                    ]
                ),
            ),
        ]
    )
