import torch
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, slow_frames: int, fast_frames: int, alpha: int = None):
        super().__init__()
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        if alpha is None:
            self.alpha = fast_frames // slow_frames
        else:
            self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, self.slow_frames).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def get_train_transform(
    mean=[0.45, 0.45, 0.45],
    std=[0.225, 0.225, 0.225],
    side_size=256,
    crop_size=224,
    num_frames=32,
    slowfast_alpha=8,
):
    """
    Standard SlowFast Training Transforms.
    """
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                ShortSideScale(side_size),
                RandomCrop(crop_size),
                RandomHorizontalFlip(p=0.5),
                PackPathway(
                    slow_frames=num_frames // slowfast_alpha,
                    fast_frames=num_frames,
                    alpha=slowfast_alpha,
                ),
            ]
        ),
    )


def get_val_transform(
    mean=[0.45, 0.45, 0.45],
    std=[0.225, 0.225, 0.225],
    side_size=256,
    crop_size=224,
    num_frames=32,
    slowfast_alpha=8,
):
    """
    Standard SlowFast Validation Transforms.
    """
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                ShortSideScale(side_size),
                CenterCrop(crop_size),
                PackPathway(
                    slow_frames=num_frames // slowfast_alpha,
                    fast_frames=num_frames,
                    alpha=slowfast_alpha,
                ),
            ]
        ),
    )


def make_dataset(
    csv_file: str,
    mode: str,
    clip_duration: float = 2.0,
    batch_size: int = 8,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    Factory for PyTorch Video DataLoader.
    """
    transform = get_train_transform() if mode == "train" else get_val_transform()

    # Check if clip_sampler is available in pytorchvideo version installed
    # Standard usage:
    dataset = labeled_video_dataset(
        data_path=csv_file,
        clip_sampler=make_clip_sampler("random", clip_duration),
        transform=transform,
        decode_audio=False,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
