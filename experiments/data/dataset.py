import torch
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset
from .transforms import get_train_transform, get_val_transform

def make_dataset(
    csv_file: str,
    root_path: str,
    mode: str,
    clip_duration: float = 2.0,
    batch_size: int = 8,
) -> torch.utils.data.DataLoader:
    """
    Factory function to create the DataLoader.
    """
    transform = get_train_transform() if mode == "train" else get_val_transform()
    
    # Check if we need a custom collate_fn or if pytorchvideo handles it.
    # LabeledVideoDataset returns a dictionary.
    
    dataset = labeled_video_dataset(
        data_path=csv_file,
        clip_sampler=make_clip_sampler("random", clip_duration),
        transform=transform,
        decode_audio=False,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )
