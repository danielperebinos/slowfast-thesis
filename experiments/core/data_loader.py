import os

import pandas as pd
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class PackPathway(torch.nn.Module):
    """Splits a video tensor into [slow, fast] pathway pair."""

    def __init__(self, slow_frames: int, fast_frames: int, alpha: int = None):
        super().__init__()
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.alpha = alpha if alpha is not None else fast_frames // slow_frames

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, self.slow_frames).long(),
        )
        return [slow_pathway, fast_pathway]


def get_train_transform(
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225),
    side_size=256,
    crop_size=224,
    num_frames=32,
    slowfast_alpha=8,
):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
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
        ]),
    )


def get_val_transform(
    mean=(0.45, 0.45, 0.45),
    std=(0.225, 0.225, 0.225),
    side_size=256,
    crop_size=224,
    num_frames=32,
    slowfast_alpha=8,
):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
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
        ]),
    )


def build_label_map(csv_path: str) -> dict:
    """
    Build a stable action_id → class_index mapping from a CSV file.

    Call this once on the full training CSV before any videos are downloaded
    so every chunk/epoch uses identical class indices.
    """
    df = pd.read_csv(csv_path)
    action_ids = sorted(df["action"].unique().tolist())
    return {aid: i for i, aid in enumerate(action_ids)}


class AvaAnticipationDataset(torch.utils.data.Dataset):
    """
    Dataset for action anticipation from AVA-format CSV annotations.

    For each (video_id, ts) entry we observe a clip that ends `anticipation_gap`
    seconds BEFORE the annotated timestamp `ts`, and predict the multi-hot action
    set occurring AT `ts`.

    CSV columns: video_id, ts, x1, y1, x2, y2, action, person_id
    Video files:  {video_dir}/{video_id}.mp4

    AVA videos were trimmed starting at `time_offset` seconds of the original
    footage (default 900 s), so local file time = AVA timestamp - time_offset.

    Parameters
    ----------
    video_ids : list | None
        If given, only samples from these video IDs are included.
        Used by the streaming trainer to restrict to the currently
        downloaded chunk.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        clip_duration: float = 2.0,
        anticipation_gap: float = 1.0,
        time_offset: float = 900.0,
        transform=None,
        label_map: dict = None,
        video_ids: list = None,
    ):
        self.video_dir = video_dir
        self.clip_duration = clip_duration
        self.anticipation_gap = anticipation_gap
        self.time_offset = time_offset
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Filter to the requested video subset (streaming mode)
        if video_ids is not None:
            df = df[df["video_id"].isin(video_ids)]

        # Build label map: action_id -> class_index (sorted for reproducibility)
        if label_map is None:
            action_ids = sorted(df["action"].unique().tolist())
            self.label_map = {aid: i for i, aid in enumerate(action_ids)}
        else:
            self.label_map = label_map
        self.num_classes = len(self.label_map)

        # One sample per (video_id, ts): pool all persons' actions → multi-hot label
        self.samples = []
        for (vid_id, ts), group in df.groupby(["video_id", "ts"]):
            label = torch.zeros(self.num_classes)
            for aid in group["action"].tolist():
                if aid in self.label_map:
                    label[self.label_map[aid]] = 1.0
            self.samples.append({"video_id": vid_id, "ts": float(ts), "label": label})

        # Verify only the files we actually need are present
        referenced = {s["video_id"] for s in self.samples}
        for vid_id in referenced:
            path = os.path.join(video_dir, f"{vid_id}.mp4")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vid_id = s["video_id"]
        ts = s["ts"]
        label = s["label"]

        # Convert AVA absolute timestamp to local file time, then apply anticipation gap.
        # We observe [clip_start_local, clip_end_local] and predict the action at ts.
        clip_end_local = (ts - self.time_offset) - self.anticipation_gap
        clip_start_local = max(0.0, clip_end_local - self.clip_duration)

        video_path = os.path.join(self.video_dir, f"{vid_id}.mp4")
        video = EncodedVideo.from_path(video_path, decode_audio=False)
        video_data = video.get_clip(clip_start_local, clip_end_local)
        frames = video_data["video"]  # (C, T, H, W)

        if self.transform is not None:
            frames = self.transform({"video": frames})["video"]
            # frames is now [slow_pathway, fast_pathway] after PackPathway

        # Pass AVA absolute clip_end_time to the temporal loss (T_start is in AVA seconds)
        clip_end_time = torch.tensor(ts - self.anticipation_gap, dtype=torch.float32)
        return frames, label, clip_end_time


def _slowfast_collate(batch):
    """Collate [slow, fast] pathway samples into a batch dict."""
    slow = torch.stack([item[0][0] for item in batch])   # (B, C, T_slow, H, W)
    fast = torch.stack([item[0][1] for item in batch])   # (B, C, T_fast, H, W)
    labels = torch.stack([item[1] for item in batch])    # (B, num_classes)
    times = torch.stack([item[2] for item in batch])     # (B,)
    return {"video": [slow, fast], "label": labels, "clip_end_time": times}


def make_ava_dataset(
    csv_path: str,
    video_dir: str,
    mode: str,
    clip_duration: float = 2.0,
    anticipation_gap: float = 1.0,
    time_offset: float = 900.0,
    batch_size: int = 4,
    num_workers: int = 2,
    label_map: dict = None,
    video_ids: list = None,
) -> tuple:
    """
    Factory returning (DataLoader, AvaAnticipationDataset).

    Always build the train split first (no label_map), then pass
    train_dataset.label_map to the val/test call so both splits share
    the same class indices.
    """
    transform = get_train_transform() if mode == "train" else get_val_transform()
    dataset = AvaAnticipationDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        clip_duration=clip_duration,
        anticipation_gap=anticipation_gap,
        time_offset=time_offset,
        transform=transform,
        label_map=label_map,
        video_ids=video_ids,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_slowfast_collate,
    )
    return loader, dataset
