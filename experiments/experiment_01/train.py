"""
Experiment 01 — Baseline SlowFast R50 with streaming data.

Streaming training loop
-----------------------
The full AVA dataset is too large to keep on disk at once.
This script processes it in video-ID chunks:

  for each epoch:
      for each chunk of CHUNK_SIZE videos:
          1. download_batch  — fetch only those .mp4 files
          2. train           — one pass over the chunk's samples
          3. delete_batch    — free disk space (pre-existing files are never deleted)
      validate on test set (assumed pre-downloaded)
      checkpoint

Resume: if a ``checkpoint.pth`` exists in the experiment directory
the model/optimizer/scheduler states are restored and training
continues from the saved epoch + chunk.
"""

import json
import os
import sys

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorchvideo.models.slowfast import create_slowfast

from experiments.core.data_loader import build_label_map, make_ava_dataset
from experiments.core.loss import ActionAnticipationLoss
from experiments.core.metrics import AverageMeter, multilabel_accuracy
from experiments.dataset.fetch import delete_batch, download_batch

# ── configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(EXPERIMENT_DIR, "data", "videos")
TRAIN_CSV  = os.path.join(EXPERIMENT_DIR, "train.csv")
VAL_CSV    = os.path.join(EXPERIMENT_DIR, "test.csv")
CKPT_PATH  = os.path.join(EXPERIMENT_DIR, "checkpoint.pth")

EPOCHS      = 10
BATCH_SIZE  = 4
LR          = 0.01
CHUNK_SIZE  = 5   # number of videos downloaded and trained per iteration
NUM_WORKERS = 2

# ── model ─────────────────────────────────────────────────────────────────────

def get_model(num_classes: int):
    """Experiment 01: Baseline SlowFast R50."""
    return create_slowfast(
        model_depth=50,
        model_num_class=num_classes,
        slowfast_channel_reduction_ratio=(8,),
        slowfast_fusion_conv_stride=(8, 1, 1),
        input_channels=(3, 3),
        head_pool=nn.AdaptiveAvgPool3d,
    )


# ── training / validation loops ───────────────────────────────────────────────

def train_chunk(model, loader, criterion, optimizer, device, epoch, chunk_idx):
    """One training pass over a single video chunk."""
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    acc    = AverageMeter("Acc",  ":6.2f")

    for i, batch in enumerate(loader):
        inputs       = [x.to(device) for x in batch["video"]]
        labels       = batch["label"].to(device)
        current_times = batch["clip_end_time"].to(device)

        preds = model(inputs)
        loss  = criterion(preds, labels, current_time=current_times)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = multilabel_accuracy(preds, labels)
        losses.update(loss.item(), labels.size(0))
        acc.update(batch_acc.item(), labels.size(0))

        if i % 10 == 0:
            print(
                f"  Epoch {epoch} | chunk {chunk_idx} | step {i}/{len(loader)} "
                f"| loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"| acc {acc.val:.2f} ({acc.avg:.2f})"
            )

    return losses.avg, acc.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    acc    = AverageMeter("Acc",  ":6.2f")

    with torch.no_grad():
        for batch in loader:
            inputs        = [x.to(device) for x in batch["video"]]
            labels        = batch["label"].to(device)
            current_times = batch["clip_end_time"].to(device)

            preds = model(inputs)
            loss  = criterion(preds, labels, current_time=current_times)

            batch_acc = multilabel_accuracy(preds, labels)
            losses.update(loss.item(), labels.size(0))
            acc.update(batch_acc.item(), labels.size(0))

    print(f"  [val] loss {losses.avg:.4f} | acc {acc.avg:.3f}")
    return losses.avg, acc.avg


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch, chunk_idx, best_val_acc, label_map):
    torch.save(
        {
            "epoch":        epoch,
            "chunk_idx":    chunk_idx,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "label_map":    label_map,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    """Load checkpoint. Returns (start_epoch, start_chunk, best_val_acc, label_map)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from checkpoint: epoch {ckpt['epoch']}, chunk {ckpt['chunk_idx']}")
    return ckpt["epoch"], ckpt["chunk_idx"], ckpt["best_val_acc"], ckpt["label_map"]


# ── streaming training orchestrator ──────────────────────────────────────────

def streaming_train(
    model, all_video_ids, label_map, val_loader,
    criterion, optimizer, scheduler,
    device, epochs, start_epoch, start_chunk, best_val_acc,
):
    """
    Main streaming loop.

    Iterates over epoch → chunk → download → train → delete.
    Saves a checkpoint after every chunk so training can be safely interrupted.
    """
    num_classes = len(label_map)

    # Split all training video IDs into chunks
    chunks = [
        all_video_ids[i : i + CHUNK_SIZE]
        for i in range(0, len(all_video_ids), CHUNK_SIZE)
    ]
    print(f"Training on {len(all_video_ids)} videos in {len(chunks)} chunk(s) of ≤{CHUNK_SIZE}")

    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        epoch_losses = AverageMeter("EpochLoss", ":.4e")
        epoch_acc    = AverageMeter("EpochAcc",  ":6.2f")

        for chunk_idx, chunk_ids in enumerate(chunks):

            # Skip chunks already processed in a resumed epoch
            if epoch == start_epoch and chunk_idx < start_chunk:
                print(f"  chunk {chunk_idx}: already done (resume) — skipping")
                continue

            print(f"\n  --- chunk {chunk_idx + 1}/{len(chunks)}: {chunk_ids} ---")

            # ── 1. download ────────────────────────────────────────────────
            available, to_delete = download_batch(chunk_ids, VIDEO_DIR)

            if not available:
                print(f"  No videos available for chunk {chunk_idx} — skipping")
                continue

            print(f"  Ready: {available} | will delete after: {to_delete}")

            # ── 2. build chunk DataLoader ──────────────────────────────────
            try:
                chunk_loader, _ = make_ava_dataset(
                    csv_path=TRAIN_CSV,
                    video_dir=VIDEO_DIR,
                    mode="train",
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    label_map=label_map,
                    video_ids=available,
                )
            except Exception as e:
                print(f"  Dataset creation failed: {e} — skipping chunk")
                delete_batch(to_delete, VIDEO_DIR)
                continue

            if len(chunk_loader.dataset) == 0:
                print(f"  Chunk has 0 samples — skipping")
                delete_batch(to_delete, VIDEO_DIR)
                continue

            # ── 3. train on this chunk ─────────────────────────────────────
            chunk_loss, chunk_acc = train_chunk(
                model, chunk_loader, criterion, optimizer, device, epoch, chunk_idx
            )
            epoch_losses.update(chunk_loss, len(chunk_loader.dataset))
            epoch_acc.update(chunk_acc, len(chunk_loader.dataset))

            mlflow.log_metrics(
                {"chunk_train_loss": chunk_loss, "chunk_train_acc": chunk_acc},
                step=epoch * len(chunks) + chunk_idx,
            )

            # ── 4. checkpoint after each chunk (safe resume point) ─────────
            save_checkpoint(
                CKPT_PATH, model, optimizer, scheduler,
                epoch=epoch, chunk_idx=chunk_idx + 1,
                best_val_acc=best_val_acc, label_map=label_map,
            )

            # ── 5. delete newly downloaded videos ─────────────────────────
            delete_batch(to_delete, VIDEO_DIR)

        # ── end of epoch: validate ─────────────────────────────────────────
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        mlflow.log_metrics(
            {
                "train_loss": epoch_losses.avg,
                "train_acc":  epoch_acc.avg,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            },
            step=epoch,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(EXPERIMENT_DIR, "best_baseline.pth")
            torch.save(model.state_dict(), best_path)
            mlflow.log_artifact(best_path)
            print(f"  New best val acc: {best_val_acc:.3f} — saved")

        # After completing an epoch fully, reset chunk counter for next epoch
        save_checkpoint(
            CKPT_PATH, model, optimizer, scheduler,
            epoch=epoch + 1, chunk_idx=0,
            best_val_acc=best_val_acc, label_map=label_map,
        )

        # Next epoch always starts from chunk 0
        start_chunk = 0

    return best_val_acc


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Experiment 01: Baseline SlowFast R50 | device={DEVICE}")

    # ── label map — built from the full CSV, no videos needed ──────────────
    label_map  = build_label_map(TRAIN_CSV)
    num_classes = len(label_map)
    print(f"Label map: {num_classes} action classes")

    # ── all training video IDs (order is preserved across epochs) ──────────
    import pandas as pd
    all_video_ids = pd.read_csv(TRAIN_CSV)["video_id"].unique().tolist()

    # ── validation loader (test videos assumed pre-downloaded) ─────────────
    val_loader, _ = make_ava_dataset(
        csv_path=VAL_CSV,
        video_dir=VIDEO_DIR,
        mode="val",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_map=label_map,
    )

    # ── model / optimiser / scheduler ──────────────────────────────────────
    model     = get_model(num_classes=num_classes).to(DEVICE)
    criterion = ActionAnticipationLoss(alpha=0.5, T_start=900.0, T_scale=100.0)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── resume from checkpoint if available ────────────────────────────────
    start_epoch, start_chunk, best_val_acc = 0, 0, 0.0
    if os.path.exists(CKPT_PATH):
        start_epoch, start_chunk, best_val_acc, label_map = load_checkpoint(
            CKPT_PATH, model, optimizer, scheduler
        )
        if start_epoch >= EPOCHS:
            print("Training already complete according to checkpoint.")
            return

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow.set_experiment("SlowFast_Anticipation")
    with mlflow.start_run(run_name="Experiment_01_Baseline"):
        mlflow.log_params(
            {
                "model":       "Baseline_SlowFast_R50",
                "num_classes": num_classes,
                "epochs":      EPOCHS,
                "batch_size":  BATCH_SIZE,
                "lr":          LR,
                "chunk_size":  CHUNK_SIZE,
                "loss_alpha":  0.5,
                "T_start":     900.0,
                "T_scale":     100.0,
            }
        )

        best_val_acc = streaming_train(
            model=model,
            all_video_ids=all_video_ids,
            label_map=label_map,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            start_epoch=start_epoch,
            start_chunk=start_chunk,
            best_val_acc=best_val_acc,
        )

    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
