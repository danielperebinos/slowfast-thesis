"""
Experiment 04 — Hybrid SlowFast R50 (YOLOv8 ROI + Non-Local Attention).

Combines ROI masking on the Fast pathway (from exp03) with Non-Local attention
blocks (from exp02) into a single hybrid model.  The loss adds a configurable
temporal-proximity penalty (LOSS_ALPHA sweep).

Resume: if ``checkpoint.pth`` exists in the experiment directory the
model/optimizer/scheduler states are restored and training continues from the
saved epoch.
"""

import os
import sys

import mlflow
import pandas as pd
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.core.data_loader import build_label_map, make_ava_dataset
from experiments.core.loss import ActionAnticipationLoss
from experiments.core.metrics import AverageMeter, multilabel_accuracy
from experiments.experiment_04.model import HybridSlowFast

# ── configuration ─────────────────────────────────────────────────────────────

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(EXPERIMENT_DIR)

VIDEO_DIR = os.getenv("VIDEO_DIR") or os.path.join(EXPERIMENTS_DIR, "experiment_01", "data", "videos")
TRAIN_CSV = os.getenv("TRAIN_CSV") or os.path.join(EXPERIMENTS_DIR, "experiment_01", "train.csv")
VAL_CSV = os.getenv("VAL_CSV") or os.path.join(EXPERIMENTS_DIR, "experiment_01", "test.csv")
YOLO_MODEL = os.getenv("YOLO_MODEL") or os.path.join(EXPERIMENTS_DIR, "yolov8n.pt")
CKPT_PATH = os.getenv("CKPT_PATH") or os.path.join(EXPERIMENT_DIR, "checkpoint.pth")

EPOCHS = 10
BATCH_SIZE = 4
LR = 0.01
NUM_WORKERS = 0
LOSS_ALPHA = float(os.getenv("LOSS_ALPHA", "0.5"))
ALPHA_BLEND = float(os.getenv("ALPHA_BLEND", "1.0"))

# ── training / validation loops ───────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")

    for i, batch in enumerate(loader):
        inputs = [x.to(device) for x in batch["video"]]
        labels = batch["label"].to(device)
        current_times = batch["clip_end_time"].to(device)

        preds = model(inputs)
        loss = criterion(preds, labels, current_time=current_times)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_acc = multilabel_accuracy(preds, labels)
        losses.update(loss.item(), labels.size(0))
        acc.update(batch_acc.item(), labels.size(0))

        if i % 10 == 0:
            print(
                f"  Epoch {epoch} | step {i}/{len(loader)} "
                f"| loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"| acc {acc.val:.2f} ({acc.avg:.2f})"
            )

    return losses.avg, acc.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")

    with torch.no_grad():
        for batch in loader:
            inputs = [x.to(device) for x in batch["video"]]
            labels = batch["label"].to(device)
            current_times = batch["clip_end_time"].to(device)

            preds = model(inputs)
            loss = criterion(preds, labels, current_time=current_times)

            batch_acc = multilabel_accuracy(preds, labels)
            losses.update(loss.item(), labels.size(0))
            acc.update(batch_acc.item(), labels.size(0))

    print(f"  [val] loss {losses.avg:.4f} | acc {acc.avg:.3f}")
    return losses.avg, acc.avg


# ── checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, label_map):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "label_map": label_map,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from checkpoint: epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt["best_val_acc"], ckpt["label_map"]


# ── entry point ───────────────────────────────────────────────────────────────


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Experiment 04: Hybrid SlowFast R50 | device={DEVICE}")

    label_map = build_label_map(TRAIN_CSV)
    num_classes = len(label_map)
    print(f"Label map: {num_classes} action classes")

    n_train = pd.read_csv(TRAIN_CSV)["video_id"].nunique()
    n_val = pd.read_csv(VAL_CSV)["video_id"].nunique()
    print(f"Train videos: {n_train} | Val videos: {n_val}")

    model = HybridSlowFast(
        num_classes=num_classes,
        yolo_model=YOLO_MODEL,
        alpha_blend=ALPHA_BLEND,
    ).to(DEVICE)

    criterion = ActionAnticipationLoss(alpha=LOSS_ALPHA, T_start=900.0, T_scale=100.0)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch, best_val_acc = 0, 0.0
    if os.path.exists(CKPT_PATH):
        start_epoch, best_val_acc, label_map = load_checkpoint(CKPT_PATH, model, optimizer, scheduler)
        if start_epoch >= EPOCHS:
            print("Training already complete according to checkpoint.")
            return

    train_loader, _ = make_ava_dataset(
        csv_path=TRAIN_CSV,
        video_dir=VIDEO_DIR,
        mode="train",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_map=label_map,
    )
    val_loader, _ = make_ava_dataset(
        csv_path=VAL_CSV,
        video_dir=VIDEO_DIR,
        mode="val",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_map=label_map,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("SlowFast_Anticipation")
    with mlflow.start_run(run_name="Experiment_04_Hybrid"):
        mlflow.log_params(
            {
                "model": "Hybrid_SlowFast_R50",
                "num_classes": num_classes,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "loss_alpha": LOSS_ALPHA,
                "T_start": 900.0,
                "T_scale": 100.0,
                "alpha_blend": ALPHA_BLEND,
                "yolo_model": os.path.basename(YOLO_MODEL),
            }
        )

        for epoch in range(start_epoch, EPOCHS):
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch + 1}/{EPOCHS}")
            print(f"{'=' * 60}")

            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(EXPERIMENT_DIR, "best_hybrid.pth")
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(best_path)
                print(f"  New best val acc: {best_val_acc:.3f} — saved")

            save_checkpoint(
                CKPT_PATH,
                model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                best_val_acc=best_val_acc,
                label_map=label_map,
            )

    print(f"\nTraining complete. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
