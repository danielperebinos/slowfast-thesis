import os
import sys

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorchvideo.models.slowfast import create_slowfast

from experiments.core.data_loader import make_ava_dataset
from experiments.core.loss import ActionAnticipationLoss
from experiments.core.metrics import AverageMeter, multilabel_accuracy


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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    acc = AverageMeter("Acc", ":6.2f")

    for i, batch in enumerate(loader):
        inputs = [x.to(device) for x in batch["video"]]   # [slow, fast]
        labels = batch["label"].to(device)                 # (B, num_classes) float
        current_times = batch["clip_end_time"].to(device)  # (B,) AVA seconds

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
                f"Epoch [{epoch}][{i}/{len(loader)}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc {acc.val:.2f} ({acc.avg:.2f})"
            )
            mlflow.log_metric("train_loss_step", losses.val)

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

    print(f" * Val Acc {acc.avg:.3f}")
    return losses.avg, acc.avg


def main():
    EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_DIR = os.path.join(EXPERIMENT_DIR, "data", "videos")
    TRAIN_CSV = os.path.join(EXPERIMENT_DIR, "train.csv")
    VAL_CSV = os.path.join(EXPERIMENT_DIR, "test.csv")

    EPOCHS = 10
    BATCH_SIZE = 4
    LR = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Experiment 01: Baseline SlowFast on {DEVICE}")

    # Build train dataset first to establish the label_map, then share it with val
    # so both splits use identical class indices.
    train_loader, train_dataset = make_ava_dataset(
        csv_path=TRAIN_CSV,
        video_dir=VIDEO_DIR,
        mode="train",
        batch_size=BATCH_SIZE,
        num_workers=2,
    )
    val_loader, val_dataset = make_ava_dataset(
        csv_path=VAL_CSV,
        video_dir=VIDEO_DIR,
        mode="val",
        batch_size=BATCH_SIZE,
        num_workers=2,
        label_map=train_dataset.label_map,  # enforce same class order as train
    )

    num_classes = train_dataset.num_classes
    print(
        f"Action classes: {num_classes} | "
        f"Train samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)}"
    )
    print(f"Label map: {train_dataset.label_map}")

    model = get_model(num_classes=num_classes).to(DEVICE)
    criterion = ActionAnticipationLoss(alpha=0.5, T_start=900.0, T_scale=100.0)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    mlflow.set_experiment("SlowFast_Anticipation")
    with mlflow.start_run(run_name="Experiment_01_Baseline"):
        mlflow.log_params({
            "model": "Baseline_SlowFast_R50",
            "num_classes": num_classes,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "loss_alpha": 0.5,
            "loss_T_start": 900.0,
            "loss_T_scale": 100.0,
        })

        best_val_acc = 0.0
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE, epoch
            )
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
                ckpt = os.path.join(EXPERIMENT_DIR, "best_baseline.pth")
                torch.save(model.state_dict(), ckpt)
                mlflow.log_artifact(ckpt)

    print(f"Done. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
