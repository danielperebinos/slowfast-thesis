import os
import sys

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path to access core/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorchvideo.models.slowfast import create_slowfast

from experiments.core.data_loader import make_dataset
from experiments.core.loss import ActionAnticipationLoss
from experiments.core.metrics import AverageMeter, topk_accuracy


def get_model(num_classes=400):
    """
    Experiment 01: Baseline SlowFast
    """
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
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    for i, batch in enumerate(loader):
        # inputs: [slow, fast], label is index
        inputs = batch["video"]
        # Move list of tensors to device
        inputs = [x.to(device) for x in inputs]
        labels = batch["label"].to(device)

        # Forward
        preds = model(inputs)
        loss = criterion(preds, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        acc1, acc5 = topk_accuracy(preds, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(acc1.item(), labels.size(0))
        top5.update(acc5.item(), labels.size(0))

        if i % 10 == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(loader)}] "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )
            mlflow.log_metric("train_loss_step", losses.val)

    return losses.avg, top1.avg, top5.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["video"]
            inputs = [x.to(device) for x in inputs]
            labels = batch["label"].to(device)

            preds = model(inputs)
            loss = criterion(preds, labels)

            acc1, acc5 = topk_accuracy(preds, labels, topk=(1, 5))
            losses.update(loss.item(), labels.size(0))
            top1.update(acc1.item(), labels.size(0))
            top5.update(acc5.item(), labels.size(0))

    print(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")
    return losses.avg, top1.avg, top5.avg


def main():
    # Configuration (Simplistic)
    DATA_PATH = os.getenv("DATA_PATH", "train.csv")  # Expects user to set env or hardcode
    VAL_PATH = os.getenv("VAL_PATH", "val.csv")
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Experiment 01: Baseline SlowFast on {DEVICE}")

    mlflow.set_experiment("SlowFast_Anticipation")

    with mlflow.start_run(run_name="Experiment_01_Baseline"):
        mlflow.log_params(
            {
                "model": "Baseline_SlowFast_R50",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
            }
        )

        # Data
        # Assume CSVs exist for now, or user handles it.
        # If not exist, this will crash, which is expected behavior for "fail fast" if data missing.
        if not os.path.exists(DATA_PATH):
            print(f"WARNING: Data path {DATA_PATH} not found. Running in DRY RUN mode with random data?")
            # For the sake of "verification" we might barely skip training loop or fail.
            # Let's fail to prompt user for data.
            pass

        # Model
        model = get_model().to(DEVICE)

        # Loss & Optimizer
        criterion = ActionAnticipationLoss(alpha=0)  # Baseline usually just CE
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Loop
        best_acc = 0.0

        try:
            train_loader = make_dataset(DATA_PATH, "train", batch_size=BATCH_SIZE)
            val_loader = make_dataset(VAL_PATH, "val", batch_size=BATCH_SIZE)

            for epoch in range(EPOCHS):
                train_loss, train_acc1, train_acc5 = train_one_epoch(
                    model, train_loader, criterion, optimizer, DEVICE, epoch
                )

                val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, DEVICE)

                scheduler.step()

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_acc1": train_acc1,
                        "val_loss": val_loss,
                        "val_acc1": val_acc1,
                        "val_acc5": val_acc5,
                    },
                    step=epoch,
                )

                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    torch.save(model.state_dict(), "best_baseline.pth")
                    mlflow.log_artifact("best_baseline.pth")

        except Exception as e:
            print(f"Training interrupted or data missing: {e}")


if __name__ == "__main__":
    main()
