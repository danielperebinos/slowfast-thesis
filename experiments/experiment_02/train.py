import os
import sys

import mlflow
import torch
import torch.optim as optim

# Path Hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.core.data_loader import make_dataset
from experiments.core.loss import ActionAnticipationLoss
from experiments.core.metrics import AverageMeter, topk_accuracy
from experiments.experiment_02.model import AttentionSlowFast


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    for i, batch in enumerate(loader):
        inputs = batch["video"]
        inputs = [x.to(device) for x in inputs]
        labels = batch["label"].to(device)

        preds = model(inputs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, _ = topk_accuracy(preds, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(acc1.item(), labels.size(0))

        if i % 10 == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}] Loss {losses.val:.4f} Acc@1 {top1.val:.3f}")
            mlflow.log_metric("train_loss_step", losses.val)

    return losses.avg, top1.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["video"]
            inputs = [x.to(device) for x in inputs]
            labels = batch["label"].to(device)

            preds = model(inputs)
            loss = criterion(preds, labels)

            acc1, _ = topk_accuracy(preds, labels, topk=(1, 5))
            losses.update(loss.item(), labels.size(0))
            top1.update(acc1.item(), labels.size(0))

    print(f" * Acc@1 {top1.avg:.3f}")
    return losses.avg, top1.avg


def main():
    DATA_PATH = os.getenv("DATA_PATH", "train.csv")
    VAL_PATH = os.getenv("VAL_PATH", "val.csv")
    EPOCHS = 10
    BATCH_SIZE = 4  # Reduced due to Non-Local memory overhead
    LR = 0.05
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Experiment 02: Attention SlowFast on {DEVICE}")

    mlflow.set_experiment("SlowFast_Anticipation")

    with mlflow.start_run(run_name="Experiment_02_Attention"):
        mlflow.log_params({"model": "Attention_SlowFast", "lr": LR})

        if not os.path.exists(DATA_PATH):
            print(f"WARNING: Data path {DATA_PATH} not found.")
            pass

        model = AttentionSlowFast(num_classes=400).to(DEVICE)

        criterion = ActionAnticipationLoss(alpha=0.0)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_acc = 0.0
        try:
            train_loader = make_dataset(DATA_PATH, "train", batch_size=BATCH_SIZE)
            val_loader = make_dataset(VAL_PATH, "val", batch_size=BATCH_SIZE)

            for epoch in range(EPOCHS):
                train_loss, train_acc1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
                val_loss, val_acc1 = validate(model, val_loader, criterion, DEVICE)
                scheduler.step()

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc1": val_acc1,
                    },
                    step=epoch,
                )

                if val_acc1 > best_acc:
                    best_acc = val_acc1
                    torch.save(model.state_dict(), "best_attention.pth")
                    mlflow.log_artifact("best_attention.pth")

        except Exception as e:
            print(f"Training interrupted: {e}")


if __name__ == "__main__":
    main()
