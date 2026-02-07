import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import time
import os
from .models.baseline import BaselineSlowFast
from .models.attention import AttentionSlowFast
from .models.roi_guidance import RoiGuidanceSlowFast
from .models.hybrid import HybridSlowFast
from .loss.anticipation_loss import ActionAnticipationLoss
from .utils.metrics import topk_accuracy, AverageMeter
from .data.dataset import make_dataset

def get_model(variant, num_classes):
    if variant == "baseline":
        return BaselineSlowFast(num_classes=num_classes)
    elif variant == "attention":
        return AttentionSlowFast(num_classes=num_classes)
    elif variant == "roi_guidance":
        return RoiGuidanceSlowFast(num_classes=num_classes)
    elif variant == "hybrid":
        return HybridSlowFast(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown variant {variant}")

def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, log_interval=10):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    end = time.time()
    
    for i, batch in enumerate(train_loader):
        # inputs is a dictionary in pytorchvideo LabeledVideoDataset
        # keys: 'video', 'label', 'video_name', 'video_index', 'clip_index', 'aug_index'
        inputs = batch['video']
        target = batch['label'] # check if this needs squeeze
        
        # Inputs should be a list [slow, fast] due to PackPathway transform
        # PackPathway returns a list. Pytorch default collate converts list of lists to list of batches?
        # Actually PackPathway returns [slow, fast]. Collate will make it list of [B, C, T, H, W]
        
        # Verify PackPathway output structure vs DataLoader collation.
        # If PackPathway returns list, default_collate(list_of_lists) -> list_of_tensors.
        # So inputs should be [slow_tensor, fast_tensor].
        
        inputs = [inp.to(device) for inp in inputs]
        target = target.to(device)
        
        output = model(inputs)
        loss = criterion(output, target, current_time=None) # Add time info if available
        
        acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), target.size(0))
        top1.update(acc1.item(), target.size(0))
        top5.update(acc5.item(), target.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % log_interval == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
            
            mlflow.log_metric("train_loss_step", losses.val, step=epoch * len(train_loader) + i)
            mlflow.log_metric("train_acc1_step", top1.val, step=epoch * len(train_loader) + i)

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['video']
            target = batch['label']
            
            inputs = [inp.to(device) for inp in inputs]
            target = target.to(device)
            
            output = model(inputs)
            loss = criterion(output, target)
            
            acc1, acc5 = topk_accuracy(output, target, topk=(1, 5))
            
            losses.update(loss.item(), target.size(0))
            top1.update(acc1.item(), target.size(0))
            top5.update(acc5.item(), target.size(0))
            
    return top1.avg, losses.avg

def run_training(
    variant: str,
    data_path: str, # path to csv
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mlflow_experiment: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Starting training for {variant} on {device}")
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(mlflow_experiment)
    
    with mlflow.start_run() as run:
        mlflow.log_params({
            "variant": variant,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device
        })
        
        # Load Data
        # Assume data_path is a directory containing train.csv and val.csv
        train_csv = os.path.join(data_path, "train.csv")
        val_csv = os.path.join(data_path, "val.csv")
        
        train_loader = make_dataset(train_csv, data_path, "train", batch_size=batch_size)
        val_loader = make_dataset(val_csv, data_path, "val", batch_size=batch_size)
        
        # Model
        model = get_model(variant, num_classes).to(device)
        
        # Criterion
        criterion = ActionAnticipationLoss(alpha=1.0).to(device) # alpha configurable?
        
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) # AdamW optional
        
        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_acc, train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
            val_acc, val_loss = validate(val_loader, model, criterion, device)
            
            scheduler.step()
            
            print(f"Epoch {epoch} Result: Train Acc {train_acc:.3f} Val Acc {val_acc:.3f}")
            
            mlflow.log_metric("train_acc_epoch", train_acc, step=epoch)
            mlflow.log_metric("val_acc_epoch", val_acc, step=epoch)
            mlflow.log_metric("val_loss_epoch", val_loss, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)
            
            if val_acc > best_acc:
                best_acc = val_acc
                # Save checkpoint
                checkpoint_path = f"checkpoint_{variant}_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)

    print("Training Complete.")
