import torch
from .train import get_model, validate
from .data.dataset import make_dataset
from .loss.anticipation_loss import ActionAnticipationLoss # Reuse loss for metric calculation if needed
import os
import mlflow

def run_evaluation(
    variant: str,
    data_path: str,
    checkpoint_path: str,
    num_classes: int,
    batch_size: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Starting evaluation for {variant} on {device}")
    
    # Load Model
    model = get_model(variant, num_classes).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Evaluating initialized model.")

    # Load Data
    val_csv = os.path.join(data_path, "val.csv")
    val_loader = make_dataset(val_csv, data_path, "val", batch_size=batch_size)
    
    # Criterion
    criterion = ActionAnticipationLoss(alpha=1.0).to(device)
    
    acc1, loss = validate(val_loader, model, criterion, device)
    
    print(f"Validation Result: Acc@1 {acc1:.3f} Loss {loss:.4f}")
    return acc1, loss
