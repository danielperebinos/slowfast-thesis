import torch
import torch.nn as nn
from experiments.models.baseline import BaselineSlowFast
from experiments.models.attention import AttentionSlowFast
from experiments.models.roi_guidance import RoiGuidanceSlowFast
from experiments.models.hybrid import HybridSlowFast
from experiments.loss.anticipation_loss import ActionAnticipationLoss
from experiments.utils.metrics import topk_accuracy
import gc

def test_on_dummy_data():
    """
    Runs a single forward and backward pass on random data for each model.
    """
    print("Running verification...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Prepare dummy data
    # Reduced size to avoid OOM on verification
    B, C, H, W = 1, 3, 160, 160
    slow_input = torch.randn(B, C, 4, H, W).to(device)
    fast_input = torch.randn(B, C, 32, H, W).to(device)
    labels = torch.zeros(B).long().to(device)
    
    criterion = ActionAnticipationLoss().to(device)
    
    models_to_test = [
        ("baseline", BaselineSlowFast),
        ("attention", AttentionSlowFast),
        ("roi_guidance", RoiGuidanceSlowFast),
        ("hybrid", HybridSlowFast),
    ]

    for name, model_cls in models_to_test:
        print(f"Testing {name}...")
        try:
            model = model_cls(num_classes=400).to(device)
            # IMPORTANT: Pass a list copy because pytorchvideo models modify the list in-place
            output = model([slow_input, fast_input])
            
            # Loss
            loss = criterion(output, labels)
            loss.backward()
            
            # Metric
            acc1, acc5 = topk_accuracy(output, labels, topk=(1, 5))
            print(f"  [PASS] {name}: Forward pass successful. Output shape: {output.shape}, Acc@1: {acc1.item():.2f}, Acc@5: {acc5.item():.2f}")
            
            # Cleanup to prevent OOM
            del model
            del output
            del loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            raise e

    print("All models verified successfully!")

if __name__ == "__main__":
    test_on_dummy_data()
