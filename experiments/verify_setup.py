import gc

import torch
import torch.nn as nn
from core.loss import ActionAnticipationLoss
from core.metrics import topk_accuracy
from experiment_01.train import get_model as get_baseline
from experiment_02.model import AttentionSlowFast
from experiment_03.model import RoiGuidanceSlowFast
from experiment_04.model import HybridSlowFast


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

    # We need to wrap get_baseline to be a class-like or return model
    class BaselineWrapper(nn.Module):
        def __init__(self, num_classes=400):
            super().__init__()
            self.model = get_baseline(num_classes)

        def forward(self, x):
            return self.model(x)

    models_to_test = [
        ("baseline", BaselineWrapper),
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
            print(
                f"[PASS] {name}: Forward pass successful. "
                f"Loss: {loss.item():.4f}, "
                f"Output shape: {output.shape}, "
                f"Acc@1: {acc1.item():.2f}, "
                f"Acc@5: {acc5.item():.2f}"
            )

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
