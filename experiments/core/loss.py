import torch
import torch.nn as nn


class ActionAnticipationLoss(nn.Module):
    """
    Asymmetric Loss for Action Anticipation:
    L_total = L_cls + alpha * exp((t - T_start) / T_scale)

    This penalizes later decisions more heavily.
    """

    def __init__(self, alpha: float = 1.0, T_start: float = 0.5, T_scale: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.T_start = T_start
        self.T_scale = T_scale
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, labels, current_time=None):
        """
        Args:
            preds: (B, NumClasses) logits
            labels: (B,) class indices
            current_time: (B,) or float, time of decision in range [0, 1] relative to action start.
                          If None, defaults to 0 penalty (standard CE).
        """
        loss_cls = self.ce_loss(preds, labels)

        if self.alpha == 0 or current_time is None:
            return loss_cls

        t = torch.tensor(current_time, device=preds.device) if isinstance(current_time, float) else current_time
        time_factor = torch.exp((t - self.T_start) / self.T_scale)

        # Apply penalty multiplicatively or additively?
        # Requirement: L_total = L_cls + alpha * exp(...)
        return loss_cls + self.alpha * time_factor.mean()
