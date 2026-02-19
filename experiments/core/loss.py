import torch
import torch.nn as nn


class ActionAnticipationLoss(nn.Module):
    """
    Multi-label anticipation loss:
    L_total = L_bce + alpha * exp((t - T_start) / T_scale)

    L_bce handles multi-hot targets (one sample can have multiple action classes).
    The temporal term penalizes late predictions: the closer t is to T_start (and beyond),
    the higher the penalty, encouraging the model to predict earlier.

    Args:
        alpha:   weight of the temporal penalty term (0 disables it)
        T_start: AVA absolute timestamp where penalty begins to grow (seconds)
        T_scale: controls how quickly the penalty grows with time (seconds)
    """

    def __init__(self, alpha: float = 1.0, T_start: float = 900.0, T_scale: float = 100.0):
        super().__init__()
        self.alpha = alpha
        self.T_start = T_start
        self.T_scale = T_scale
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, labels, current_time=None):
        """
        Args:
            preds:        (B, num_classes) raw logits
            labels:       (B, num_classes) multi-hot float targets
            current_time: (B,) or scalar, AVA absolute time of the clip end (seconds).
                          None disables the temporal penalty.
        """
        loss_cls = self.bce_loss(preds, labels)

        if self.alpha == 0 or current_time is None:
            return loss_cls

        t = (
            torch.tensor(current_time, dtype=torch.float32, device=preds.device)
            if isinstance(current_time, float)
            else current_time.float()
        )
        time_factor = torch.exp((t - self.T_start) / self.T_scale)
        return loss_cls + self.alpha * time_factor.mean()
