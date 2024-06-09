from torch import nn
import torch


class QuantileLoss(nn.Module):
    def __init__(self, quantile) -> None:
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        q = self.quantile
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch. Received preds with shape {preds.shape} and targets with shape {targets.shape}")
        error = targets - preds
        loss = torch.max(q * error, (q - 1) * error)
        return torch.mean(loss)


class ApproximatedQuantileLoss(nn.Module):
    """
    Implementation of Approximated Quantile loss as suggested in:

    "Very-Short-Term Probabilistic Forecasting for a Risk-Aware Participation in the 
    Single Price Imbalance Settlement"  by J. Bottieau, L. Hubert, Z. De Grève, F. Vallée and J. -F. Toubeau
    """

    def __init__(self, quantile, treshold=10 ** -6):
        super().__init__()
        self.quantile = quantile
        self.treshold = treshold

    def forward(self, preds, targets):
        q = self.quantile
        if preds.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch. Received preds with shape {preds.shape} and targets with shape {targets.shape}")
        hub = self.hubert(preds, targets, self.treshold)
        loss = torch.where(
            preds < targets,  # Condition.
            q * hub,  # If condition True.
            (1 - q) * hub  # Else.
        )
        return torch.mean(loss)

    def hubert(self, preds, targets, treshold):
        abs_error = torch.abs(preds - targets)
        return torch.where(
            abs_error <= treshold,
            torch.square(preds - targets) / (2 * treshold),
            abs_error - (treshold / 2)
        )
