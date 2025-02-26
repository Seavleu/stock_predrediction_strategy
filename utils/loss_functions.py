import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        return self.mse(predictions, targets)


class AggressiveDirectionLoss(nn.Module):
    """Loss function for directional prediction & price accuracy"""
    def __init__(self, min_change_threshold=0.001):
        super().__init__()
        self.min_change_threshold = min_change_threshold
        self.epsilon = 1e-7

    def forward(self, predictions, targets, current_prices):
        pred_change = (predictions - current_prices)
        actual_change = (targets - current_prices)

        # Penalize weak signals
        inertia_penalty = torch.exp(-torch.abs(pred_change) / self.min_change_threshold)

        # Direction accuracy loss
        direction_match = torch.sign(pred_change) * torch.sign(actual_change)
        direction_loss = -torch.log((direction_match + 1) / 2 + self.epsilon)

        # Magnitude loss
        magnitude_loss = torch.abs(
            torch.abs(pred_change) - torch.abs(actual_change)
        ) / (torch.abs(actual_change) + self.epsilon)

        total_loss = (
            direction_loss * 0.4 +
            magnitude_loss * 0.3 +
            inertia_penalty * 0.3
        )

        return torch.mean(total_loss)

