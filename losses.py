import torch
import torch.nn as nn
import torch.nn.functional as F

"""Define the loss functions for the model
"""


class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten the tensors to make the calculation easier
        preds_flat = torch.mean(preds, dim=1).view(-1)
        targets_flat = torch.mean(targets, dim=1).view(-1)

        # Calculate intersection and union areas
        intersection = torch.sum(preds_flat * targets_flat)
        total = torch.sum(preds_flat + targets_flat)
        union = total - intersection

        # Compute the IoU score
        IoU = (intersection + self.smooth) / (union + self.smooth)

        # Return the IoU loss
        return - IoU  # Subtracting to make it a loss (lower is better)
    


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, mean=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mean = mean

    def forward(self, input, target):
        gray_input = torch.mean(input, dim=1, keepdim=True)
        p_pred = torch.sigmoid(gray_input)
        target_squeezed = target  # Squeeze the singleton dimension
        ce_loss = F.binary_cross_entropy_with_logits(
            p_pred, target_squeezed, reduction="none")
        p_t = p_pred * target_squeezed + \
            (1 - p_pred) * (1 - target_squeezed)  # p_t = p if y = 1 else 1-p
        # log(p)(1 - p) ** gamma  if y = 1 else log(1-p)p ** gamma
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target_squeezed + \
                (1 - self.alpha) * \
                (1 - target_squeezed)  # alpha if y = 1 else 1-alpha
            loss = alpha_t * loss

        if self.mean:
            return loss.mean()
        else:
            return loss.sum()
