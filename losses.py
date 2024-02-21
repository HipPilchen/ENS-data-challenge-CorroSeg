import torch
import torch.nn as nn
import os 
import pandas as pd
import numpy as np
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Flatten the tensors to make the calculation easier
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union areas
        intersection = torch.sum(preds_flat * targets_flat)
        total = torch.sum(preds_flat + targets_flat)
        union = total - intersection

        # Compute the IoU score
        IoU = (intersection + self.smooth) / (union + self.smooth)

        # Return the IoU loss
        return 1 - IoU  # Subtracting from 1 to make it a loss (lower is better)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, mean=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.mean = mean

    def forward(self, input, target):
        gray_input = input[:,0,:,:]
        p_pred = torch.sigmoid(gray_input)
        target_squeezed = target.squeeze(1)  # Squeeze the singleton dimension
        ce_loss = F.binary_cross_entropy_with_logits(p_pred, target_squeezed, reduction="none")
        p_t = p_pred * target_squeezed + (1 - p_pred) * (1 - target_squeezed) # p_t = p if y = 1 else 1-p
        loss = ce_loss * ((1 - p_t) ** self.gamma) # log(p)(1 - p) ** gamma  if y = 1 else log(1-p)p ** gamma  
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target_squeezed + (1 - self.alpha) * (1 - target_squeezed) # alpha if y = 1 else 1-alpha
            loss = alpha_t * loss
        
        if self.mean: return loss.mean()
        else: return loss.sum()