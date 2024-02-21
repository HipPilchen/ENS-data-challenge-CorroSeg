import torch
import torch.nn as nn
import os 
import pandas as pd
import numpy as np

def iou_score(preds, targets):
    """
    Computes the Intersection over Union (IoU) score between predicted and target masks.

    Args:
        preds (torch.Tensor): Predicted binary masks, tensor of shape (N, 1, H, W),
                              where N is the batch size, and H, W are the dimensions of the mask.
        targets (torch.Tensor): Ground truth binary masks, tensor of shape (N, 1, H, W).

    Returns:
        torch.Tensor: The IoU score for each mask in the batch.
    """
    # Calculate the intersection and union
    intersection = torch.logical_and(preds, targets).float().sum((1, 2, 3))  # Sum over the mask dimensions
    union = torch.logical_or(preds, targets).float().sum((1, 2, 3))

    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-6

    # Calculate IoU
    iou = (intersection + epsilon) / (union + epsilon)

    return iou.mean()  # Return the mean IoU score over the batch

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

def iou_loss_pytorch(boxA, boxB):
    # Ensure the coordinates are tensors
    boxA = torch.tensor(boxA, dtype=torch.float32)
    boxB = torch.tensor(boxB, dtype=torch.float32)

    # Calculate intersection coordinates
    xA = torch.max(boxA[..., 0], boxB[..., 0])
    yA = torch.max(boxA[..., 1], boxB[..., 1])
    xB = torch.min(boxA[..., 2], boxB[..., 2])
    yB = torch.min(boxA[..., 3], boxB[..., 3])

    # Calculate intersection area
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    # Calculate union area
    boxAArea = (boxA[..., 2] - boxA[..., 0]) * (boxA[..., 3] - boxA[..., 1])
    boxBArea = (boxB[..., 2] - boxB[..., 0]) * (boxB[..., 3] - boxB[..., 1])
    unionArea = boxAArea + boxBArea - interArea

    # Calculate IoU
    iou = interArea / unionArea

    # Loss is 1 - IoU
    return 1 - iou

class RollTransform:
    """Roll by one of the given angles."""

    def __init__(self):
        self.shifts = [(i*5,i*5) for i in range(20)]

    def __call__(self, x):
        idx = np.random.randint(len(self.shifts))
        shift = self.shifts[idx]
        return torch.roll(x, shift, dims = (1, 2))
    

# Average the results of several models from submissions placed in a directory
def average_submissions(dir_path = "data/predictions/averaging", new_name = "averaged_submission.csv"):
    n = len(os.listdir(dir_path))
    pred = np.zeros((n, 1296))
    for i, sub_files in enumerate(os.listdir(dir_path)):
        sub = pd.read_csv(os.path.join(dir_path, sub_files), index_col = 0)
        pred[i, :] += sub.iloc[i].values
        
    predictions = (pred / n)>0.5
    predictons = predictions.int()
    df = pd.DataFrame(predictions)

    files = [f.replace('.npy','') for f in os.listdir('data/processed/images_test')]
    df.index = files

    df.to_csv(os.path.join(dir_path,new_name), index=True)

    print("Predicted averaged masks saved in predictions/averaging")


