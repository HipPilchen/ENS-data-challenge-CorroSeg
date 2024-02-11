import torch

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