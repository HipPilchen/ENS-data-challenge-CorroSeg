# Function to calculate IoU score
def iou_score(pred, target):
    # Ensure pred and target are boolean for bitwise operations
    pred_bool = pred.bool()
    target_bool = target.bool()

    intersection = (pred_bool & target_bool).float().sum((1, 2))  # Intersection points
    union = (pred_bool | target_bool).float().sum((1, 2))         # Union points

    iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return iou.mean()  # Compute the mean IoU score over the batch