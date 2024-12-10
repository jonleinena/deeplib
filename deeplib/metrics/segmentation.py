from typing import Optional

import torch
import torch.nn.functional as F


def iou_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Calculate IoU score.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
        eps: Small constant to avoid division by zero
        
    Returns:
        IoU score for each class
    """
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_mask = (outputs == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        iou = (intersection + eps) / (union + eps)
        ious.append(iou)
    
    return torch.stack(ious).mean()


def dice_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Calculate Dice score.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
        eps: Small constant to avoid division by zero
        
    Returns:
        Dice score for each class
    """
    outputs = torch.argmax(outputs, dim=1)
    dice_scores = []
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        pred_mask = (outputs == cls)
        target_mask = (targets == cls)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = pred_mask.float().sum() + target_mask.float().sum()
        
        dice = (2. * intersection + eps) / (union + eps)
        dice_scores.append(dice)
    
    return torch.stack(dice_scores).mean()


def pixel_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: Optional[int] = None
) -> torch.Tensor:
    """Calculate pixel accuracy.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        ignore_index: Index to ignore from evaluation
        
    Returns:
        Pixel accuracy
    """
    outputs = torch.argmax(outputs, dim=1)
    mask = targets != ignore_index if ignore_index is not None else torch.ones_like(targets, dtype=torch.bool)
    correct = (outputs == targets) & mask
    return correct.float().sum() / mask.float().sum() 