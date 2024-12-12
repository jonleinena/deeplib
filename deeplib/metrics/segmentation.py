from typing import Optional

import torch
import torch.nn.functional as F


def iou_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    eps: float = 1e-6,
    exclude_background: bool = True
) -> torch.Tensor:
    """Calculate IoU score.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
        eps: Small constant to avoid division by zero
        exclude_background: Whether to exclude background class (0) from calculation
        
    Returns:
        Mean IoU score over non-background classes
    """
    outputs = torch.argmax(outputs, dim=1)
    ious = []
    
    # Start from 1 if excluding background
    start_cls = 1 if exclude_background else 0
    
    for cls in range(start_cls, num_classes):
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
    eps: float = 1e-6,
    exclude_background: bool = True
) -> torch.Tensor:
    """Calculate Dice score.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore from evaluation
        eps: Small constant to avoid division by zero
        exclude_background: Whether to exclude background class (0) from calculation
        
    Returns:
        Mean Dice score over non-background classes
    """
    outputs = torch.argmax(outputs, dim=1)
    dice_scores = []
    
    # Start from 1 if excluding background
    start_cls = 1 if exclude_background else 0
    
    for cls in range(start_cls, num_classes):
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
    ignore_index: Optional[int] = None,
    exclude_background: bool = True
) -> torch.Tensor:
    """Calculate pixel accuracy.
    
    Args:
        outputs: Model outputs of shape (N, C, H, W)
        targets: Ground truth of shape (N, H, W)
        ignore_index: Index to ignore from evaluation
        exclude_background: Whether to exclude background pixels from calculation
        
    Returns:
        Pixel accuracy (excluding background if specified)
    """
    outputs = torch.argmax(outputs, dim=1)
    
    # Create mask for valid pixels
    if exclude_background:
        valid_mask = targets > 0  # Exclude background
    else:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    
    if ignore_index is not None:
        valid_mask &= targets != ignore_index
    
    correct = (outputs == targets) & valid_mask
    return correct.float().sum() / valid_mask.float().sum() 