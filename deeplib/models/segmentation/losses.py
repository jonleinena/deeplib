from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Dice Loss for multi-class segmentation.
        Args:
            pred: Tensor of predictions (batch_size, C, H, W)
            target: Ground truth tensor (batch_size, H, W)
            
        Returns:
            Scalar Dice Loss averaged across classes
        """
        pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
        num_classes = pred.shape[1]
        
        # Convert target to one-hot and handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target * mask
            pred = pred * mask.unsqueeze(1)
        
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        dice = 0  # Initialize Dice score accumulator
        
        # Calculate Dice score for each class
        for c in range(num_classes):
            pred_c = pred[:, c]  # Predictions for class c
            target_c = target_one_hot[:, c]  # Ground truth for class c
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
            
            dice += (2. * intersection + self.smooth) / (union + self.smooth)  # Per-class Dice score
        
        return 1 - dice.mean() / num_classes  # Average Dice Loss across classes


class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes Jaccard/IoU Loss for multi-class segmentation.
        Args:
            pred: Tensor of predictions (batch_size, C, H, W)
            target: Ground truth tensor (batch_size, H, W)
            
        Returns:
            Scalar Jaccard Loss averaged across classes
        """
        pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
        num_classes = pred.shape[1]
        
        # Convert target to one-hot and handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target * mask
            pred = pred * mask.unsqueeze(1)
        
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        jaccard = 0  # Initialize IoU score accumulator
        
        # Calculate IoU score for each class
        for c in range(num_classes):
            pred_c = pred[:, c]  # Predictions for class c
            target_c = target_one_hot[:, c]  # Ground truth for class c
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2)) - intersection  # Sum of all pixels minus intersection
            
            jaccard += (intersection + self.smooth) / (union + self.smooth)  # Per-class IoU score
        
        return 1 - jaccard.mean() / num_classes  # Average Jaccard Loss across classes


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: Optional[int] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ComboLoss(nn.Module):
    def __init__(
        self,
        weights: dict = {'ce': 1.0, 'dice': 1.0},
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if 'ce' in weights else None
        self.dice = DiceLoss(ignore_index=ignore_index) if 'dice' in weights else None
        self.jaccard = JaccardLoss(ignore_index=ignore_index) if 'jaccard' in weights else None
        self.focal = FocalLoss(ignore_index=ignore_index) if 'focal' in weights else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        if self.ce is not None:
            loss += self.weights['ce'] * self.ce(pred, target)
        if self.dice is not None:
            loss += self.weights['dice'] * self.dice(pred, target)
        if self.jaccard is not None:
            loss += self.weights['jaccard'] * self.jaccard(pred, target)
        if self.focal is not None:
            loss += self.weights['focal'] * self.focal(pred, target)
        
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor] = None, ignore_index: Optional[int] = None):
        """
        Args:
            weights: Optional tensor of class weights (C,)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.weights = weights
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            pred, target,
            weight=self.weights,
            ignore_index=self.ignore_index if self.ignore_index is not None else -100
        )


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary Cross Entropy for binary segmentation.
        
        Args:
            pred: Predictions (batch_size, 1, H, W) or (batch_size, H, W)
            target: Binary targets (batch_size, H, W)
        """
        # Ensure pred has correct shape and apply sigmoid
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        pred = torch.sigmoid(pred)
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        
        return F.binary_cross_entropy(pred, target.float())


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q: float = 0.7, ignore_index: Optional[int] = None):
        """Generalized Cross Entropy Loss (https://arxiv.org/abs/1805.07836)
        
        Args:
            q: Quality parameter for noise estimation (default: 0.7)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.q = q
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            pred = pred[mask.unsqueeze(1).expand_as(pred)].view(-1, pred.size(1))
        
        # Get predicted probabilities for the target class
        pred = torch.gather(pred, 1, target.unsqueeze(1)).squeeze(1)
        
        # Calculate generalized cross entropy
        loss = (1 - pred.pow(self.q)) / self.q
        return loss.mean()


# Dictionary of available loss functions
LOSS_FUNCTIONS = {
    'ce': lambda ignore_index: nn.CrossEntropyLoss(ignore_index=ignore_index),
    'wce': WeightedCrossEntropyLoss,
    'bce': BinaryCrossEntropyLoss,
    'gce': GeneralizedCrossEntropyLoss,
    'dice': DiceLoss,
    'jaccard': JaccardLoss,
    'focal': FocalLoss,
    'combo': ComboLoss
} 