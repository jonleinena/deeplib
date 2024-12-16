from typing import Dict, Tuple, List, Callable, Optional

import torch
from tqdm import tqdm

from .base import BaseTrainer
from ..metrics import iou_score, dice_score, pixel_accuracy


class SegmentationTrainer(BaseTrainer):
    """Trainer class for semantic segmentation models."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics: Optional[List[Callable]] = None,
        ignore_index: Optional[int] = None,
        monitor_metric: str = "seg_loss"
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            metrics: List of metric functions to compute during validation
            ignore_index: Index to ignore in metrics computation
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            monitor_metric=monitor_metric
        )
        self.metrics = metrics or [
            lambda x, y: iou_score(x, y, model.num_classes, ignore_index),
            lambda x, y: dice_score(x, y, model.num_classes, ignore_index),
            lambda x, y: pixel_accuracy(x, y, ignore_index)
        ]
        self.metric_names = ["iou", "dice", "accuracy"]
        self.ignore_index = ignore_index
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        metrics = {}
        metrics.update(self.model.get_loss(outputs, masks))  # These will have '_loss' suffix
        
        # Calculate additional metrics during training (with no_grad)
        with torch.no_grad():
            for name, metric_fn in zip(self.metric_names, self.metrics):
                metrics[name] = metric_fn(outputs["out"], masks)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        total_metrics = {}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, batch in enumerate(pbar, 1):
                    metrics = self.validate_step(batch)
                    
                    # Update total metrics
                    for k, v in metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0) + v.item()
                    
                    # Update progress bar with running averages
                    running_metrics = {k: f"{v / batch_idx:.4f}" for k, v in total_metrics.items()}
                    pbar.set_postfix(running_metrics)
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        return avg_metrics
    
    def validate_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate validation metrics
        metrics = {}
        
        # Calculate loss
        metrics.update(self.model.get_loss(outputs, masks))
        
        # Calculate additional metrics
        for name, metric_fn in zip(self.metric_names, self.metrics):
            metrics[name] = metric_fn(outputs["out"], masks)
        
        return metrics