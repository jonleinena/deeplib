from typing import Dict, Tuple, List, Callable, Optional

import torch
from tqdm import tqdm

from .base import BaseTrainer
from ..metrics import iou_score, dice_score, pixel_accuracy
from ..loggers import BaseLogger


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
        monitor_metric: str = "seg_loss",
        logger: Optional[BaseLogger] = None,
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
            monitor_metric: Metric to monitor for early stopping
            logger: Logger instance for experiment tracking
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            monitor_metric=monitor_metric,
            logger=logger
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