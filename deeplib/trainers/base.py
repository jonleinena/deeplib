from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer(ABC):
    """Base trainer class for all models."""
    
    def __init__(
        self,
        model: Any,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        monitor_metric: str = "loss",
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer to use
            scheduler: Learning rate scheduler
            device: Device to use for training
            monitor_metric: Metric name to monitor for early stopping and model saving
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        self.monitor_metric = monitor_metric
        self.epoch = 0
        self.best_metric = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        metrics = {}
        
        with tqdm(self.train_loader, desc=f'Epoch {self.epoch}') as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                loss_dict = self.train_step(batch)
                loss = sum(loss_dict.values())
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                metrics.update({k: v.item() for k, v in loss_dict.items()})
                pbar.set_postfix({k: f'{v:.4f}' for k, v in metrics.items()})
        
        if self.scheduler is not None:
            self.scheduler.step()
            
        return metrics
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Perform a single training step."""
        pass
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                batch_metrics = self.validate_step(batch)
                for k, v in batch_metrics.items():
                    metrics[k] = metrics.get(k, 0) + v.item()
        
        # Average the metrics
        metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        return metrics
    
    @abstractmethod
    def validate_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Perform a single validation step."""
        pass
    
    def train(
        self,
        num_epochs: int,
        save_path: Optional[str] = None,
        early_stopping: Optional[int] = None,
    ) -> Dict[str, list]:
        """Train the model for the specified number of epochs."""
        history = {'train': [], 'val': []}
        no_improve = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            # Save best model based on validation metric if available, otherwise training metric
            if save_path:
                # First try validation metrics, then training metrics
                current_metric = (
                    val_metrics.get(self.monitor_metric, float('inf'))
                    if val_metrics
                    else train_metrics.get(self.monitor_metric, float('inf'))
                )
                
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    self.model.save_weights(save_path)
                    no_improve = 0
                else:
                    no_improve += 1
                    
                if early_stopping and no_improve >= early_stopping:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        return history 