from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)


class BaseModel(nn.Module, ABC):
    """Base class for all models in DeepLib."""
    
    def __init__(self):
        super().__init__()
        self.model_type: str = ""  # detection, segmentation, or anomaly
        self.loss_fn = None
        
    def configure_loss(self, loss_type: str, loss_params: Optional[Dict] = None) -> None:
        """Configure loss function.
        
        Args:
            loss_type: Type of loss function to use
            loss_params: Parameters for the loss function
        """
        from .segmentation.losses import LOSS_FUNCTIONS
        
        if loss_type not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss type: {loss_type}. "
                           f"Supported types are: {list(LOSS_FUNCTIONS.keys())}")
        
        params = loss_params or {}
        self.loss_fn = LOSS_FUNCTIONS[loss_type](**params)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_loss(self, predictions: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """Calculate loss for the model."""
        pass
    
    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str) -> None:
        """Load model weights from disk."""
        self.load_state_dict(torch.load(path))
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> Any:
        """Make prediction for inference."""
        pass
    
    @property
    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 