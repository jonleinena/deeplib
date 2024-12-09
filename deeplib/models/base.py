from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models in DeepLib."""
    
    def __init__(self):
        super().__init__()
        self.model_type: str = ""  # detection, segmentation, or anomaly
        
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