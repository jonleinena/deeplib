from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights

from ..base import BaseModel


class DeepLabV3(BaseModel):
    """DeepLabV3 model from torchvision with custom number of classes."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        backbone: str = "resnet50",
        aux_loss: bool = True,
        dropout_p: float = 0.1,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            pretrained: Whether to use pretrained backbone
            backbone: Backbone network (resnet50 or resnet101)
            aux_loss: Whether to use auxiliary loss
            dropout_p: Dropout probability (0.0 means no dropout)
        """
        super().__init__()
        self.model_type = "segmentation"
        self.aux_loss = aux_loss
        
        # Load pretrained model
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=weights,
                aux_loss=aux_loss,
                **kwargs
            )
        elif backbone == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=weights,
                aux_loss=aux_loss,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier head with dropout
        self.model.classifier = nn.Sequential(
            DeepLabHead(2048, num_classes),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        )
        
        if aux_loss:
            self.model.aux_classifier[4] = nn.Sequential(
                nn.Conv2d(256, num_classes, kernel_size=1),
                nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        result = self.model(x)
        return result
    
    def get_loss(self, predictions: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate segmentation loss."""
        if self.loss_fn is None:
            # Default to cross entropy if no loss function is configured
            self.configure_loss('ce', {'ignore_index': 255})
            
        losses = {}
        
        # Main segmentation loss
        losses["seg_loss"] = self.loss_fn(predictions["out"], target)
        
        # Auxiliary loss (always use cross entropy for auxiliary loss)
        if self.aux_loss and self.training:
            aux_loss = F.cross_entropy(
                predictions["aux"],
                target,
                ignore_index=255
            )
            losses["aux_loss"] = aux_loss * 0.5
            
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for inference."""
        self.eval()
        with torch.no_grad():
            output = self.model(x)
        return torch.argmax(output["out"], dim=1) 