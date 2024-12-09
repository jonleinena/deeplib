from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from ..base import BaseModel


class DeepLabV3(BaseModel):
    """DeepLabV3 model from torchvision with custom number of classes."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        backbone: str = "resnet50",
        aux_loss: bool = True,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes
            pretrained: Whether to use pretrained backbone
            backbone: Backbone network (resnet50 or resnet101)
            aux_loss: Whether to use auxiliary loss
        """
        super().__init__()
        self.model_type = "segmentation"
        self.aux_loss = aux_loss
        
        # Load pretrained model
        if backbone == "resnet50":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=pretrained,
                aux_loss=aux_loss,
                **kwargs
            )
        elif backbone == "resnet101":
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                pretrained=pretrained,
                aux_loss=aux_loss,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier head
        self.model.classifier = DeepLabHead(2048, num_classes)
        if aux_loss:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        result = self.model(x)
        return result
    
    def get_loss(self, predictions: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate segmentation loss."""
        losses = {}
        
        # Main segmentation loss
        losses["seg_loss"] = F.cross_entropy(
            predictions["out"],
            target,
            ignore_index=255
        )
        
        # Auxiliary loss
        if self.aux_loss and self.training:
            losses["aux_loss"] = F.cross_entropy(
                predictions["aux"],
                target,
                ignore_index=255
            ) * 0.5
            
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for inference."""
        self.eval()
        with torch.no_grad():
            output = self.model(x)
        return torch.argmax(output["out"], dim=1) 