from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..base import BaseModel


class FasterRCNN(BaseModel):
    """FasterRCNN model from torchvision with custom number of classes."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        **kwargs
    ):
        """
        Args:
            num_classes: Number of classes (including background)
            pretrained: Whether to use pretrained backbone
            trainable_backbone_layers: Number of trainable layers in backbone
        """
        super().__init__()
        self.model_type = "detection"
        
        # Load pretrained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs
        )
        
        # Replace the classifier with a new one for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """Forward pass with optional targets for training."""
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        self.model.training = self.training
        return self.model(images, targets)
    
    def get_loss(self, predictions: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Get loss dictionary from predictions and targets."""
        # FasterRCNN returns losses during training
        return predictions
    
    def predict(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Make predictions for inference."""
        self.eval()
        if not isinstance(images, list):
            images = [images]
            
        with torch.no_grad():
            predictions = self.model(images)
        return predictions 