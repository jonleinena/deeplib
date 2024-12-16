from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights, ResNet101_Weights

from ..base import BaseModel


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list):
        super().__init__()
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        
        # Atrous convolutions
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project to output channels
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(BaseModel):
    """DeepLabV3+ architecture for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_stride: int = 16,
        dropout_p: float = 0.1,  # Default dropout probability
        **kwargs
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone network ('resnet50' or 'resnet101')
            pretrained: Whether to use pretrained backbone
            output_stride: Output stride of the encoder (16 or 8)
        """
        super().__init__()
        self.model_type = "segmentation"
        self.num_classes = num_classes
        
        if output_stride not in [8, 16]:
            raise ValueError("Output stride should be 8 or 16")
        
        # Load pretrained backbone
        replace_stride_with_dilation = [False, False, output_stride == 8]
        if backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        elif backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.resnet101(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract backbone layers
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # For low-level features
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # ASPP module
        inplanes = 2048
        aspp_dilations = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.aspp = ASPP(inplanes, 256, aspp_dilations)
        
        # Low-level features processing (from layer1 which has 256 channels)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        # Final decoder with dropout
        decoder_layers = [
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(256, num_classes, 1)
        ]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        input_shape = x.shape[-2:]
        
        # Backbone
        x = self.initial(x)
        low_level_feat = self.layer1(x)  # Get low-level features from layer1
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        low_level_feat = self.low_level_conv(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return {"out": x}
    
    def get_loss(self, predictions: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate segmentation loss."""
        if self.loss_fn is None:
            # Default to cross entropy if no loss function is configured
            self.configure_loss('ce', {'ignore_index': 255})
            
        return {
            "seg_loss": self.loss_fn(predictions["out"], target)
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for inference."""
        self.eval()
        with torch.no_grad():
            output = self(x)
        return torch.argmax(output["out"], dim=1) 