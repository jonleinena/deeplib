from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModel


class DoubleConv(nn.Module):
    """Double convolution block used in UNet."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool and double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_p)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with double convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout_p: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle cases where input size is not perfectly divisible by 2^n
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseModel):
    """U-Net architecture for semantic segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        features: int = 64,
        bilinear: bool = True,
        dropout_p: float = 0.0
    ):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB)
            features: Number of features in first layer (doubles in each down step)
            bilinear: Whether to use bilinear upsampling or transposed convolutions
            dropout_p: Dropout probability (0.0 means no dropout)
        """
        super().__init__()
        self.model_type = "segmentation"
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Initial double convolution
        self.inc = DoubleConv(in_channels, features, dropout_p)
        
        # Downsampling path
        self.down1 = Down(features, features * 2, dropout_p)
        self.down2 = Down(features * 2, features * 4, dropout_p)
        self.down3 = Down(features * 4, features * 8, dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor, dropout_p)
        
        # Upsampling path
        self.up1 = Up(features * 16, features * 8 // factor, bilinear, dropout_p)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear, dropout_p)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear, dropout_p)
        self.up4 = Up(features * 2, features, bilinear, dropout_p)
        
        # Output convolution
        self.outc = nn.Conv2d(features, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Dictionary containing output logits under key 'out'
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return {"out": logits}
    
    def get_loss(self, predictions: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate segmentation loss.
        
        Args:
            predictions: Dictionary containing model outputs
            target: Ground truth segmentation masks
            
        Returns:
            Dictionary containing the loss value under key 'seg_loss'
        """
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