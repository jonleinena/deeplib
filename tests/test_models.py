import pytest
import torch
from deeplib.models.segmentation import UNet, DeepLabV3, DeepLabV3Plus

def test_unet_initialization():
    model = UNet(num_classes=4)
    assert isinstance(model, UNet)
    
    # Test output shape with larger input
    x = torch.randn(2, 3, 256, 256)  # Increased batch size and spatial dimensions
    with torch.no_grad():
        output = model(x)
        if isinstance(output, dict):
            output = output['out']  # Handle dict output
    assert output.shape == (2, 4, 256, 256)

def test_deeplabv3_initialization():
    model = DeepLabV3(num_classes=4, pretrained=False)
    assert isinstance(model, DeepLabV3)
    
    # Test output shape with larger input
    x = torch.randn(2, 3, 256, 256)  # Increased batch size and spatial dimensions
    model.eval()  # Set to eval mode to avoid batch norm issues
    with torch.no_grad():
        output = model(x)
        if isinstance(output, dict):
            output = output['out']  # Handle dict output
    assert output.shape == (2, 4, 256, 256)

def test_deeplabv3plus_initialization():
    model = DeepLabV3Plus(num_classes=4, pretrained=False)
    assert isinstance(model, DeepLabV3Plus)
    
    # Test output shape with larger input
    x = torch.randn(2, 3, 256, 256)  # Increased batch size and spatial dimensions
    model.eval()  # Set to eval mode to avoid batch norm issues
    with torch.no_grad():
        output = model(x)
        if isinstance(output, dict):
            output = output['out']  # Handle dict output
    assert output.shape == (2, 4, 256, 256) 