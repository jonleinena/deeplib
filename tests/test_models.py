import pytest
import torch
from deeplib.models.segmentation import UNet, DeepLabV3, DeepLabV3Plus

def test_unet_initialization():
    model = UNet(num_classes=4)
    assert isinstance(model, UNet)
    
    # Test output shape
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 4, 224, 224)

def test_deeplabv3_initialization():
    model = DeepLabV3(num_classes=4, pretrained=False)
    assert isinstance(model, DeepLabV3)
    
    # Test output shape
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 4, 224, 224)

def test_deeplabv3plus_initialization():
    model = DeepLabV3Plus(num_classes=4, pretrained=False)
    assert isinstance(model, DeepLabV3Plus)
    
    # Test output shape
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 4, 224, 224) 