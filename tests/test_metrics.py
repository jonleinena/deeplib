import pytest
import torch
from deeplib.metrics import iou_score, dice_score, pixel_accuracy

def test_perfect_prediction():
    pred = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0)  # Add batch dimension
    target = torch.tensor([[0, 1], [2, 3]])
    num_classes = 4
    
    iou = iou_score(pred, target, num_classes)
    dice = dice_score(pred, target, num_classes)
    acc = pixel_accuracy(pred, target)
    
    assert torch.allclose(iou, torch.tensor(1.0))
    assert torch.allclose(dice, torch.tensor(1.0))
    assert torch.allclose(acc, torch.tensor(1.0))

def test_completely_wrong_prediction():
    pred = torch.tensor([[3, 2], [1, 0]]).unsqueeze(0)  # Add batch dimension
    target = torch.tensor([[0, 1], [2, 3]])
    num_classes = 4
    
    iou = iou_score(pred, target, num_classes)
    dice = dice_score(pred, target, num_classes)
    acc = pixel_accuracy(pred, target)
    
    assert torch.allclose(iou, torch.tensor(0.0))
    assert torch.allclose(dice, torch.tensor(0.0))
    assert torch.allclose(acc, torch.tensor(0.0))

def test_ignore_index():
    pred = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0)
    target = torch.tensor([[0, 255], [2, 3]])  # 255 is ignore index
    num_classes = 4
    
    iou = iou_score(pred, target, num_classes, ignore_index=255)
    dice = dice_score(pred, target, num_classes, ignore_index=255)
    acc = pixel_accuracy(pred, target, ignore_index=255)
    
    assert torch.allclose(iou, torch.tensor(1.0))
    assert torch.allclose(dice, torch.tensor(1.0))
    assert torch.allclose(acc, torch.tensor(1.0)) 