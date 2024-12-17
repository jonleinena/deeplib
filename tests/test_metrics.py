import pytest
import torch
from deeplib.metrics import iou_score, dice_score, pixel_accuracy

def print_debug_info(name, pred, target, iou, dice, acc):
    print(f"\n=== {name} Debug Info ===")
    print(f"Predictions:\n{torch.argmax(pred, dim=1).squeeze()}")
    print(f"Target:\n{target}")
    print(f"IoU: {iou}")
    print(f"Dice: {dice}")
    print(f"Accuracy: {acc}")

def create_one_hot_pred(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class indices to logits that will produce the same predictions."""
    one_hot = torch.zeros((indices.shape[0], num_classes, *indices.shape[1:]))
    for i in range(num_classes):
        one_hot[:, i][indices == i] = 10.0  # High logit value for correct class
    return one_hot

def test_perfect_prediction():
    indices = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0)  # Add batch dimension
    target = torch.tensor([[0, 1], [2, 3]])
    num_classes = 4
    
    # Convert to one-hot logits
    pred = create_one_hot_pred(indices, num_classes)
    
    # Test with background included
    iou = iou_score(pred, target, num_classes, exclude_background=False)
    dice = dice_score(pred, target, num_classes, exclude_background=False)
    acc = pixel_accuracy(pred, target, exclude_background=False)
    
    print_debug_info("Perfect Prediction", pred, target, iou, dice, acc)
    
    # Each class should have perfect scores
    assert torch.allclose(iou, torch.tensor(1.0), rtol=1e-3), f"Expected IoU to be 1.0, got {iou}"
    assert torch.allclose(dice, torch.tensor(1.0), rtol=1e-3), f"Expected Dice to be 1.0, got {dice}"
    assert torch.allclose(acc, torch.tensor(1.0), rtol=1e-3), f"Expected accuracy to be 1.0, got {acc}"

def test_completely_wrong_prediction():
    indices = torch.tensor([[3, 2], [1, 0]]).unsqueeze(0)  # Add batch dimension
    target = torch.tensor([[0, 1], [2, 3]])
    num_classes = 4
    
    # Convert to one-hot logits
    pred = create_one_hot_pred(indices, num_classes)
    
    # Test with background included
    iou = iou_score(pred, target, num_classes, exclude_background=False)
    dice = dice_score(pred, target, num_classes, exclude_background=False)
    acc = pixel_accuracy(pred, target, exclude_background=False)
    
    print_debug_info("Completely Wrong", pred, target, iou, dice, acc)
    
    # All predictions are wrong
    assert torch.allclose(iou, torch.tensor(0.0), rtol=1e-3, atol=1e-3), f"Expected IoU to be 0.0, got {iou}"
    assert torch.allclose(dice, torch.tensor(0.0), rtol=1e-3, atol=1e-3), f"Expected Dice to be 0.0, got {dice}"
    assert torch.allclose(acc, torch.tensor(0.0), rtol=1e-3, atol=1e-3), f"Expected accuracy to be 0.0, got {acc}"

def test_ignore_index():
    indices = torch.tensor([[0, 1], [2, 3]]).unsqueeze(0)
    target = torch.tensor([[0, 255], [2, 3]])  # 255 is ignore index
    num_classes = 4
    
    # Convert to one-hot logits
    pred = create_one_hot_pred(indices, num_classes)
    
    # Test with background included
    iou = iou_score(pred, target, num_classes, ignore_index=255, exclude_background=False)
    dice = dice_score(pred, target, num_classes, ignore_index=255, exclude_background=False)
    acc = pixel_accuracy(pred, target, ignore_index=255, exclude_background=False)
    
    print_debug_info("Ignore Index", pred, target, iou, dice, acc)
    
    # Should have 0.75 score because we're averaging across all classes
    # Only 3 out of 4 classes appear in the valid pixels
    expected_iou = torch.tensor(0.75)
    expected_dice = torch.tensor(0.75)
    expected_acc = torch.tensor(1.0)  # All valid pixels are correct
    
    assert torch.allclose(iou, expected_iou, rtol=1e-3), f"Expected IoU to be {expected_iou}, got {iou}"
    assert torch.allclose(dice, expected_dice, rtol=1e-3), f"Expected Dice to be {expected_dice}, got {dice}"
    assert torch.allclose(acc, expected_acc, rtol=1e-3), f"Expected accuracy to be {expected_acc}, got {acc}"

def test_partial_match():
    indices = torch.tensor([[0, 0], [1, 1]]).unsqueeze(0)
    target = torch.tensor([[0, 1], [0, 1]])
    num_classes = 2
    
    # Convert to one-hot logits
    pred = create_one_hot_pred(indices, num_classes)
    
    # Test with background included
    iou = iou_score(pred, target, num_classes, exclude_background=False)
    dice = dice_score(pred, target, num_classes, exclude_background=False)
    acc = pixel_accuracy(pred, target, exclude_background=False)
    
    print_debug_info("Partial Match", pred, target, iou, dice, acc)
    
    # Expected values when including background
    # Class 0: IoU = (1 TP) / (2 FP + 1 TP + 1 FN) = 1/4
    # Class 1: IoU = (1 TP) / (2 FP + 1 TP + 1 FN) = 1/4
    # Mean IoU = (1/3 + 1/3) / 2 = 1/3
    expected_iou = torch.tensor(1/3)
    expected_dice = torch.tensor(0.5)  # Mean of class 0 (0.5) and class 1 (0.5)
    expected_acc = torch.tensor(0.5)   # 2 correct out of 4 pixels
    
    assert torch.allclose(iou, expected_iou, rtol=1e-3), f"Expected IoU {expected_iou}, got {iou}"
    assert torch.allclose(dice, expected_dice, rtol=1e-3), f"Expected Dice {expected_dice}, got {dice}"
    assert torch.allclose(acc, expected_acc, rtol=1e-3), f"Expected accuracy {expected_acc}, got {acc}"
  