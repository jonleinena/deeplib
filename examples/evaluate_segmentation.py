import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplib.datasets import SegmentationDataset
from deeplib.models.segmentation import DeepLabV3, DeepLabV3Plus, UNet
from deeplib.metrics import iou_score, dice_score, pixel_accuracy
from train_segmentation import get_transform


def save_visualization(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, save_path: str):
    """Save visualization of original image, prediction and ground truth side by side."""
    # Convert tensors to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Normalize image to 0-255 range
    image = ((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    # Define colors for each class (in RGB format)
    colors = np.array([
        [0, 0, 0],      # Class 0 (Background) - Black
        [255, 0, 0],    # Class 1 - Red
        [0, 255, 0],    # Class 2 - Green
        [0, 0, 255]     # Class 3 - Blue
    ])
    
    # Create colored masks
    h, w = pred.shape
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    target_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Print unique values for debugging
    print(f"Unique values in prediction: {np.unique(pred)}")
    print(f"Unique values in target: {np.unique(target)}")
    
    # Map each pixel to its color
    for i in range(len(colors)):
        pred_colored[pred == i] = colors[i]
        target_colored[target == i] = colors[i]
    
    # Create the final visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_height = 30
    
    # Create a white background for titles
    title_strip = np.ones((title_height, image.shape[1] * 3, 3), dtype=np.uint8) * 255
    
    # Add titles
    cv2.putText(title_strip, 'Original Image', (image.shape[1]//3 - 80, 20), font, 0.6, (0,0,0), 2)
    cv2.putText(title_strip, 'Ground Truth', (image.shape[1] + image.shape[1]//3 - 70, 20), font, 0.6, (0,0,0), 2)
    cv2.putText(title_strip, 'Prediction', (2*image.shape[1] + image.shape[1]//3 - 60, 20), font, 0.6, (0,0,0), 2)
    
    # Add color legend
    legend_height = 20
    legend = np.ones((legend_height, image.shape[1] * 3, 3), dtype=np.uint8) * 255
    x_offset = 10
    for i, color in enumerate(colors[1:], 1):  # Skip background color
        cv2.putText(legend, f'Class {i}', (x_offset, 15), font, 0.4, (0,0,0), 1)
        cv2.rectangle(legend, (x_offset + 50, 5), (x_offset + 70, 15), color.tolist(), -1)
        x_offset += 100
    
    # Concatenate images horizontally
    vis = np.concatenate([image, target_colored, pred_colored], axis=1)
    
    # Add title strip and legend
    vis = np.concatenate([title_strip, vis, legend], axis=0)
    
    # Save visualization
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))  # Convert to BGR for cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="deeplabv3",
                      choices=["deeplabv3", "deeplabv3plus", "unet"],
                      help="Type of model to evaluate")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--masks_dir", type=str, default="masks")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    parser.add_argument("--ignore_index", type=int, default=255)
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(
        root=args.data_root,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        num_classes=args.num_classes,
        split="val",
        transform=get_transform(train=False, input_size=args.input_size)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type in ["cuda", "mps"] else False
    )
    
    # Create model and load checkpoint
    if args.model_type == "deeplabv3":
        model = DeepLabV3(num_classes=args.num_classes, pretrained=False)
    elif args.model_type == "deeplabv3plus":
        model = DeepLabV3Plus(num_classes=args.num_classes, pretrained=False)
    else:  # unet
        model = UNet(num_classes=args.num_classes)
    
    model.load_weights(args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create visualization directory
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(exist_ok=True)
    
    # Evaluate
    total_metrics = {
        "iou": 0.0,
        "dice": 0.0,
        "accuracy": 0.0,
        "loss": 0.0
    }
    
    num_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = outputs["out"]
            
            # Print raw prediction stats for first batch
            if batch_idx == 0:
                print("\nRaw prediction analysis:")
                print("Raw output shape:", predictions.shape)
                print("Raw output min/max:", predictions.min().item(), predictions.max().item())
                
                # Look at raw predictions for each class
                for i in range(predictions.shape[1]):
                    class_preds = predictions[:, i]
                    print(f"Class {i} stats - min: {class_preds.min().item():.3f}, "
                          f"max: {class_preds.max().item():.3f}, "
                          f"mean: {class_preds.mean().item():.3f}")
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(predictions, dim=1)
            pred_masks = torch.argmax(probabilities, dim=1)
            
            if batch_idx == 0:
                print("\nAfter softmax/argmax:")
                print("Probability min/max:", probabilities.min().item(), probabilities.max().item())
                for i in range(probabilities.shape[1]):
                    class_probs = probabilities[:, i]
                    print(f"Class {i} probability stats - min: {class_probs.min().item():.3f}, "
                          f"max: {class_probs.max().item():.3f}, "
                          f"mean: {class_probs.mean().item():.3f}")
                print("\nClass distribution in first batch:")
            
            # Calculate metrics
            total_metrics["iou"] += iou_score(
                predictions, masks, args.num_classes, 
                args.ignore_index, exclude_background=True
            ).item()
            total_metrics["dice"] += dice_score(
                predictions, masks, args.num_classes, 
                args.ignore_index, exclude_background=True
            ).item()
            total_metrics["accuracy"] += pixel_accuracy(
                predictions, masks, args.ignore_index, 
                exclude_background=True
            ).item()
            
            # Calculate loss
            loss_dict = model.get_loss(outputs, masks)
            total_metrics["loss"] += loss_dict["seg_loss"].item()
            
            # Save visualizations for first few batches
            if 6 < batch_idx < 9:  # Save first 3 batches
                print(f"\nBatch {batch_idx} analysis:")
                print("Predictions shape:", predictions.shape)
                print("Unique values in predictions after argmax:", torch.unique(pred_masks).cpu().numpy())
                print("Unique values in ground truth:", torch.unique(masks).cpu().numpy())
                
                # Print class distribution
                for i in range(args.num_classes):
                    pred_count = (pred_masks == i).sum().item()
                    target_count = (masks == i).sum().item()
                    print(f"Class {i} - predicted: {pred_count}, target: {target_count}")
                
                # Save visualizations for each image in the batch
                for i in range(min(4, len(images))):
                    save_path = vis_dir / f"batch_{batch_idx}_sample_{i}.png"
                    save_visualization(
                        images[i],
                        pred_masks[i],
                        masks[i],
                        str(save_path)
                    )
    
    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # Print results
    print("\nEvaluation Results:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main() 