import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplib.datasets import NEUSegDataset
from deeplib.models.segmentation import DeepLabV3
from train_segmentation import get_transform


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> dict:
    """Calculate segmentation metrics."""
    metrics = {}
    
    # Calculate IoU for each class
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        iou = intersection / (union + 1e-10)
        metrics[f"iou_class_{cls}"] = iou
    
    # Calculate mean IoU
    metrics["mean_iou"] = sum(metrics[f"iou_class_{i}"] for i in range(num_classes)) / num_classes
    
    # Calculate pixel accuracy
    correct = (pred == target).sum().item()
    total = target.numel()
    metrics["pixel_accuracy"] = correct / total
    
    return metrics


def save_visualization(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, save_path: str):
    """Save visualization of prediction vs ground truth."""
    # Convert tensors to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Normalize image
    image = ((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    # Create color maps for prediction and target
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # BGR format
    pred_vis = np.zeros_like(image)
    target_vis = np.zeros_like(image)
    
    for cls, color in enumerate(colors):
        pred_vis[pred == cls] = color
        target_vis[target == cls] = color
    
    # Blend with original image
    alpha = 0.5
    pred_blend = cv2.addWeighted(image, 1 - alpha, pred_vis, alpha, 0)
    target_blend = cv2.addWeighted(image, 1 - alpha, target_vis, alpha, 0)
    
    # Concatenate horizontally
    vis = np.concatenate([image, pred_blend, target_blend], axis=1)
    
    # Save visualization
    cv2.imwrite(save_path, vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    args = parser.parse_args()
    
    # Create dataset and dataloader
    dataset = NEUSegDataset(
        root=args.data_root,
        split="val",
        transform=get_transform(train=False, input_size=args.input_size)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model and load checkpoint
    model = DeepLabV3(num_classes=3, pretrained=False)
    model.load_weights(args.checkpoint)
    model = model.to(args.device)
    model.eval()
    
    # Create visualization directory
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(exist_ok=True)
    
    # Evaluate
    metrics_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs["out"], dim=1)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(predictions, masks)
            metrics_list.append(batch_metrics)
            
            # Save visualizations for first batch
            if batch_idx == 0:
                for i in range(min(4, len(images))):
                    save_path = vis_dir / f"sample_{i}.png"
                    save_visualization(
                        images[i],
                        predictions[i],
                        masks[i],
                        str(save_path)
                    )
    
    # Average metrics
    final_metrics = {}
    for key in metrics_list[0].keys():
        final_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    
    # Print results
    print("\nEvaluation Results:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main() 