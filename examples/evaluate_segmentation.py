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
    """Save visualization of prediction vs ground truth."""
    # Convert tensors to numpy arrays
    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Normalize image
    image = ((image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    
    # Create color maps for prediction and target
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR format, up to 4 classes
    pred_vis = np.zeros_like(image)
    target_vis = np.zeros_like(image)
    
    for cls, color in enumerate(colors[:pred.max() + 1]):
        pred_vis[pred == cls] = color
        target_vis[target == cls] = color
    
    # Blend with original image
    alpha = 0.5
    pred_blend = cv2.addWeighted(image, 1 - alpha, pred_vis, alpha, 0)
    target_blend = cv2.addWeighted(image, 1 - alpha, target_vis, alpha, 0)
    
    # Concatenate horizontally: Original | Prediction | Ground Truth
    vis = np.concatenate([image, pred_blend, target_blend], axis=1)
    
    # Save visualization
    cv2.imwrite(save_path, vis)


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
            print(np.max(predictions))
            print(predictions.shape)
            
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
            
            # Save visualizations for first batch
            if batch_idx == 0:
                pred_masks = torch.argmax(predictions, dim=1)
                for i in range(min(4, len(images))):
                    save_path = vis_dir / f"sample_{i}.png"
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