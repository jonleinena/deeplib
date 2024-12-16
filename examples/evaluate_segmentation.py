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
from deeplib.trainers import SegmentationTrainer
from train_segmentation import get_transform

import random


def save_visualization(image: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, save_path: str):
    """Save visualization of original image, prediction and ground truth side by side."""
    # Convert tensors to numpy arrays and denormalize the image
    image = image.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    
    # Denormalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    pred = pred.squeeze(0).cpu().numpy()
    target = target.cpu().numpy()
 
    # Define colors for each class (in RGB format)
    colors = np.array([
        [0, 0, 0],      # Class 0 (Background) - Black
        [255, 0, 0],    # Class 1 - Red
        [0, 255, 0],    # Class 2 - Green
        [0, 0, 255]     # Class 3 - Blue
    ])

    # Create colored masks for prediction and target
    pred_mask = colors[pred]
    target_mask = colors[target]

    # Create a side-by-side visualization
    h, w = image.shape[:2]
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # Place images side by side
    vis[:, :w] = image
    vis[:, w:2*w] = target_mask
    vis[:, 2*w:] = pred_mask
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, 'Original', (w//2 - 50, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(vis, 'Ground Truth', (3*w//2 - 80, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(vis, 'Prediction', (5*w//2 - 60, 30), font, 1, (255, 255, 255), 2)
    
    # Save the visualization
    cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=192)
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
        transform=get_transform(train=False, input_size=args.input_size),
        file_extension="png"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
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

    trainer = SegmentationTrainer(model=model, train_loader=None, val_loader=dataloader, device=device)

    # Create visualization directory
    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(exist_ok=True)
    
 
    metrics = trainer.validate()
    print(metrics)

    random_numbers = random.sample(range(0, 841), 8)

    for i in random_numbers:
        image, mask = dataset[i]
        pred = model.predict(image.unsqueeze(0).to(device))
        save_visualization(image, pred, mask, vis_dir / f"visualization_{i}.png")

    


if __name__ == "__main__":
    main() 