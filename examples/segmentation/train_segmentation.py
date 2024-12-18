import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import albumentations as A
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deeplib.datasets import SegmentationDataset
from deeplib.models.segmentation import UNet
from deeplib.trainers import SegmentationTrainer
from deeplib.metrics import iou_score, dice_score
from deeplib.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_transform(train: bool = True, input_size: int = 224):
    """Get albumentations transforms."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(input_size, input_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2(),
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--images_dir", type=str, default="images")
    parser.add_argument("--masks_dir", type=str, default="masks")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--input_size", type=int, default=192)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, mps, or cpu). If not specified, will use the best available.")
    parser.add_argument("--monitor_metric", type=str, default="iou",
                      help="Metric to monitor for early stopping.")
    parser.add_argument("--loss", type=str, default="dice", choices=["ce", "dice", "wce", "jaccard", "focal"],
                      help="Loss function to use (ce: cross entropy, dice: dice loss, wce: weighted cross entropy, jaccard: IoU loss, focal: focal loss)")
    parser.add_argument("--logger", type=str, choices=["tensorboard", "mlflow", "wandb", "none"], default="tensorboard",
                      help="Logger to use for tracking experiments")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        root=args.data_root,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        num_classes=args.num_classes,
        split="train",
        transform=get_transform(train=True, input_size=args.input_size),
        file_extension="png"
    )
    
    val_dataset = SegmentationDataset(
        root=args.data_root,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        num_classes=args.num_classes,
        split="val",
        transform=get_transform(train=False, input_size=args.input_size),
        file_extension="png"
    )
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if device.type in ["cuda", "mps"] else False
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device.type in ["cuda", "mps"] else False
    )
    
    # Create model
    model = UNet(
        num_classes=args.num_classes,
        dropout_p=args.dropout_p
    )
    
    # Configure loss function
    model.configure_loss(args.loss, {'ignore_index': args.ignore_index})
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    # Define custom metrics
    custom_metrics = [
        lambda x, y: iou_score(x, y, args.num_classes, args.ignore_index),
        lambda x, y: dice_score(x, y, args.num_classes, args.ignore_index)
    ]
    metric_names = ["iou", "dice"]
    
    # Set up logger if requested
    logger = None
    if args.logger != "none":
        # Create experiment name from parameters
        experiment_name = f"segmentation_{args.loss}_loss"
        run_name = f"bs{args.batch_size}_lr{args.learning_rate}"
        
        if args.logger == "tensorboard":
            logger = TensorBoardLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                artifact_location=str(output_dir / "tensorboard")
            )
        elif args.logger == "mlflow":
            logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                artifact_location=str(output_dir / "mlflow")
            )
        elif args.logger == "wandb":
            logger = WandbLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                project="deeplib-segmentation"
            )
            
        with logger:
            logger.log_params({
                "loss_type": args.loss,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "input_size": args.input_size,
                "num_classes": args.num_classes,
            "dropout_p": args.dropout_p,
            "ignore_index": args.ignore_index,
            "device": str(device),
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau"
        })
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        metrics=custom_metrics,
        ignore_index=args.ignore_index,
        monitor_metric=args.monitor_metric,
        logger=logger
    )
    
    # Train model
    save_path = output_dir / "checkpoints" / "UNet_segmentation.pth"
    save_path.parent.mkdir(exist_ok=True)
    
    trainer.train(
        num_epochs=args.num_epochs,
        save_path=str(save_path)
    )



if __name__ == "__main__":
    main() 