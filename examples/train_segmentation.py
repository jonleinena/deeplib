import argparse
from pathlib import Path

import albumentations as A
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import CosineAnnealingLR

from deeplib.datasets import NEUSegDataset
from deeplib.models.segmentation import DeepLabV3
from deeplib.trainers import BaseTrainer


def get_transform(train: bool = True, input_size: int = 512):
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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = NEUSegDataset(
        root=args.data_root,
        split="train",
        transform=get_transform(train=True, input_size=args.input_size)
    )
    
    val_dataset = NEUSegDataset(
        root=args.data_root,
        split="val",
        transform=get_transform(train=False, input_size=args.input_size)
    )
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = DeepLabV3(
        num_classes=4,  # background + 3 defect types
        pretrained=True,
        backbone="resnet50"
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Create trainer
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device
    )
    
    # Train model
    save_path = Path("checkpoints") / "deeplabv3_neu_seg.pth"
    save_path.parent.mkdir(exist_ok=True)
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        save_path=str(save_path),
        early_stopping=10
    )


if __name__ == "__main__":
    main() 