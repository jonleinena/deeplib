# DeepLib

A unified PyTorch library for computer vision tasks, focusing on object detection, semantic segmentation, and anomaly detection.

## Features

- **Semantic Segmentation Models** (✅ Implemented)
  - UNet
  - DeepLabV3
  - DeepLabV3+

- **Object Detection Models** (🚧 In Progress)
  - YOLOv4
  - YOLOv5
  - YOLOX
  - YOLOv7 and YOLOv9
  - Faster R-CNN

- **Anomaly Detection Models** (🚧 In Progress)
  - PatchCore
  - FastFlow
  - PADIM
  - Other anomalib implementations

## Installation

```bash
pip install -e .
```

## Quick Start - Semantic Segmentation

```python
from deeplib.models.segmentation import UNet, DeepLabV3, DeepLabV3Plus
from deeplib.trainers import SegmentationTrainer
from deeplib.datasets import SegmentationDataset
from torch.utils.data import DataLoader
import torch

# Initialize model (choose one)
model = UNet(num_classes=4)  # Simple and effective
# model = DeepLabV3(num_classes=4, pretrained=True)  # Good for high-level features
# model = DeepLabV3Plus(num_classes=4, pretrained=True)  # Best performance, slower

# Prepare dataset
train_dataset = SegmentationDataset(
    root="path/to/data",
    images_dir="images",
    masks_dir="masks",
    num_classes=4,
    split="train"
)
val_dataset = SegmentationDataset(
    root="path/to/data",
    images_dir="images",
    masks_dir="masks",
    num_classes=4,
    split="val"
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Configure loss function (choose one):

# 1. Standard Cross Entropy
model.configure_loss('ce', {'ignore_index': 255})

# 2. Weighted Cross Entropy (for class imbalance)
class_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])  # Higher weight = more importance
model.configure_loss('wce', {
    'weights': class_weights,
    'ignore_index': 255
})

# 3. Dice Loss (better for imbalanced segmentation)
model.configure_loss('dice', {
    'smooth': 1.0,
    'ignore_index': 255
})

# 4. Jaccard/IoU Loss
model.configure_loss('jaccard', {
    'smooth': 1.0,
    'ignore_index': 255
})

# 5. Focal Loss (for hard examples)
model.configure_loss('focal', {
    'alpha': 0.25,
    'gamma': 2.0,
    'ignore_index': 255
})

# 6. Binary Cross Entropy (for binary segmentation)
model.configure_loss('bce', {'ignore_index': 255})

# 7. Generalized Cross Entropy (robust to noisy labels)
model.configure_loss('gce', {
    'q': 0.7,
    'ignore_index': 255
})

# 8. Combo Loss (combine multiple losses)
model.configure_loss('combo', {
    'weights': {
        'ce': 1.0,    # Cross Entropy
        'dice': 1.0,  # Dice Loss
        'focal': 0.5  # Focal Loss
    },
    'ignore_index': 255
})

# Initialize trainer with learning rate scheduler
trainer = SegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    monitor_metric='iou'  # Monitor IoU for LR scheduling
)

# Train model
history = trainer.train(
    num_epochs=100,
    save_path='best_model.pth',
    early_stopping=20
)
```

## Running the Training example

Train a segmentation model with different loss functions:
```bash
# Cross Entropy Loss
python examples/train_segmentation.py \
    --data_root ./data \
    --num_classes 4 \
    --loss_type ce \
    --loss_params '{"ignore_index": 255}' \
    --monitor_metric iou

# Dice Loss
python examples/train_segmentation.py \
    --data_root ./data \
    --num_classes 4 \
    --loss_type dice \
    --loss_params '{"smooth": 1.0, "ignore_index": 255}' \
    --monitor_metric iou

# Combo Loss (CE + Dice)
python examples/train_segmentation.py \
    --data_root ./data \
    --num_classes 4 \
    --loss_type combo \
    --loss_params '{"weights": {"ce": 1.0, "dice": 1.0}, "ignore_index": 255}' \
    --monitor_metric iou
```

## Project Structure

```
deeplib/
├── models/
│   ├── segmentation/  # ✅ Semantic segmentation models
│   ├── detection/     # 🚧 Object detection models (TODO)
│   └── anomaly/       # 🚧 Anomaly detection models (TODO)
├── trainers/          # Training logic
├── datasets/          # Dataset implementations
└── utils/            # Utility functions
```

## TODO List

### High Priority
- [ ] Implement object detection models
  - [ ] YOLOv4
  - [ ] YOLOv5
  - [ ] Faster R-CNN
- [ ] Add anomaly detection support
  - [ ] PatchCore
  - [ ] FastFlow
  - [ ] PADIM
- [ ] Add data augmentation pipeline
- [ ] Add model export (ONNX, TorchScript)

### Medium Priority
- [ ] Add more segmentation models
  - [ ] FPN
  - [ ] SegFormer
  - [ ] BEiT
- [ ] Add test suite
- [ ] Add model benchmarks
- [ ] Add visualization tools

### Low Priority
- [ ] Add multi-GPU training support
- [ ] Add quantization support
- [ ] Add model pruning
- [ ] Add hyperparameter tuning

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This library is inspired upon the following projects:
- torchvision
- anomalib
- segmentation-models-pytorch
- YOLOMIT

