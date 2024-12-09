# DeepLib

A unified PyTorch library for computer vision tasks, focusing on object detection, semantic segmentation, and anomaly detection.

## Features

- **Object Detection Models**
  - YOLOv4
  - YOLOv5 (GPL3.0 implementation)
  - YOLOX
  - YOLOv7 and YOLOv9 (from YOLOMIT)
  - Faster R-CNN

- **Semantic Segmentation Models**
  - UNet
  - DeepLabV3+
  - FPN
  - SegFormer
  - BEiT

- **Anomaly Detection Models**
  - Implementations from anomalib

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from deeplib.models import YOLOv5
from deeplib.trainers import DetectionTrainer
from deeplib.datasets import COCODataset

# Initialize model
model = YOLOv5(num_classes=80)

# Prepare dataset
train_dataset = COCODataset(root="path/to/coco", split="train")
val_dataset = COCODataset(root="path/to/coco", split="val")

# Initialize trainer
trainer = DetectionTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

# Train model
history = trainer.train(num_epochs=100)
```

## Project Structure

```
deeplib/
├── models/
│   ├── detection/     # Object detection models
│   ├── segmentation/  # Semantic segmentation models
│   └── anomaly/       # Anomaly detection models
├── trainers/          # Training logic
├── datasets/          # Dataset implementations
└── utils/            # Utility functions
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This library incorporates code from various open-source projects:
- YOLOv5 (GPL3.0)
- YOLOMIT
- anomalib 