# DeepLib

A unified PyTorch library for computer vision tasks, focusing on object detection, semantic segmentation, and anomaly detection.

## Installation

### Prerequisites

DeepLib requires PyTorch and torchvision to be installed first. For optimal performance, CUDA 11.8 or above is recommended.

You can install PyTorch with CUDA support using:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For other installation options (CPU-only, different CUDA versions), see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Installing DeepLib

#### Full Installation
Install DeepLib with all optional dependencies (recommended):
```bash
pip install deeplib
```

#### Core Installation
Install only the core functionality (no logging backends):
```bash
pip install deeplib[core]
```

## Documentation

Full documentation is available at [https://jonleinena.github.io/deeplib/](https://jonleinena.github.io/deeplib/)

## Features

- **Semantic Segmentation Models** (✅ Implemented)
  - UNet
  - DeepLabV3
  - DeepLabV3+

- **Experiment Tracking** (✅ Implemented)
  - TensorBoard
  - MLflow
  - Weights & Biases (W&B)

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

## Quick Start - Semantic Segmentation

```python
from deeplib.models.segmentation import UNet
from deeplib.trainers import SegmentationTrainer
from deeplib.datasets import SegmentationDataset
from deeplib.loggers import WandbLogger  # or TensorBoardLogger, MLFlowLogger
from torch.utils.data import DataLoader

# Initialize model
model = UNet(num_classes=4)

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

# Configure loss function (e.g., Dice Loss)
model.configure_loss('dice', {'ignore_index': 255})

# Initialize logger (choose one)
logger = WandbLogger(
    experiment_name="segmentation_experiment",
    project="deeplib-segmentation"
)
# Or use TensorBoard:
# logger = TensorBoardLogger(experiment_name="segmentation_experiment")
# Or use MLflow:
# logger = MLFlowLogger(experiment_name="segmentation_experiment")

# Initialize trainer with logger
trainer = SegmentationTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    monitor_metric='iou',  # Monitor IoU for LR scheduling
    logger=logger
)

# Train model
trainer.train(
    num_epochs=100,
    save_path='best_model.pth'
)
```

For more examples and detailed usage, check the [examples directory](examples/).

## Project Structure

```
deeplib/
├── models/
│   ├── segmentation/  # ✅ Semantic segmentation models
│   ├── detection/     # 🚧 Object detection models (TODO)
│   └── anomaly/       # 🚧 Anomaly detection models (TODO)
├── trainers/          # Training logic and utilities
├── datasets/          # Dataset implementations
├── loggers/           # ✅ Experiment tracking (TensorBoard, MLflow, W&B)
└── utils/             # Utility functions
```

## Development Roadmap

### High Priority
- [x] Implement experiment tracking
  - [x] TensorBoard support
  - [x] MLflow support
  - [x] W&B support
- [x] Add comprehensive documentation
  - [x] API Reference
  - [x] Examples
  - [x] Installation Guide
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

This library is inspired by the following projects:
- torchvision
- anomalib
- segmentation-models-pytorch
- YOLOMIT

