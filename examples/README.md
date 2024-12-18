# DeepLib Examples

This directory contains example scripts demonstrating how to use DeepLib for various deep learning tasks.

## Prerequisites

Before running any example, make sure you have installed DeepLib and its dependencies:

```bash
# Install PyTorch and torchvision first (required)
pip install torch torchvision

# Install DeepLib
pip install deeplib  # for latest release
# or
pip install -e .    # for development installation
```

## Available Examples

### Semantic Segmentation (`train_segmentation.py`)

A complete example showing how to train a UNet model for image segmentation tasks.

#### Dataset Structure

Your dataset should be organized as follows:
```
data_root/
├── images/
│   ├── train/
│   └── val/
├── masks/
│   ├── train/
│   └── val/
```

#### Basic Usage

```bash
python train_segmentation.py \
    --data_root ./data/segmentation \
    --num_classes 3
```

#### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_root` | Root directory containing the dataset | Required |
| `--images_dir` | Directory name containing images | "images" |
| `--masks_dir` | Directory name containing masks | "masks" |
| `--num_classes` | Number of segmentation classes | Required |
| `--num_epochs` | Number of training epochs | 50 |
| `--batch_size` | Batch size for training | 64 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--input_size` | Input image size | 192 |
| `--ignore_index` | Index to ignore in loss calculation | 255 |
| `--dropout_p` | Dropout probability | 0.1 |

#### Loss Functions

Choose from multiple loss functions using the `--loss` argument:
- `ce`: Cross Entropy Loss (default)
- `dice`: Dice Loss
- `wce`: Weighted Cross Entropy Loss
- `jaccard`: IoU Loss
- `focal`: Focal Loss

Example with Dice loss:
```bash
python train_segmentation.py \
    --data_root ./data/segmentation \
    --num_classes 3 \
    --loss dice
```

#### Experiment Tracking

Track your experiments using different loggers with the `--logger` argument:
- `tensorboard`: TensorBoard logging (default)
- `mlflow`: MLflow logging
- `wandb`: Weights & Biases logging
- `none`: No logging

Example with W&B logging:
```bash
python train_segmentation.py \
    --data_root ./data/segmentation \
    --num_classes 3 \
    --logger wandb
``` 