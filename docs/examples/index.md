# Examples

## Image Segmentation Training

The `train_segmentation.py` script demonstrates how to train a UNet model for image segmentation tasks. This example showcases DeepLib's capabilities for handling semantic segmentation problems with various customization options.

### Prerequisites

- PyTorch and torchvision installed
- A dataset organized with the following structure:
  ```
  data_root/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── masks/
  │   ├── train/
  │   └── val/
  ```

### Available Options

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
| `--device` | Device to use (cuda, mps, or cpu) | Best available |
| `--monitor_metric` | Metric to monitor for early stopping | "iou" |
| `--loss` | Loss function | "dice" |
| `--logger` | Logger to use | "tensorboard" |

### Loss Functions

The example supports multiple loss functions:
- `ce`: Cross Entropy Loss
- `dice`: Dice Loss
- `wce`: Weighted Cross Entropy Loss
- `jaccard`: IoU Loss
- `focal`: Focal Loss

### Logging Options

Choose from multiple logging backends:
- `tensorboard`: TensorBoard logging (default)
- `mlflow`: MLflow logging
- `wandb`: Weights & Biases logging
- `none`: No logging

### Example Usage

Basic usage:
```bash
python examples/train_segmentation.py --data_root ./data/segmentation --num_classes 3
```

Advanced usage with custom parameters:
```bash
python examples/train_segmentation.py \
    --data_root ./data/segmentation \
    --num_classes 3 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --input_size 256 \
    --loss focal \
    --logger wandb
```

### Features

- Automatic device selection (CUDA, MPS, or CPU)
- Multiple loss functions
- Various logging backends
- Learning rate scheduling
- Data augmentation
- Early stopping based on monitored metrics
- Model checkpointing 