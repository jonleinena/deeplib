# DeepLib Benchmarks

This directory contains benchmarking scripts to evaluate DeepLib's models against popular datasets and compare them with published results.

## Available Benchmarks

Currently supported benchmarks:

- Semantic Segmentation Models (UNet, DeepLabV3Plus)
  - Pascal VOC 2012
  - Cityscapes

## Running Benchmarks

### Prerequisites

Make sure you have DeepLib installed with all dependencies:

```bash
pip install -e .[all]  # Install from root directory with all extras
```

### Running Segmentation Benchmarks

1. **Pascal VOC 2012**:
```bash
python segmentation_benchmarks.py --dataset pascal_voc --data_dir data/pascal_voc
```

2. **Cityscapes**:
```bash
python segmentation_benchmarks.py --dataset cityscapes --data_dir data/cityscapes
```

## Benchmark Results

Results are saved in the `benchmark_results` directory with the following structure:

```
benchmark_results/
├── pascal_voc/
│   └── YYYYMMDD_HHMMSS/
│       ├── DeepLabV3Plus_results.json
│       └── all_results.json
└── cityscapes/
    └── YYYYMMDD_HHMMSS/
        ├── UNet_results.json
        ├── DeepLabV3Plus_results.json
        └── all_results.json
```

Each JSON file contains:
- Achieved metrics
- Reference metrics from published papers (with citations)
- Training parameters used
- Difference between achieved and reference metrics

## Reference Benchmarks

The reference benchmarks are sourced from the following papers:

### Pascal VOC 2012

| Model | Mean IoU | Source |
|-------|----------|---------|
| DeepLabV3+ | 89.0% | [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611) |

Training parameters for DeepLabV3+:
- Backbone: Xception
- Pretrained on: ImageNet
- Output stride: 16
- Learning rate: 0.007 with polynomial decay
- Batch size: 16
- Data augmentation: random scaling, cropping, and horizontal flipping
- Optimizer: SGD with momentum 0.9
- Training iterations: 90,000

### Cityscapes

| Model | Mean IoU | Accuracy | Mean DICE | Source |
|-------|----------|----------|-----------|---------|
| U-Net | 85.0% | 96.0% | 87.0% | [Semantic segmentation of urban environments](https://www.researchgate.net/publication/379602500_Semantic_segmentation_of_urban_environments_Leveraging_U-Net_deep_learning_model_for_cityscape_image_analysis) |
| DeepLabV3+ | 82.1% | - | - | [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611) |

Training parameters for U-Net:
- Architecture: Encoder-Decoder with skip connections
- Learning rate: 0.001
- Batch size: 16
- Data augmentation: random rotations, scaling, and flips
- Optimizer: Adam
- Training epochs: 50

Training parameters for DeepLabV3+:
- Backbone: Xception
- Pretrained on: ImageNet
- Output stride: 16
- Learning rate: 0.007 with polynomial decay
- Batch size: 16
- Data augmentation: random scaling, cropping, and horizontal flipping
- Optimizer: SGD with momentum 0.9
- Training iterations: 90,000

## Adding New Benchmarks

To add support for new datasets or models:

1. Add the dataset handling logic in `_get_dataset()` method
2. Add reference benchmarks in the `REFERENCE_BENCHMARKS` dictionary (with proper citations and training parameters)
3. Update the model initialization in `_get_model()` if adding new model architectures