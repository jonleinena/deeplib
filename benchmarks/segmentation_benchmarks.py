"""Benchmarking script for DeepLib segmentation models.

This script evaluates DeepLib's segmentation models against popular datasets
and compares results with published benchmarks.
"""

import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import yaml
from datetime import datetime
import logging
import argparse
from typing import Dict, Any
import sys

from deeplib.models.segmentation import UNet, DeepLabV3Plus
from deeplib.trainers import SegmentationTrainer
from deeplib.datasets import SegmentationDataset
from deeplib.metrics import iou_score, dice_score, pixel_accuracy
from deeplib.loggers import TensorBoardLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)

# Reference benchmarks from papers with citations and detailed parameters
REFERENCE_BENCHMARKS = {
    'pascal_voc': {
        'DeepLabV3Plus': {
            'mean_iou': 0.890,  # 89.0% mIoU
            'source': 'https://arxiv.org/abs/1802.02611',
            'parameters': {
                'backbone': 'Xception',
                'pretrained_on': 'ImageNet',
                'output_stride': 16,
                'learning_rate': 0.007,
                'batch_size': 16,
                'data_augmentation': 'random scaling, cropping, and horizontal flipping',
                'optimizer': 'SGD with momentum 0.9',
                'training_iterations': 90000,
                'input_size': 513  # DeepLabV3+ paper crop size
            }
        }
    },
    'cityscapes': {
        'UNet': {
            'mean_iou': 0.85,  # 85.0% mean IoU
            'accuracy': 0.96,   # 96.0% accuracy
            'mean_dice': 0.87,  # 87.0% mean DICE
            'source': 'https://www.researchgate.net/publication/379602500_Semantic_segmentation_of_urban_environments_Leveraging_U-Net_deep_learning_model_for_cityscape_image_analysis',
            'parameters': {
                'architecture': 'Encoder-Decoder with skip connections',
                'learning_rate': 0.001,
                'batch_size': 16,
                'data_augmentation': 'random rotations, scaling, and flips',
                'optimizer': 'Adam',
                'training_epochs': 50,
                'input_size': 512
            }
        },
        'DeepLabV3Plus': {
            'mean_iou': 0.821,  # 82.1% mIoU
            'source': 'https://arxiv.org/abs/1802.02611',
            'parameters': {
                'backbone': 'Xception',
                'pretrained_on': 'ImageNet',
                'output_stride': 16,
                'learning_rate': 0.007,
                'batch_size': 16,
                'data_augmentation': 'random scaling, cropping, and horizontal flipping',
                'optimizer': 'SGD with momentum 0.9',
                'training_iterations': 90000,
                'input_size': 768  # Cityscapes typical size
            }
        }
    }
}

def get_transform(train: bool = True, input_size: int = 224):
    """Get albumentations transforms matching paper implementations."""
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

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class SegmentationBenchmark:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        output_dir: str = 'benchmark_results',
        batch_size: int = 16,
        num_epochs: int = 50,
        device: str = None,
        model_name: str = None,
        ignore_index: int = 255
    ):
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device(device) if device else get_device()
        self.model_name = model_name
        self.ignore_index = ignore_index
        self.results = {}
        
        # Create output directory for benchmark results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / dataset_name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        fh = logging.FileHandler(self.output_dir / 'benchmark.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    def _get_dataset(self, model_name: str):
        """Initialize dataset based on dataset_name with appropriate augmentations."""
        ref_params = REFERENCE_BENCHMARKS[self.dataset_name][model_name]['parameters']
        input_size = ref_params.get('input_size', 224)
        
        if self.dataset_name == 'pascal_voc':
            num_classes = 21
            images_dir = 'JPEGImages'
            masks_dir = 'SegmentationClass'
        else:  # cityscapes
            num_classes = 19
            images_dir = 'leftImg8bit'
            masks_dir = 'gtFine'
        
        train_dataset = SegmentationDataset(
            root=str(self.data_dir),
            images_dir=images_dir,
            masks_dir=masks_dir,
            num_classes=num_classes,
            split="train",
            transform=get_transform(train=True, input_size=input_size)
        )
        
        val_dataset = SegmentationDataset(
            root=str(self.data_dir),
            images_dir=images_dir,
            masks_dir=masks_dir,
            num_classes=num_classes,
            split="val",
            transform=get_transform(train=False, input_size=input_size)
        )
        
        return train_dataset, val_dataset, num_classes

    def _configure_model(self, model_name: str, num_classes: int):
        """Configure model and training parameters based on reference implementation."""
        if model_name == 'UNet':
            model = UNet(num_classes=num_classes, dropout_p=0.1)
            model.configure_loss('dice', {'ignore_index': self.ignore_index})
        else:  # DeepLabV3Plus
            model = DeepLabV3Plus(num_classes=num_classes, pretrained=True)
            model.configure_loss('ce', {'ignore_index': self.ignore_index})
        
        return model

    def _configure_optimizer(self, model: torch.nn.Module, model_name: str):
        """Configure optimizer and scheduler based on reference implementation."""
        ref_params = REFERENCE_BENCHMARKS[self.dataset_name][model_name]['parameters']
        
        if model_name == 'DeepLabV3Plus':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=ref_params['learning_rate'],
                momentum=0.9,
                weight_decay=4e-5
            )
        else:  # UNet
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=ref_params['learning_rate']
            )
        
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        
        return optimizer, scheduler

    def benchmark_model(self, model_name: str = None) -> Dict[str, Any]:
        """Run benchmark for a specific model."""
        model_name = model_name or self.model_name
        if not model_name:
            raise ValueError("Model name must be specified")
            
        logger.info(f"Starting benchmark for {model_name} on {self.dataset_name}")
        
        if model_name not in REFERENCE_BENCHMARKS.get(self.dataset_name, {}):
            logger.warning(f"No reference benchmark available for {model_name} on {self.dataset_name}")
            return None
        
        # Setup data
        train_dataset, val_dataset, num_classes = self._get_dataset(model_name)
        
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True if self.device.type in ["cuda", "mps"] else False
        )
        
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True if self.device.type in ["cuda", "mps"] else False
        )
        
        # Initialize model and training components
        model = self._configure_model(model_name, num_classes)
        optimizer, scheduler = self._configure_optimizer(model, model_name)
        
        # Setup metrics
        custom_metrics = [
            lambda x, y: iou_score(x, y, num_classes, self.ignore_index),
            lambda x, y: dice_score(x, y, num_classes, self.ignore_index),
            lambda x, y: pixel_accuracy(x, y, self.ignore_index)
        ]
        
        # Configure training
        experiment_name = f"{model_name}_{self.dataset_name}"
        logger_dir = self.output_dir / 'tensorboard' / experiment_name
        tb_logger = TensorBoardLogger(log_dir=str(logger_dir))
        
        # Initialize trainer
        trainer = SegmentationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            metrics=custom_metrics,
            ignore_index=self.ignore_index,
            monitor_metric='iou',
            logger=tb_logger
        )
        
        # Train and evaluate
        save_path = self.output_dir / "checkpoints" / f"{model_name}_benchmark.pth"
        save_path.parent.mkdir(exist_ok=True)
        
        trainer.train(
            num_epochs=self.num_epochs,
            save_path=str(save_path)
        )
        
        # Get final metrics
        metrics = trainer.evaluate()
        
        # Compare with reference benchmarks
        reference = REFERENCE_BENCHMARKS[self.dataset_name][model_name]
        
        results = {
            'model': model_name,
            'dataset': self.dataset_name,
            'achieved_metrics': metrics,
            'reference_metrics': {
                'mean_iou': reference['mean_iou'],
                'source': reference['source']
            },
            'reference_parameters': reference['parameters'],
            'difference': {
                'mean_iou': metrics.get('mean_iou', 0) - reference['mean_iou']
            }
        }
        
        # Add additional metrics if available
        if 'accuracy' in reference:
            results['reference_metrics']['accuracy'] = reference['accuracy']
            results['difference']['accuracy'] = metrics.get('accuracy', 0) - reference['accuracy']
        if 'mean_dice' in reference:
            results['reference_metrics']['mean_dice'] = reference['mean_dice']
            results['difference']['mean_dice'] = metrics.get('mean_dice', 0) - reference['mean_dice']
        
        # Save results to YAML
        results_yaml = {
            'model': model_name,
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'achieved': metrics,
                'reference': reference,
                'difference': results['difference']
            },
            'parameters': reference['parameters'],
            'source': reference['source']
        }
        
        with open(self.output_dir / 'benchmark_summary.yaml', 'a') as f:
            yaml.dump({model_name: results_yaml}, f, default_flow_style=False)
        
        logger.info(f"Completed benchmark for {model_name}")
        logger.info(f"Results saved to {self.output_dir}/benchmark_summary.yaml")
        logger.info(f"TensorBoard logs available in {self.output_dir}/tensorboard")
        
        return results

    def run_all_benchmarks(self):
        """Run benchmarks for all supported models with available references."""
        if self.model_name:
            self.results[self.model_name] = self.benchmark_model(self.model_name)
        else:
            available_models = list(REFERENCE_BENCHMARKS.get(self.dataset_name, {}).keys())
            if not available_models:
                logger.warning(f"No reference benchmarks available for dataset: {self.dataset_name}")
                return {}
            
            for model_name in available_models:
                self.results[model_name] = self.benchmark_model(model_name)
        
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Run segmentation model benchmarks')
    parser.add_argument('--dataset', type=str, required=True, choices=['pascal_voc', 'cityscapes'],
                      help='Dataset to benchmark on')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                      help='Directory to save benchmark results')
    parser.add_argument('--model', type=str, choices=['UNet', 'DeepLabV3Plus'],
                      help='Specific model to benchmark (optional)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                      help='Device to run on (default: best available)')
    parser.add_argument('--ignore_index', type=int, default=255,
                      help='Index to ignore in loss calculation')
    
    args = parser.parse_args()
    
    benchmark = SegmentationBenchmark(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        model_name=args.model,
        ignore_index=args.ignore_index
    )
    
    results = benchmark.run_all_benchmarks()
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)
    for model_name, result in results.items():
        if result:
            print(f"\n{model_name} Results:")
            print(f"Mean IoU: {result['achieved_metrics']['mean_iou']:.3f}")
            print(f"Reference: {result['reference_metrics']['mean_iou']:.3f}")
            print(f"Difference: {result['difference']['mean_iou']:.3f}")
    print("\nDetailed results saved in benchmark_summary.yaml")
    print(f"TensorBoard logs available in {args.output_dir}/tensorboard")

if __name__ == "__main__":
    main() 