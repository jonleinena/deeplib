import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from deeplib.loggers import BaseLogger, MLFlowLogger, TensorBoardLogger, WandbLogger

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

@pytest.fixture
def test_data():
    """Create test data for loggers."""
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Conv2d(3, 1, kernel_size=3, padding=1)
    )
    
    # Create a random tensor for image testing
    image = torch.randn(3, 32, 32)
    
    return {
        "model": model,
        "image": image,
        "params": {"batch_size": 32, "lr": 0.001},
        "metrics": {"loss": 0.5, "accuracy": 0.95}
    }

@pytest.fixture
def cleanup_artifacts():
    # Setup - nothing to do
    yield
    
    # Cleanup after tests
    paths_to_clean = [
        "runs",
        "mlruns",
        "wandb",
        "test_artifacts"
    ]
    for path in paths_to_clean:
        if os.path.exists(path):
            shutil.rmtree(path)

def test_tensorboard_logger(test_data, cleanup_artifacts):
    logger = TensorBoardLogger(
        experiment_name="test_experiment",
        run_name="test_run",
        artifact_location="test_artifacts"
    )
    
    with logger:
        # Test parameter logging
        logger.log_params(test_data["params"])
        
        # Test metrics logging
        logger.log_metrics(test_data["metrics"], step=0)
        
        # Test model logging
        logger.log_model(test_data["model"], "model.pt")
        
        # Test image logging
        logger.log_image(test_data["image"], "test_image", step=0)
    
    # Verify files were created
    log_dir = Path("test_artifacts") / "test_experiment"
    if "test_run" in str(logger.log_dir):
        log_dir = log_dir / "test_run"
    
    assert log_dir.exists()
    assert any(log_dir.glob("events.out.tfevents.*"))

def test_mlflow_logger(test_data, cleanup_artifacts):
    # Set MLflow to use local directory
    logger = MLFlowLogger(
        experiment_name="test_experiment",
        run_name="test_run",
        tracking_uri="file:./mlruns"
    )

    with logger:
        # Test parameter logging
        logger.log_params(test_data["params"])
        
        # Test metrics logging
        logger.log_metrics(test_data["metrics"], step=0)
        
        # Test model logging
        logger.log_model(test_data["model"], "model.pth")
        
        # Test image logging
        logger.log_image(test_data["image"], "test_image", step=0)
    
    # Verify files were created
    mlruns_dir = Path("mlruns")
    assert mlruns_dir.exists(), "MLflow directory not created"
    
    # Use recursive glob to find files anywhere in the directory tree
    model_files = list(mlruns_dir.rglob("model.pth"))
    assert len(model_files) > 0, "No model file found in MLflow directory"
    
    image_files = list(mlruns_dir.rglob("test_image*.png"))
    assert len(image_files) > 0, "No image file found in MLflow directory"
    
    metric_files = list(mlruns_dir.rglob("metrics/*"))
    assert len(metric_files) > 0, "No metrics file found in MLflow directory"
    
    param_files = list(mlruns_dir.rglob("params/*"))
    assert len(param_files) > 0, "No params file found in MLflow directory"

# @patch('wandb.init')
# @patch('wandb.finish')
# @patch('wandb.config')
# @patch('wandb.log')
# @patch('wandb.Artifact')
# @patch('wandb.Image')
# def test_wandb_logger(mock_image, mock_artifact, mock_log, mock_config,
#                      mock_finish, mock_init, test_data, cleanup_artifacts):
#     logger = WandbLogger(
#         experiment_name="test_experiment",
#         run_name="test_run",
#         project="test_project"
#     )
    
#     with logger:
#         logger.log_params(test_data["params"])
#         logger.log_metrics(test_data["metrics"], step=0)
#         logger.log_model(test_data["model"], "model.pt")
#         logger.log_image(test_data["image"], "test_image", step=0)
    
#     # Verify W&B functions were called
#     mock_init.assert_called_once()
#     assert mock_config.update.called  # Called for params
#     assert mock_log.called  # Called for metrics and images
#     assert mock_artifact.called  # Called for model logging
#     mock_finish.assert_called_once() 