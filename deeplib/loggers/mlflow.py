from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow
import torch
from torch import nn
import numpy as np
from PIL import Image

from .base import BaseLogger

class MLFlowLogger(BaseLogger):
    """MLflow implementation of the experiment logger."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize the MLflow logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run within the experiment
            tracking_uri: URI for the MLflow tracking server
            artifact_location: Location to store artifacts (e.g., S3 bucket)
            tags: Additional tags to attach to the experiment
        """
        super().__init__(experiment_name, run_name, tracking_uri, artifact_location, tags)
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        # Get or create the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            kwargs = {}
            if artifact_location:
                kwargs["artifact_location"] = artifact_location
            self.experiment_id = mlflow.create_experiment(experiment_name, **kwargs)
        else:
            self.experiment_id = experiment.experiment_id
            
        self.active_run = None
    
    def start_run(self) -> None:
        """Start a new MLflow run."""
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=self.run_name,
            tags=self.tags,
            nested=True
        )
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters using MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        # Convert all values to strings for MLflow compatibility
        params = {k: str(v) if isinstance(v, (dict, list, tuple)) else v 
                 for k, v in params.items()}
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics using MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a PyTorch model using MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path where to save the model
            metadata: Additional metadata to log with the model
        """
        # Log model with PyTorch flavor
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=self.experiment_name,
            metadata=metadata
        )
    
    def log_image(
        self,
        image: Union[torch.Tensor, Path, str],
        title: str,
        step: Optional[int] = None,
    ) -> None:
        """Log an image using MLflow.
        
        Args:
            image: Image to log (can be a tensor, file path, or URL)
            title: Title/tag for the image
            step: Current step/epoch number
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            if image.ndim == 4:  # batch dimension present
                image = image[0]  # take first image from batch
            if image.ndim == 3:
                if image.shape[0] in [1, 3]:  # CHW format
                    image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
            
            # Normalize if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            image = Image.open(str(image))
        
        # Create a temporary file to save the image
        temp_path = f"{title}.png" if not step else f"{title}_{step}.png"
        image.save(temp_path)
        
        # Log the image file
        mlflow.log_artifact(temp_path)
        
        # Clean up
        Path(temp_path).unlink() 