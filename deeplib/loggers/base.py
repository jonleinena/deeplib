from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

class BaseLogger(ABC):
    """Base class for all experiment loggers."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize the logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run within the experiment
            tracking_uri: URI for the tracking server (e.g., MLflow server, W&B server)
            artifact_location: Location to store artifacts (e.g., S3 bucket, local path)
            tags: Additional tags to attach to the experiment
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.tags = tags or {}
        self._active_run = False
        
    def __enter__(self):
        """Context manager entry."""
        if not self._active_run:
            self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._active_run:
            self.end_run()
    
    @abstractmethod
    def start_run(self) -> None:
        """Start a new run."""
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        """End the current run."""
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for the run.
        
        Args:
            params: Dictionary of parameters to log
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for the run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch number
        """
        pass
    
    @abstractmethod
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a model artifact.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path where to save the model
            metadata: Additional metadata to log with the model
        """
        pass
    
    @abstractmethod
    def log_image(
        self,
        image: Union[torch.Tensor, Path, str],
        title: str,
        step: Optional[int] = None,
    ) -> None:
        """Log an image.
        
        Args:
            image: Image to log (can be a tensor, file path, or URL)
            title: Title/tag for the image
            step: Current step/epoch number
        """
        pass
