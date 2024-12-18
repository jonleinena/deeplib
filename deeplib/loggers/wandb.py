from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
import wandb
import numpy as np
from PIL import Image

from .base import BaseLogger

class WandbLogger(BaseLogger):
    """Weights & Biases implementation of the experiment logger."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        """Initialize the W&B logger.
        
        Args:
            experiment_name: Name of the experiment (used as group in W&B)
            run_name: Name of the run
            tracking_uri: Not used in W&B
            artifact_location: Local directory for artifacts
            tags: Additional tags for the run
            project: W&B project name (required)
            entity: W&B entity (username or team name)
        """
        super().__init__(experiment_name, run_name, tracking_uri, artifact_location, tags)
        self.project = project or "deeplib"
        self.entity = entity
        self._active_run = False
    
    def start_run(self) -> None:
        """Start a new W&B run."""
        if self._active_run:
            return
            
        # Initialize W&B run
        wandb.init(
            project=self.project,
            name=self.run_name,
            tags=self.tags,
            dir=self.artifact_location
        )
        
        self._active_run = True
    
    def end_run(self) -> None:
        """End the current W&B run."""
        if self._active_run:
            wandb.finish()
            self._active_run = False
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters using W&B.
        
        Args:
            params: Dictionary of parameters to log
        """
        wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics using W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch number
        """
        if step is not None:
            metrics['epoch'] = step
        wandb.log(metrics)
    
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a PyTorch model using W&B.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path where to save the model
            metadata: Additional metadata to log with the model
        """
        # Create a W&B Artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            metadata=metadata
        )
        
        # Save model locally first
        save_path = Path(artifact_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)
        
        # Add file to artifact
        artifact.add_file(str(save_path))
        
        # Log artifact to W&B
        wandb.log_artifact(artifact)
        
        # Watch model for gradient and parameter logging
        wandb.watch(model)
    
    def log_image(
        self,
        image: Union[torch.Tensor, Path, str],
        title: str,
        step: Optional[int] = None,
    ) -> None:
        """Log an image using W&B.
        
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
        elif isinstance(image, (str, Path)):
            image = np.array(Image.open(str(image)))
        
        # Log image
        log_dict = {title: wandb.Image(image)}
        if step is not None:
            log_dict['epoch'] = step
        wandb.log(log_dict) 