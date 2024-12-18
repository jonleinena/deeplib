from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

from .base import BaseLogger

class TensorBoardLogger(BaseLogger):
    """TensorBoard implementation of the experiment logger."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize the TensorBoard logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run within the experiment
            tracking_uri: Not used in TensorBoard
            artifact_location: Root directory for TensorBoard logs
            tags: Additional tags (added as text in TensorBoard)
        """
        super().__init__(experiment_name, run_name, tracking_uri, artifact_location, tags)
        
        # Set up log directory
        self.log_dir = Path(artifact_location or "runs") / experiment_name
        if run_name:
            self.log_dir = self.log_dir / run_name
            
        self.writer = None
    
    def start_run(self) -> None:
        """Start a new TensorBoard run by creating a SummaryWriter."""
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Log tags as text
        if self.tags:
            self.writer.add_text(
                "tags",
                "\n".join([f"{k}: {v}" for k, v in self.tags.items()])
            )
    
    def end_run(self) -> None:
        """End the current TensorBoard run."""
        if self.writer:
            self.writer.close()
            self.writer = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters using TensorBoard.
        
        Args:
            params: Dictionary of parameters to log
        """
        # TensorBoard doesn't have a direct parameter logging feature
        # We'll add them as text
        self.writer.add_text(
            "parameters",
            "\n".join([f"{k}: {v}" for k, v in params.items()])
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics using TensorBoard.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch number
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_model(
        self,
        model: nn.Module,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a PyTorch model using TensorBoard.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path where to save the model
            metadata: Additional metadata to log with the model
        """
        # TensorBoard can't directly log models, so we'll save it to disk
        save_path = self.log_dir / artifact_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)
        
        # Add model graph if possible (requires a forward pass)
        try:
            # Assuming a standard input size for visualization
            dummy_input = torch.randn(1, 3, 224, 224)
            self.writer.add_graph(model, dummy_input)
        except Exception:
            # Skip graph logging if it fails
            pass
    
    def log_image(
        self,
        image: Union[torch.Tensor, Path, str],
        title: str,
        step: Optional[int] = None,
    ) -> None:
        """Log an image using TensorBoard.
        
        Args:
            image: Image to log (can be a tensor, file path, or URL)
            title: Title/tag for the image
            step: Current step/epoch number
        """
        if isinstance(image, (str, Path)):
            image = Image.open(str(image))
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:  # batch dimension present
                image = image[0]  # take first image from batch
            if image.ndim == 3 and image.shape[0] not in [1, 3]:  # HWC format
                image = image.permute(2, 0, 1)  # convert to CHW
                
        # Ensure the image is on CPU and detached from computation graph
        image = image.detach().cpu()
        
        # Add the image to TensorBoard
        self.writer.add_image(title, image, step) 