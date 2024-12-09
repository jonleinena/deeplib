from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

from .base import BaseDataset


class SegmentationDataset(BaseDataset):
    """Base class for semantic segmentation datasets."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        ignore_index: int = 255,
        num_classes: int = 21,
    ):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        super().__init__(root, split, transform)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset sample with image and mask."""
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        mask = self._load_mask(sample["mask_path"])
        
        sample = {
            "image": image,
            "mask": mask,
            "image_id": idx
        }
        return self._prepare_sample(sample)
    
    def _load_mask(self, mask_path: Union[str, Image.Image]) -> torch.Tensor:
        """Load and process segmentation mask."""
        if isinstance(mask_path, str):
            mask = Image.open(mask_path)
        else:
            mask = mask_path
            
        # Convert PIL image to numpy array
        mask = np.array(mask)
        
        # Handle different mask formats (RGB, L, etc.)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel if RGB
            
        # Convert to tensor
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        # Verify mask values
        assert mask.max() < self.num_classes, f"Invalid mask values. Expected max value < {self.num_classes}, got {mask.max()}"
        return mask 