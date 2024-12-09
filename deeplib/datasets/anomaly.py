from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .base import BaseDataset


class AnomalyDataset(BaseDataset):
    """Base class for anomaly detection datasets."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        normal_class: Optional[int] = None,
    ):
        self.mask_transform = mask_transform
        self.normal_class = normal_class
        super().__init__(root, split, transform)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset sample with image, label, and optional mask."""
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        
        # For training, we only need normal/anomaly label
        label = torch.tensor(sample["label"], dtype=torch.float32)
        
        sample = {
            "image": image,
            "label": label,
            "image_id": idx
        }
        
        # Load mask if available (usually for validation/test)
        if "mask_path" in sample and sample["mask_path"] is not None:
            mask = self._load_anomaly_mask(sample["mask_path"])
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            sample["mask"] = mask
            
        return self._prepare_sample(sample)
    
    def _load_anomaly_mask(self, mask_path: Union[str, Image.Image]) -> torch.Tensor:
        """Load and process anomaly mask."""
        if isinstance(mask_path, str):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = mask_path
            
        # Convert to numpy array and normalize to [0, 1]
        mask = np.array(mask) / 255.0
        
        # Convert to tensor
        mask = torch.as_tensor(mask, dtype=torch.float32)
        return mask
    
    def get_normal_samples(self) -> Tuple[torch.Tensor, ...]:
        """Get all normal samples for training reconstruction-based models."""
        normal_indices = [i for i, s in enumerate(self.samples) if s["label"] == 0]
        return tuple(self[i]["image"] for i in normal_indices) 