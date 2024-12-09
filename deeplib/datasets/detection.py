from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseDataset


class DetectionDataset(BaseDataset):
    """Base class for object detection datasets."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(root, split, transform)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset sample with bounding boxes and labels."""
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        
        # Get bounding boxes and labels
        boxes = torch.as_tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(sample["labels"], dtype=torch.int64)
        
        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        sample = {"image": image, "target": target}
        return self._prepare_sample(sample)
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """Custom collate function for detection datasets."""
        images = [item["image"] for item in batch]
        targets = [item["target"] for item in batch]
        return images, targets 