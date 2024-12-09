from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Base class for all datasets."""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Args:
            root: Path to dataset root
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to data
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist")
            
        self._load_dataset()
    
    @abstractmethod
    def _load_dataset(self) -> None:
        """Load dataset samples into memory."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a dataset sample."""
        pass
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load an image from disk using OpenCV."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare a sample for training/inference."""
        if self.transform is not None:
            sample = self.transform(**sample)
        return sample 