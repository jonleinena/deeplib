import glob
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from .segmentation import SegmentationDataset


class NEUSegDataset(SegmentationDataset):
    """NEU-SEG Dataset for surface defect segmentation."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        super().__init__(root, split, transform, num_classes=3)  # 3 classes: background + 2 defect types
    
    def _load_dataset(self) -> None:
        """Load NEU-SEG dataset images and masks."""
        split_dir = self.root / self.split
        
        # Get all image files
        image_files = sorted(glob.glob(str(split_dir / "images" / "*.jpg")))
        mask_files = sorted(glob.glob(str(split_dir / "masks" / "*.png")))
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {split_dir / 'images'}")
        if not mask_files:
            raise FileNotFoundError(f"No masks found in {split_dir / 'masks'}")
        
        if len(image_files) != len(mask_files):
            raise ValueError(f"Number of images ({len(image_files)}) does not match number of masks ({len(mask_files)})")
        
        # Create samples list
        for image_path, mask_path in zip(image_files, mask_files):
            self.samples.append({
                "image_path": image_path,
                "mask_path": mask_path
            })
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and process segmentation mask."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
            
        # Convert grayscale values to class indices
        # Assuming: 0 = background, 128 = class 1, 255 = class 2
        mask = np.where(mask > 192, 2, np.where(mask > 64, 1, 0))
        return torch.as_tensor(mask, dtype=torch.long) 