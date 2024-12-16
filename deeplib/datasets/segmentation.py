import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple

from .base import BaseDataset


class SegmentationDataset(BaseDataset):
    """Generic dataset for semantic segmentation tasks."""
    
    def __init__(
        self,
        root: str,
        images_dir: str,
        masks_dir: str,
        num_classes: int,
        split: str = "train",
        transform: Optional[Callable] = None,
        file_extension: str = "jpg"
    ):
        """
        Args:
            root: Root directory path
            images_dir: Directory name containing images relative to root
            masks_dir: Directory name containing masks relative to root
            num_classes: Number of classes (including background)
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transform to be applied
            file_extension: Image file extension to look for
        """
        self.images_subdir = images_dir
        self.masks_subdir = masks_dir
        self.num_classes = num_classes
        self.file_extension = file_extension
        super().__init__(root=root, split=split, transform=transform)
    
    def _load_dataset(self) -> None:
        """Load dataset samples into memory."""
        images_dir = self.root / self.split / self.images_subdir
        masks_dir = self.root / self.split / self.masks_subdir
        
        if not images_dir.exists():
            raise RuntimeError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise RuntimeError(f"Masks directory not found: {masks_dir}")
        
        # Get all image files
        image_files = sorted(list(images_dir.glob(f"*.{self.file_extension}")))
        if not image_files:
            raise RuntimeError(f"No images found in {images_dir}")
        
        # Create samples list with image-mask pairs
        for img_path in image_files:
            mask_path = masks_dir / f"{img_path.stem}.png"  # Assuming masks are PNG
            if not mask_path.exists():
                raise RuntimeError(f"Mask not found for {img_path}")
            
            self.samples.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path)
            })
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as a float32 tensor with shape (H, W, 3)
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 but don't divide by 255 since normalization is handled in transformations
        image = image.astype(np.float32)
        
        return torch.from_numpy(image)
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess a mask.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Preprocessed mask as a long tensor with shape (H, W)
            Values are in range [0, num_classes-1] where:
            - 0 represents background (will be ignored in loss and metrics)
            - [1, num_classes-1] represent actual classes
        """
        # Read mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
            
        # Convert to numpy array
        mask = np.array(mask)
            
        # Verify mask values
        unique_values = np.unique(mask)
        if len(unique_values) == 1 and unique_values[0] == 0:
            raise RuntimeError(f"Mask contains only background values: {unique_values}")
        if len(unique_values) > self.num_classes:
            raise RuntimeError(
                f"Mask contains invalid values. Expected values in range [0, {self.num_classes-1}], "
                f"got values: {unique_values}"
            )
            
        return torch.from_numpy(mask).long()  # Convert to long tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a dataset sample.
        
        Returns:
            Tuple of (image, mask) where:
                - image is a float32 tensor of shape (H, W, 3) or (3, H, W) if transformed
                - mask is a long tensor of shape (H, W)
        """
        sample = self.samples[idx]
        
        # Load image and mask with proper types
        image = self._load_image(sample["image_path"])
        mask = self._load_mask(sample["mask_path"])
        
        if self.transform:
            transformed = self.transform(image=image.numpy(), mask=mask.numpy())
            image = transformed["image"]  # Will be (3, H, W) float32 tensor after ToTensorV2
            mask = torch.as_tensor(transformed["mask"], dtype=torch.long)  # Ensure long type after transform
        
        return image, mask