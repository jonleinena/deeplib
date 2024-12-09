import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .detection import DetectionDataset


class COCODataset(DetectionDataset):
    """COCO Dataset for object detection."""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Any] = None,
        year: str = "2017",
    ):
        self.year = year
        super().__init__(root, split, transform)
    
    def _load_dataset(self) -> None:
        """Load COCO dataset annotations."""
        # Define paths
        split_name = "train" if self.split == "train" else "val"
        ann_file = self.root / "annotations" / f"instances_{split_name}{self.year}.json"
        img_dir = self.root / f"{split_name}{self.year}"
        
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file {ann_file} does not exist")
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory {img_dir} does not exist")
            
        # Load annotations
        with open(ann_file) as f:
            coco = json.load(f)
            
        # Create category id to continuous label mapping
        self.cat_ids = sorted([cat["id"] for cat in coco["categories"]])
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        # Create image id to filename mapping
        img_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
        
        # Group annotations by image
        img_to_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
            
        # Create samples list
        for img_id, filename in img_id_to_filename.items():
            if img_id not in img_to_anns:
                continue
                
            anns = img_to_anns[img_id]
            boxes = []
            labels = []
            
            for ann in anns:
                # Skip crowd annotations
                if ann.get("iscrowd", 0):
                    continue
                    
                # Get bbox and convert from xywh to xyxy format
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_label[ann["category_id"]])
            
            if boxes:  # Only add images with annotations
                self.samples.append({
                    "image_path": img_dir / filename,
                    "boxes": np.array(boxes, dtype=np.float32),
                    "labels": np.array(labels, dtype=np.int64)
                }) 