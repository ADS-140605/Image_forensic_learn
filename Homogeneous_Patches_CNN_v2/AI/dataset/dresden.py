"""
dresden.py
Dresden dataset wrapper.
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from AI.config import SEED
from AI.dataset.patch_extractor import PatchExtractor

class DresdenDataset(Dataset):
    def __init__(self, records, brand_to_idx, model_to_idx, level='brand', 
                 patches_per_img=200, transform=None):
        self.records = records
        self.brand_to_idx = brand_to_idx
        self.model_to_idx = model_to_idx
        self.level = level
        self.patches_per_img = patches_per_img
        self.transform = transform
        self.extractor = PatchExtractor(target_p=patches_per_img, seed=SEED)
        
        self.cache_path = None
        self.cache_patches = None

    def __len__(self):
        return len(self.records) * self.patches_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img
        patch_idx = idx % self.patches_per_img
        
        path, brand, model, device = self.records[img_idx]
        
        if self.cache_path != path:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cache_patches = self.extractor.extract(img)
            self.cache_path = path
            
        # Handle cases where fewer than target_p patches were extracted
        actual_patch_idx = patch_idx % max(len(self.cache_patches), 1)
        patch = self.cache_patches[actual_patch_idx]
        
        label = self.brand_to_idx[brand] if self.level == 'brand' else self.model_to_idx[model]
        
        if self.transform:
            patch = self.transform(patch)
            
        return patch, label

def build_label_maps(dataset_dir):
    dataset_dir = Path(dataset_dir)
    records = []
    brands = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    brand_to_idx = {b: i for i, b in enumerate(brands)}
    
    models = []
    for brand in brands:
        brand_dir = dataset_dir / brand
        brand_models = sorted([d.name for d in brand_dir.iterdir() if d.is_dir()])
        for m in brand_models:
            full_model_name = f"{brand}_{m}"
            models.append(full_model_name)
            model_dir = brand_dir / m
            # Handle device subfolders if they exist
            sub_items = list(model_dir.iterdir())
            if any(s.is_dir() for s in sub_items):
                for dev_dir in sorted([s for s in sub_items if s.is_dir()]):
                    for img in dev_dir.iterdir():
                        if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif'):
                            records.append((img, brand, full_model_name, dev_dir.name))
            else:
                for img in model_dir.iterdir():
                    if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif'):
                        records.append((img, brand, full_model_name, "default"))
                        
    model_to_idx = {m: i for i, m in enumerate(sorted(list(set(models))))}
    return brand_to_idx, model_to_idx, records
