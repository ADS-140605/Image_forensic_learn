"""
predictor.py
Inference helper for hierarchical classification.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from AI.models.convnet import CameraConvNet
from AI.dataset.patch_extractor import PatchExtractor

class CameraPredictor:
    def __init__(self, brand_ckpt, model_ckpt_dir, brand_to_idx, model_to_idx, device='cuda'):
        self.device = device
        self.brand_to_idx = brand_to_idx
        self.idx_to_brand = {i: b for b, i in brand_to_idx.items()}
        self.model_to_idx = model_to_idx
        self.idx_to_model = {i: m for m, i in model_to_idx.items()}
        
        # Load Brand Classifier
        self.brand_model = CameraConvNet(num_classes=len(brand_to_idx))
        self.brand_model.load_state_dict(torch.load(brand_ckpt, map_location=device))
        self.brand_model.to(device)
        self.brand_model.eval()
        
        # Dictionary of brand -> model classifier
        self.model_classifiers = {}
        model_ckpt_dir = Path(model_ckpt_dir)
        for brand_dir in model_ckpt_dir.iterdir():
            if brand_dir.is_dir():
                brand = brand_dir.name
                ckpt = brand_dir / "best.pt"
                if ckpt.exists():
                    # Need to know how many models for this brand
                    models_for_brand = [m for m in model_to_idx if m.startswith(brand)]
                    m_model = CameraConvNet(num_classes=len(models_for_brand))
                    m_model.load_state_dict(torch.load(ckpt, map_location=device))
                    m_model.to(device)
                    m_model.eval()
                    # Store mapping for local indices
                    local_idx_to_model = {i: m for i, m in enumerate(sorted(models_for_brand))}
                    self.model_classifiers[brand] = (m_model, local_idx_to_model)
        
        self.extractor = PatchExtractor(target_p=200)

    def predict(self, image_path):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = self.extractor.extract(img) # (N, 3, 128, 128)
        
        if len(patches) == 0:
            return "Unknown", "Unknown"
            
        patches_t = torch.from_numpy(patches).to(self.device)
        
        # 1. Brand Majority Vote
        with torch.no_grad():
            brand_logits = self.brand_model(patches_t)
            brand_preds = brand_logits.argmax(1).cpu().numpy()
            
        unique_brands, counts = np.unique(brand_preds, return_counts=True)
        pred_brand_idx = unique_brands[np.argmax(counts)]
        pred_brand = self.idx_to_brand[pred_brand_idx]
        
        # 2. Model Majority Vote
        if pred_brand in self.model_classifiers:
            m_model, local_map = self.model_classifiers[pred_brand]
            with torch.no_grad():
                model_logits = m_model(patches_t)
                model_preds = model_logits.argmax(1).cpu().numpy()
            
            unique_models, m_counts = np.unique(model_preds, return_counts=True)
            pred_model_local_idx = unique_models[np.argmax(m_counts)]
            pred_model = local_map[pred_model_local_idx]
        else:
            # Trivial case: brand only has one model
            # Find the model name for this brand
            pred_model = next((m for m in self.model_to_idx if m.startswith(pred_brand)), pred_brand)
            
        return pred_brand, pred_model
