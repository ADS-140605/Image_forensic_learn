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
        """
        Performs hierarchical inference on `image_path` and returns a rich
        result dictionary containing brand/model names, confidence scores,
        vote distributions and the number of patches analysed.

        Returned keys:
          - brand: predicted brand name (str)
          - model: predicted model name (str)
          - brand_confidence: fraction of patches that voted for the brand (0..1)
          - model_confidence: fraction of patches that voted for the model (0..1)
          - num_patches: number of patches analysed (int)
          - brand_votes: dict mapping brand -> vote count
          - model_votes: dict mapping model -> vote count
        """
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = self.extractor.extract(img)  # (N, 3, 128, 128)

        num_patches = len(patches)
        if num_patches == 0:
            return {
                'brand': 'Unknown', 'model': 'Unknown',
                'brand_confidence': 0.0, 'model_confidence': 0.0,
                'num_patches': 0, 'brand_votes': {}, 'model_votes': {}
            }

        patches_t = torch.from_numpy(patches).to(self.device)

        # 1. Brand logits -> per-patch brand predictions
        with torch.no_grad():
            brand_logits = self.brand_model(patches_t)
            brand_preds = brand_logits.argmax(1).cpu().numpy()

        unique_brands, counts = np.unique(brand_preds, return_counts=True)
        # Map predicted indices to names and build vote dict
        brand_votes = {self.idx_to_brand[int(idx)]: int(cnt) for idx, cnt in zip(unique_brands, counts)}
        # Choose brand with most votes
        top_brand_idx = unique_brands[np.argmax(counts)]
        pred_brand = self.idx_to_brand[int(top_brand_idx)]
        brand_confidence = float(np.max(counts) / num_patches)

        # 2. If we have a model classifier for the predicted brand, run it
        model_votes = {}
        model_confidence = 0.0
        pred_model = None

        if pred_brand in self.model_classifiers:
            m_model, local_map = self.model_classifiers[pred_brand]
            with torch.no_grad():
                model_logits = m_model(patches_t)
                model_preds = model_logits.argmax(1).cpu().numpy()

            unique_models, m_counts = np.unique(model_preds, return_counts=True)
            # local_map maps local idx -> global model name
            model_votes = {local_map[int(idx)]: int(cnt) for idx, cnt in zip(unique_models, m_counts)}
            top_model_local_idx = unique_models[np.argmax(m_counts)]
            pred_model = local_map[int(top_model_local_idx)]
            model_confidence = float(np.max(m_counts) / num_patches)
        else:
            # Brand-level classifier only; try to pick the canonical model name
            candidate = next((m for m in self.model_to_idx if m.startswith(pred_brand)), None)
            pred_model = candidate or pred_brand
            model_votes = {pred_model: num_patches}
            model_confidence = 1.0 if num_patches > 0 else 0.0

        return {
            'brand': pred_brand,
            'model': pred_model,
            'brand_confidence': brand_confidence,
            'model_confidence': model_confidence,
            'num_patches': num_patches,
            'brand_votes': brand_votes,
            'model_votes': model_votes,
        }
