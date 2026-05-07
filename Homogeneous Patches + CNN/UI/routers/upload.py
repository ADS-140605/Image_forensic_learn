"""
upload.py — POST /api/predict
Accepts an uploaded image file, runs hierarchical inference, returns prediction.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from UI.schemas import PredictionResponse

router = APIRouter()

# Lazy-load predictor once (avoids reloading weights on every request)
_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from AI.inference.predictor import HierarchicalPredictor
        from AI.config import CHECKPOINT_DIR
        meta = CHECKPOINT_DIR / "metadata.json"
        if not meta.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained weights found. "
                    "Run 'python train.py' first or load a checkpoint."
                ),
            )
        _predictor = HierarchicalPredictor()
    return _predictor


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image (JPEG / PNG / TIFF) and receive the hierarchical
    source camera brand and model prediction.
    """
    if file.content_type not in (
        "image/jpeg", "image/png", "image/tiff",
        "image/jpg", "image/bmp", "image/webp",
    ):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    data  = await file.read()
    try:
        pil   = Image.open(io.BytesIO(data)).convert("RGB")
        image = np.array(pil)                   # (H, W, 3) uint8
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {e}")

    predictor = _get_predictor()
    result    = predictor.predict(image)

    # Compute human-readable confidence scores
    total_brand = max(sum(result["brand_votes"].values()), 1)
    total_model = max(sum(result["model_votes"].values()), 1)
    brand_conf  = result["brand_votes"].get(result["brand"], 0) / total_brand
    model_conf  = (result["model_votes"].get(result["model"], 0) / total_model
                   if result["model"] else 0.0)

    return PredictionResponse(
        brand             = result["brand"],
        model             = result["model"],
        num_patches       = result["num_patches"],
        brand_votes       = result["brand_votes"],
        model_votes       = result["model_votes"],
        brand_probs       = result["brand_probs"],
        model_probs       = result["model_probs"],
        brand_confidence  = round(brand_conf, 4),
        model_confidence  = round(model_conf, 4),
    )
