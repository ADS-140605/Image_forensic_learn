from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
import cv2
import json

from AI.inference.predictor import CameraPredictor
from AI.config import CHECKPOINT_DIR

app = FastAPI(title="Image Forensic - Camera Model Identification")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
UI_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

# Load label maps and predictor if available
PREDICTOR = None
try:
    # Load mapping files and initialize the predictor if checkpoints exist.
    with open(CHECKPOINT_DIR / "label_maps.json", "r") as f:
        maps = json.load(f)
    PREDICTOR = CameraPredictor(
        brand_ckpt=CHECKPOINT_DIR / "brand_classifier" / "best.pt",
        model_ckpt_dir=CHECKPOINT_DIR / "model_classifier",
        brand_to_idx=maps["brand"],
        model_to_idx=maps["model"]
    )
except Exception as e:
    # Predictor is optional for development; frontend shows 'Server offline' if not present.
    print(f"Predictor not loaded: {e}")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(UI_DIR / "static" / "index.html")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns a JSON object with hierarchical
    prediction details. The response mirrors the frontend expectations and
    includes vote distributions and confidence scores when available.
    """
    temp_path = Path(f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if PREDICTOR:
            # Predictor.predict now returns a dict with detailed fields.
            result = PREDICTOR.predict(temp_path)
        else:
            result = {
                'brand': 'System not trained', 'model': 'System not trained',
                'brand_confidence': 0.0, 'model_confidence': 0.0,
                'num_patches': 0, 'brand_votes': {}, 'model_votes': {}
            }
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
