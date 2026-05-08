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
    with open(CHECKPOINT_DIR / "label_maps.json", "r") as f:
        maps = json.load(f)
    PREDICTOR = CameraPredictor(
        brand_ckpt=CHECKPOINT_DIR / "brand_classifier" / "best.pt",
        model_ckpt_dir=CHECKPOINT_DIR / "model_classifier",
        brand_to_idx=maps["brand"],
        model_to_idx=maps["model"]
    )
except Exception as e:
    print(f"Predictor not loaded: {e}")

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse(UI_DIR / "static" / "index.html")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    temp_path = Path(f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if PREDICTOR:
        brand, model = PREDICTOR.predict(temp_path)
    else:
        brand, model = "System not trained", "System not trained"
    
    os.remove(temp_path)
    return {"brand": brand, "model": model}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
