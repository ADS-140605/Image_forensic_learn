"""
main.py — FastAPI application entrypoint.

Run:
    cd "d:/Image_Forensic/Homogeneous Patches + CNN"
    uvicorn UI.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from UI.routers import upload, training

app = FastAPI(
    title="Camera Forensics API",
    description="Hierarchical Source Camera Model Identification",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router,   prefix="/api", tags=["prediction"])
app.include_router(training.router, prefix="/api", tags=["training"])

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    import torch
    from AI.config import CHECKPOINT_DIR
    gpu      = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else None

    # Try to read known brands from metadata
    brands, num_models = [], 0
    meta_path = CHECKPOINT_DIR / "metadata.json"
    if meta_path.exists():
        import json
        meta       = json.loads(meta_path.read_text())
        brands     = list(meta.get("brand_to_idx", {}).keys())
        num_models = len(meta.get("model_to_idx", {}))

    return {
        "status":     "ok",
        "gpu":        gpu,
        "gpu_name":   gpu_name,
        "brands":     brands,
        "num_models": num_models,
    }
