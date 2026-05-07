"""
training.py — POST /api/train  |  GET /api/train/status/{job_id}
Starts training jobs in a background thread and streams log updates.
"""
from __future__ import annotations

import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from UI.schemas import TrainRequest, TrainResponse, TrainStatusResponse

router  = APIRouter()

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: Dict[str, dict] = {}   # job_id → {status, level, epoch, log, thread}
_MAX_LOG_LINES = 200


def _run_training_job(job_id: str, request: TrainRequest):
    """Execute in a daemon thread; captures print output into the job log."""
    log: Deque[str] = _jobs[job_id]["log"]

    def _emit(msg: str):
        log.append(msg)
        if len(log) > _MAX_LOG_LINES:
            log.popleft()

    try:
        _jobs[job_id]["status"] = "running"
        from AI.training.hierarchical import HierarchicalTrainer
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _emit(f"[job {job_id}] Using device: {device}")

        trainer = HierarchicalTrainer(device=device)

        if request.level == "all":
            trainer.train_all(max_epochs=request.max_epochs)

        elif request.level == "brand":
            trainer.train_brand_level(max_epochs=request.max_epochs)
            trainer.save_metadata()

        elif request.level == "model":
            if not request.brand:
                raise ValueError("brand is required for level='model'")
            trainer.train_model_level(request.brand, max_epochs=request.max_epochs)
            trainer.save_metadata()

        _jobs[job_id]["status"] = "done"
        _emit(f"[job {job_id}] Training complete.")

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _emit(f"[job {job_id}] ERROR: {exc}")


@router.post("/train", response_model=TrainResponse)
async def start_training(request: TrainRequest):
    """
    Kick off a training job in the background.
    Returns a job_id to poll for status.
    """
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "level":  request.level,
        "epoch":  None,
        "log":    deque(maxlen=_MAX_LOG_LINES),
    }
    t = threading.Thread(
        target=_run_training_job,
        args=(job_id, request),
        daemon=True,
    )
    t.start()
    _jobs[job_id]["thread"] = t
    return TrainResponse(
        job_id=job_id,
        message=f"Training job '{job_id}' started (level={request.level}).",
    )


@router.get("/train/status/{job_id}", response_model=TrainStatusResponse)
async def get_status(job_id: str):
    """Poll training job status and get recent log lines."""
    if job_id not in _jobs:
        return TrainStatusResponse(
            job_id=job_id, status="not_found",
            level=None, epoch=None, message=None,
        )
    job = _jobs[job_id]
    return TrainStatusResponse(
        job_id   = job_id,
        status   = job["status"],
        level    = job.get("level"),
        epoch    = job.get("epoch"),
        message  = None,
        log_tail = list(job["log"])[-30:],
    )
