"""
schemas.py — Pydantic models for request/response contracts.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    brand:        str
    model:        Optional[str]  = None
    num_patches:  int
    brand_votes:  Dict[str, int]
    model_votes:  Dict[str, int] = Field(default_factory=dict)
    brand_probs:  List[float]    = Field(default_factory=list)
    model_probs:  List[float]    = Field(default_factory=list)
    # human-readable confidence (winning brand vote / total patches)
    brand_confidence: float      = 0.0
    model_confidence: float      = 0.0


# ─── Training status ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    level:       str = Field("all", pattern="^(all|brand|model)$")
    brand:       Optional[str] = None
    max_epochs:  int = Field(100, ge=1, le=500)


class TrainResponse(BaseModel):
    job_id:  str
    message: str


class TrainStatusResponse(BaseModel):
    job_id:   str
    status:   str          # "running" | "done" | "error" | "not_found"
    level:    Optional[str]
    epoch:    Optional[int]
    message:  Optional[str]
    log_tail: List[str]    = Field(default_factory=list)


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:     str
    gpu:        bool
    gpu_name:   Optional[str]
    brands:     List[str] = Field(default_factory=list)
    num_models: int       = 0
