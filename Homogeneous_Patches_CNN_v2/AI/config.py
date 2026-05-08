"""
config.py
Configuration for the Camera Model Identification project.
"""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dresden"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Patch Extraction ---
PATCH_SIZE = 128
STRIDE = 32 # 0.25 * PATCH_SIZE
SIGMA_MIN = 0.005
SIGMA_MAX = 0.02
TARGET_PATCHES = 200

# --- Training Hyperparameters (from the paper) ---
BATCH_SIZE = 512 # Paper uses 512, adjust to 64 or 128 if OOM
INITIAL_LR = 0.1
MOMENTUM = 0.8
WEIGHT_DECAY = 0.0005
LR_DECAY = 0.9 # Multiplicative decay per epoch
MAX_EPOCHS = 100

# Early Stopping
ES_PATIENCE = 5
ES_MIN_DELTA = 0.02

# --- Dataset ---
SEED = 42
NUM_WORKERS = 4
