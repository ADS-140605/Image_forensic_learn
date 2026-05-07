"""
config.py
Central configuration for the Hierarchical Source Camera Identification system.
"""
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATASET_DIR     = ROOT_DIR / "data" / "dresden"   # point to your Dresden DB root
CHECKPOINT_DIR  = ROOT_DIR / "checkpoints"
LOG_DIR         = ROOT_DIR / "logs"
RESULTS_DIR     = ROOT_DIR / "results"

for _d in (CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── Patch Extraction ─────────────────────────────────────────────────────────
PATCH_SIZE      = 128          # spatial size of each patch (pixels)
STRIDE          = 32           # sliding-window stride (0.25 × PATCH_SIZE)
TARGET_PATCHES  = 200          # P: desired patches per image
SIGMA_MIN       = 0.005        # homogeneity lower bound (normalised [0,1] image)
SIGMA_MAX       = 0.020        # homogeneity upper bound

# ─── ConvNet Architecture ─────────────────────────────────────────────────────
IN_CHANNELS     = 3
FC1_UNITS       = 1024
FC2_UNITS       = 200
DROPOUT_RATE    = 0.3

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
INITIAL_LR      = 0.1
LR_DECAY        = 0.9          # multiplicative decay per epoch
MOMENTUM        = 0.8
WEIGHT_DECAY    = 5e-4
MAX_EPOCHS      = 100

# Early stopping
ES_MIN_DELTA    = 0.02         # minimum loss fluctuation to keep training
ES_PATIENCE     = 5            # consecutive epochs without fluctuation → stop

# ─── Inference ────────────────────────────────────────────────────────────────
NUM_WORKERS     = 4            # DataLoader workers
PIN_MEMORY      = True         # set False if CPU-only

# ─── Random seed ──────────────────────────────────────────────────────────────
SEED            = 42
