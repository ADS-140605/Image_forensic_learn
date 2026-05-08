# Camera Model Identification (v2 - Strict Paper Architecture)

This version strictly follows the architecture and hyperparameters described in:
**"Camera model identification based on forensic traces extracted from homogeneous patches" (2022)**

## Key Features
- **Strict 7-Block Architecture**: Verified to match the 2,585,149 parameters for 13 brands.
- **Hierarchical Training**: Independent brand and model level classifiers.
- **Homogeneous Patch Selection**: Integral image accelerated extraction.
- **Optimized Dataset**: Image-wise patch caching to speed up training.

## Usage
1. Set `DATA_DIR` in `AI/config.py`.
2. Run training:
   ```bash
   python pipeline.py --epochs 30
   ```
3. Run UI:
   ```bash
   python UI/main.py
   ```
