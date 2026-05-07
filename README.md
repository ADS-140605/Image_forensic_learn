# Camera Model Identification via Hierarchical CNN on Homogeneous Patches

## 🎯 Project Overview
This repository implements a **hierarchical classification system** for source‑camera model identification using forensic traces extracted from homogeneous image patches. The pipeline:
1. **Patch extraction** – overlapping 128×128 windows (stride = 32) with homogeneity filtering based on per‑channel standard deviation.
2. **Convolutional neural network** – a 7‑block architecture (see `AI/models/convnet.py`).
3. **Hierarchical training** – brand‑level classifier → model‑level classifiers for each brand.
4. **Inference** – majority‑vote over 200 patches per image for brand, then model.

The implementation is fully GPU‑accelerated (tested on an RTX 3050) and includes utilities for fast integral‑image computation, data loading, training loops, and a minimal FastAPI UI.

---

## 📂 Directory Structure
```
Image_Forensic/
├─ .gitignore                # ignored files (env, caches, etc.)
├─ requirements.txt          # Python deps
├─ Homogeneous Patches + CNN/
│   ├─ AI/                  # Core model, dataset, utils, training
│   │   ├─ config.py        # Hyper‑parameters
│   │   ├─ models/convnet.py
│   │   ├─ dataset/         # Dresden dataset wrapper & transforms
│   │   ├─ utils/           # Integral image, metrics
│   │   ├─ training/        # Trainer, distribution helper
│   │   ├─ inference/       # Predictor utilities
│   ├─ UI/                  # FastAPI backend & simple HTML UI
│   │   ├─ static/          # index.html, style.css, app.js
│   │   ├─ routers/         # upload / training routes
│   │   └─ schemas.py
│   ├─ pipeline.py          # End‑to‑end script (extract → train → infer)
│   ├─ tests/               # PyTest suite
│   └─ conftest.py
└─ README.md                # **this file**
```

---

## ⚙️ Installation
```bash
# 1️⃣ Clone the repository
git clone <repo‑url> Image_Forensic
cd Image_Forensic

# 2️⃣ Create a virtual environment (recommended)
python -m venv myenv
myenv\Scripts\activate   # Windows PowerShell

# 3️⃣ Install dependencies
pip install -r requirements.txt
```
> **Note**: The code has been tested with **Python 3.11** and **torch 2.5.1** (CUDA‑enabled). Adjust `torch` version if you use a different CUDA toolkit.

---

## 📦 Data Preparation
1. **Download the Dresden Image Database** (or any similar dataset). The expected layout is:
```
<DATA_ROOT>/dresden/
│   ├─ Nikon/
│   ├─ Sony/
│   └─ Samsung/
```
2. Set the path when running the pipeline:
```bash
python pipeline.py \
    --data-dir "d:/Image_Forensic/Homogeneous Patches + CNN/data/dresden" \
    --output-dir outputs \
    --epochs 30 \
    --batch-size 64
```
The script will automatically create a `patches/` folder under the output directory containing the extracted homogeneous patches.

---

## 🚀 Usage
### 1️⃣ Run the full pipeline (extract → train → optional inference)
```bash
python pipeline.py \
    --data-dir <PATH_TO_DRESDEN> \
    --output-dir outputs \
    --epochs 30 \
    --batch-size 64
```
- **Level‑1 (brand) model** is saved as `outputs/brand_classifier/best.pt`.
- **Level‑2 (model) classifiers** are stored under `outputs/model_classifiers/<brand>/`.

### 2️⃣ Inference on a single image
```python
from AI.inference.predictor import CameraPredictor
predictor = CameraPredictor(
    brand_ckpt="outputs/brand_classifier/best.pt",
    model_ckpt_dir="outputs/model_classifiers",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
brand, model = predictor.predict("path/to/image.jpg")
print(f"Brand: {brand}, Model: {model}")
```
The predictor extracts 200 patches, runs the hierarchical classifiers, and returns the majority‑vote result.

---

## 📈 Evaluation
The test suite covers:
- Patch extraction correctness (including edge‑case handling of insufficient windows).
- Forward‑pass shapes, dropout behavior, and parameter counts.
- GPU compatibility.
Run the full suite with:
```bash
pytest -q
```
You can also evaluate on a custom test set by providing a folder of images and calling the `predictor` on each image, then computing accuracy against ground‑truth labels.

---

## 🎨 UI (FastAPI)
A lightweight web UI lives in `UI/`:
- **Upload** a JPEG/PNG image to obtain brand/model predictions.
- **Start training** for a new dataset via the `/train` endpoint.
Run the API with:
```bash
uvicorn UI.main:app --reload --port 8000
```
Open `http://localhost:8000` in a browser.
The UI uses a dark‑mode glassmorphism style with subtle hover animations (see `UI/static/style.css`).

---

## 🛠️ Development & Testing
- **Code formatting** – `black` and `isort` are recommended.
- **Linting** – `flake8` (project follows PEP 8). 
- **Adding new models** – extend `AI/models/convnet.py` and update `pipeline.py` accordingly.

---

## 🤝 Contributing
Contributions are welcome! Please:
1. Fork the repo.
2. Create a feature branch.
3. Ensure all tests pass (`pytest`).
4. Open a Pull Request with a clear description.

---

## 📜 License
This project is released under the **MIT License**.

---

## 🙏 Acknowledgments
- Dresden Image Database for providing diverse camera images.
- PyTorch community for the robust training utilities.
- FastAPI for the simple web interface.
