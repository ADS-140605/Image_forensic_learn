import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from AI.dataset.dresden_dataset import DresdenDataset
from AI.dataset.patch_extractor import PatchExtractor
from AI.models.convnet import CameraConvNet
from AI.training.trainer import Trainer
from AI.inference.predictor import HierarchicalPredictor
from AI.utils.integral_image import compute_integral_image

def parse_args():
    parser = argparse.ArgumentParser(description="End‑to‑end pipeline for source‑camera model identification")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root folder of the Dresden Image Database")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory where checkpoints, logs and results are stored")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training / inference")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum number of training epochs per level")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1️⃣  Patch extraction (homogeneous + fallback) – shared across levels
    # ---------------------------------------------------------------------
    print("[Pipeline] Extracting patches …")
    extractor = PatchExtractor(sigma_min=0.005, sigma_max=0.02, seed=42)
    dataset = DresdenDataset(root=data_dir, patch_extractor=extractor, target_patches=200)
    # DataLoader for brand‑level training (labels are brand IDs)
    brand_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # ---------------------------------------------------------------
    # 2️⃣  Train brand classifier (Level 1)
    # ---------------------------------------------------------------
    print("[Pipeline] Training brand classifier …")
    brand_model = CameraConvNet(num_classes=dataset.num_brands)
    brand_trainer = Trainer(brand_model, name="brand_level", device=args.device)
    brand_trainer.fit(brand_loader, max_epochs=args.epochs)
    brand_ckpt = output_dir / "brand_level_best.pt"
    torch.save({
        "model_state": brand_model.state_dict(),
        "num_classes": dataset.num_brands,
    }, brand_ckpt)

    # ---------------------------------------------------------------
    # 3️⃣  Train model‑level classifiers (Level 2) – one per brand
    # ---------------------------------------------------------------
    print("[Pipeline] Training model‑level classifiers …")
    predictor = HierarchicalPredictor(brand_ckpt=str(brand_ckpt), device=args.device)
    for brand_id in range(dataset.num_brands):
        brand_name = dataset.brand_names[brand_id]
        print(f"[Pipeline]   → brand {brand_name} (ID {brand_id})")
        # Filter dataset for the current brand
        brand_subset = dataset.filter_by_brand(brand_id)
        model_loader = DataLoader(brand_subset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        model = CameraConvNet(num_classes=dataset.models_per_brand[brand_id])
        trainer = Trainer(model, name=f"model_{brand_name}", device=args.device)
        trainer.fit(model_loader, max_epochs=args.epochs)
        ckpt_path = output_dir / f"model_{brand_name}_best.pt"
        torch.save({
            "model_state": model.state_dict(),
            "num_classes": dataset.models_per_brand[brand_id],
            "brand_id": brand_id,
        }, ckpt_path)

    # ---------------------------------------------------------------
    # 4️⃣  Inference demo (optional)
    # ---------------------------------------------------------------
    print("[Pipeline] Running a quick inference demo …")
    demo_img_path = data_dir / "dresden" / "sample.jpg"
    if demo_img_path.exists():
        brand, model = predictor.predict(demo_img_path)
        print(f"[Demo] Predicted brand: {brand}, model: {model}")
    else:
        print("[Demo] No sample image found – skip demo inference.")

if __name__ == "__main__":
    main()
