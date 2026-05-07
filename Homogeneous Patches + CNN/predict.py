"""
predict.py — CLI entry point for single-image or batch inference.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --checkpoints checkpoints/
    python predict.py --batch path/to/folder/  --out results/predictions.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchical Source Camera Identification — Inference"
    )
    p.add_argument("--image",       type=Path, default=None,
                   help="Path to a single test image.")
    p.add_argument("--batch",       type=Path, default=None,
                   help="Directory of images for batch inference.")
    p.add_argument("--out",         type=Path, default=None,
                   help="CSV output path for batch mode.")
    p.add_argument("--checkpoints", type=Path, default=None,
                   help="Checkpoint directory (defaults to AI/config.py::CHECKPOINT_DIR).")
    p.add_argument("--gpu",         type=int,  default=0,
                   help="GPU index (default: 0). Use -1 to force CPU.")
    return p.parse_args()


def load_predictor(checkpoint_dir, gpu):
    import torch
    from AI.inference.predictor import HierarchicalPredictor
    from AI.config import CHECKPOINT_DIR

    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    ckpt = checkpoint_dir or CHECKPOINT_DIR
    return HierarchicalPredictor(checkpoint_dir=ckpt, device=device)


def predict_single(predictor, img_path: Path) -> dict:
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return predictor.predict(img)


def main():
    args = parse_args()

    if args.image is None and args.batch is None:
        print("[ERROR] Provide --image or --batch.")
        sys.exit(1)

    predictor = load_predictor(args.checkpoints, args.gpu)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    if args.image:
        result = predict_single(predictor, args.image)
        print(f"\n{'─'*40}")
        print(f"  Image  : {args.image.name}")
        print(f"  Brand  : {result['brand']}")
        print(f"  Model  : {result['model']}")
        print(f"  Patches: {result['num_patches']}")
        print(f"  Brand votes : {result['brand_votes']}")
        print(f"  Model votes : {result['model_votes']}")
        print(f"{'─'*40}\n")

    elif args.batch:
        images = [p for p in Path(args.batch).rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            print(f"[ERROR] No images found in {args.batch}")
            sys.exit(1)

        rows = []
        for img_path in images:
            try:
                res = predict_single(predictor, img_path)
                rows.append({
                    "file":        img_path.name,
                    "brand":       res["brand"],
                    "model":       res["model"],
                    "num_patches": res["num_patches"],
                    "brand_votes": res["brand_votes"],
                    "model_votes": res["model_votes"],
                })
                print(f"✓ {img_path.name:40s} → {res['brand']} / {res['model']}")
            except Exception as e:
                print(f"✗ {img_path.name}: {e}")
                rows.append({"file": img_path.name, "brand": "ERROR", "model": str(e)})

        out_path = args.out or Path("predictions.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "brand", "model", "num_patches",
                                                    "brand_votes", "model_votes"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[predict.py] Results saved to {out_path}")


if __name__ == "__main__":
    main()
