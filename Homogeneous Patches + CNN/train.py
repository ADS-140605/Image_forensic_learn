"""
train.py — CLI entry point for hierarchical training.

Usage:
    python train.py [--dataset <path>] [--epochs <N>] [--level brand|model|all] [--brand <BrandName>]

Examples:
    python train.py                                        # train all levels
    python train.py --level brand --epochs 50
    python train.py --level model --brand Nikon --epochs 80
    python train.py --dataset D:/data/dresden --epochs 100
"""
import argparse
import sys
from pathlib import Path

# ── make the project root importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from AI.training.hierarchical import HierarchicalTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchical Source Camera Identification — Training"
    )
    p.add_argument("--dataset", type=Path, default=None,
                   help="Path to Dresden Image Database root. Defaults to AI/config.py::DATASET_DIR.")
    p.add_argument("--epochs",  type=int,  default=100,
                   help="Maximum number of training epochs (default: 100).")
    p.add_argument("--level",   choices=["brand", "model", "all"], default="all",
                   help="Which hierarchy level to train (default: all).")
    p.add_argument("--brand",   type=str, default=None,
                   help="Brand name for level='model'. Required when --level model.")
    p.add_argument("--gpu",     type=int, default=0,
                   help="GPU index (default: 0). Use -1 to force CPU.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"[train.py] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("[train.py] Using CPU.")

    trainer = HierarchicalTrainer(dataset_dir=args.dataset, device=device)

    if args.level == "all":
        trainer.train_all(max_epochs=args.epochs)

    elif args.level == "brand":
        trainer.train_brand_level(max_epochs=args.epochs)
        trainer.save_metadata()

    elif args.level == "model":
        if args.brand is None:
            print("[ERROR] --brand is required when --level model.")
            sys.exit(1)
        if args.brand not in trainer.brand_to_idx:
            available = list(trainer.brand_to_idx.keys())
            print(f"[ERROR] Brand '{args.brand}' not found. Available: {available}")
            sys.exit(1)
        trainer.train_model_level(args.brand, max_epochs=args.epochs)
        trainer.save_metadata()

    print("[train.py] Done.")


if __name__ == "__main__":
    main()
