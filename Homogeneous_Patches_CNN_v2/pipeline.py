"""
pipeline.py
End-to-end training and evaluation pipeline.
"""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from AI.config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, CHECKPOINT_DIR
from AI.dataset.dresden import build_label_maps, DresdenDataset
from AI.training.distribution import get_sampler
from AI.training.trainer import Trainer
from AI.models.convnet import CameraConvNet

def run_pipeline(data_dir, epochs=MAX_EPOCHS):
    data_dir = Path(data_dir)
    print(f"Loading dataset from {data_dir}...")
    brand_to_idx, model_to_idx, records = build_label_maps(data_dir)
    
    # Save maps
    with open(CHECKPOINT_DIR / "label_maps.json", "w") as f:
        json.dump({"brand": brand_to_idx, "model": model_to_idx}, f)

    # 1. Train Brand Classifier
    print("--- Training Brand Classifier ---")
    train_size = int(0.8 * len(records))
    val_size = len(records) - train_size
    train_records, val_records = random_split(records, [train_size, val_size])
    
    train_ds = DresdenDataset(train_records, brand_to_idx, model_to_idx, level='brand')
    val_ds = DresdenDataset(val_records, brand_to_idx, model_to_idx, level='brand')
    
    train_sampler = get_sampler(train_records)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    brand_model = CameraConvNet(num_classes=len(brand_to_idx))
    brand_trainer = Trainer(brand_model, "brand_classifier")
    brand_trainer.fit(train_loader, val_loader, epochs=epochs)

    # 2. Train Model Classifiers for brands with multiple models
    # Nikon, Samsung, Sony are the main ones in Dresden
    brand_models = {}
    for m in model_to_idx:
        b = m.split('_')[0]
        if b not in brand_models: brand_models[b] = []
        brand_models[b].append(m)
        
    for brand, models in brand_models.items():
        if len(models) > 1:
            print(f"--- Training Model Classifier for {brand} ---")
            # Filter records for this brand
            b_records = [r for r in records if r[1] == brand]
            b_train_size = int(0.8 * len(b_records))
            b_val_size = len(b_records) - b_train_size
            b_train_recs, b_val_recs = random_split(b_records, [b_train_size, b_val_size])
            
            # Local mapping for this brand
            local_model_to_idx = {m: i for i, m in enumerate(sorted(models))}
            
            b_train_ds = DresdenDataset(b_train_recs, brand_to_idx, local_model_to_idx, level='model')
            b_val_ds = DresdenDataset(b_val_recs, brand_to_idx, local_model_to_idx, level='model')
            
            b_sampler = get_sampler(b_train_recs)
            b_train_loader = DataLoader(b_train_ds, batch_size=BATCH_SIZE, sampler=b_sampler, num_workers=NUM_WORKERS)
            b_val_loader = DataLoader(b_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
            m_model = CameraConvNet(num_classes=len(models))
            m_trainer = Trainer(m_model, f"model_classifier/{brand}")
            m_trainer.fit(b_train_loader, b_val_loader, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.epochs)
