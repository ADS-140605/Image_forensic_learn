"""
trainer.py
Shared training logic.
"""
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from AI.config import INITIAL_LR, MOMENTUM, WEIGHT_DECAY, LR_DECAY, LOG_DIR, CHECKPOINT_DIR

class Trainer:
    def __init__(self, model, name, device='cuda'):
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=INITIAL_LR, 
                             momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=LR_DECAY)
        self.writer = SummaryWriter(log_dir=str(LOG_DIR / name))
        self.ckpt_dir = CHECKPOINT_DIR / name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for patches, labels in loader:
            patches, labels = patches.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(patches)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * patches.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / total, correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for patches, labels in loader:
                patches, labels = patches.to(self.device), labels.to(self.device)
                outputs = self.model(patches)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * patches.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch(train_loader)
            vl_loss, vl_acc = self.validate(val_loader)
            dt = time.time() - t0
            
            lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            self.writer.add_scalar('Loss/train', tr_loss, epoch)
            self.writer.add_scalar('Loss/val', vl_loss, epoch)
            self.writer.add_scalar('Acc/train', tr_acc, epoch)
            self.writer.add_scalar('Acc/val', vl_acc, epoch)
            
            print(f"[{self.name}] Epoch {epoch:03d} | tr_loss: {tr_loss:.4f} tr_acc: {tr_acc:.4f} | "
                  f"vl_loss: {vl_loss:.4f} vl_acc: {vl_acc:.4f} | lr: {lr:.6f} | {dt:.1f}s")
            
            # Save latest
            torch.save(self.model.state_dict(), self.ckpt_dir / "latest.pt")
            # Save best
            if vl_acc >= self.best_acc:
                self.best_acc = vl_acc
                torch.save(self.model.state_dict(), self.ckpt_dir / "best.pt")
