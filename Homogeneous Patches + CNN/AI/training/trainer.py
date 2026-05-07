"""
trainer.py
Core training loop shared by both hierarchy levels.

Implements:
  - SGD with momentum=0.8, weight_decay=5e-4
  - Exponential LR decay (×0.9 per epoch)
  - Early stopping (no fluctuation >0.02 for 5 consecutive epochs)
  - TensorBoard logging
  - Checkpoint saving (best + latest)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AI.config import (
    BATCH_SIZE, INITIAL_LR, LR_DECAY, MOMENTUM, WEIGHT_DECAY,
    MAX_EPOCHS, ES_MIN_DELTA, ES_PATIENCE, CHECKPOINT_DIR, LOG_DIR,
    NUM_WORKERS, PIN_MEMORY,
)


class EarlyStopping:
    """
    Halts training if the training loss does not fluctuate by more than
    `min_delta` for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = ES_PATIENCE, min_delta: float = ES_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self._history:  List[float] = []
        self._counter:  int = 0

    def step(self, loss: float) -> bool:
        """Returns True if training should stop."""
        self._history.append(loss)
        if len(self._history) < 2:
            return False

        recent = self._history[-self.patience:]
        if len(recent) < self.patience:
            return False

        fluctuation = max(recent) - min(recent)
        if fluctuation <= self.min_delta:
            self._counter += 1
        else:
            self._counter = 0

        return self._counter >= self.patience


class Trainer:
    """
    Generic single-level trainer.

    Parameters
    ----------
    model      : CameraConvNet instance
    name       : unique name used for checkpoint / log subdirectory
    device     : torch device
    """

    def __init__(
        self,
        model:   nn.Module,
        name:    str,
        device:  Optional[torch.device] = None,
    ):
        self.model  = model
        self.name   = name
        self.device = device or self._auto_device()
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=INITIAL_LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=LR_DECAY
        )
        self.early_stop = EarlyStopping()

        self.ckpt_dir = CHECKPOINT_DIR / name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(LOG_DIR / name))

        self.best_val_acc: float = 0.0
        self.history: List[Dict] = []

    # ── public ────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
        max_epochs:   int = MAX_EPOCHS,
    ) -> List[Dict]:
        """
        Train for up to max_epochs with early stopping.
        Returns history list of per-epoch metrics.
        """
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            epoch_time = time.time() - t0

            val_loss = val_acc = None
            if val_loader:
                val_loss, val_acc = self._run_epoch(val_loader, training=False)

            lr = self.optimizer.param_groups[0]["lr"]
            self._log(epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time)
            self._save_checkpoint(epoch, val_acc or train_acc)

            self.scheduler.step()

            if self.early_stop.step(train_loss):
                print(f"[{self.name}] Early stopping at epoch {epoch}.")
                break

        self.writer.close()
        return self.history

    # ── private ───────────────────────────────────────────────────────────────
    def _run_epoch(
        self,
        loader:   DataLoader,
        training: bool,
    ) -> Tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        correct    = 0
        total      = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for patches, labels in loader:
                patches = patches.to(self.device, non_blocking=True)
                labels  = labels.to(self.device, non_blocking=True)

                logits = self.model(patches)
                loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(labels)
                correct    += (logits.argmax(dim=1) == labels).sum().item()
                total      += len(labels)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _log(self, epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr, t):
        rec = {
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc":  round(tr_acc,  4),
            "val_loss":   round(vl_loss, 4) if vl_loss is not None else None,
            "val_acc":    round(vl_acc,  4) if vl_acc  is not None else None,
            "lr":         lr,
            "time_s":     round(t, 2),
        }
        self.history.append(rec)
        self.writer.add_scalar("Loss/train", tr_loss, epoch)
        self.writer.add_scalar("Acc/train",  tr_acc,  epoch)
        if vl_loss is not None:
            self.writer.add_scalar("Loss/val", vl_loss, epoch)
            self.writer.add_scalar("Acc/val",  vl_acc,  epoch)
        self.writer.add_scalar("LR", lr, epoch)
        print(
            f"[{self.name}] Epoch {epoch:03d} | "
            f"loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            + (f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f} | " if vl_loss else "")
            + f"lr={lr:.6f} | {t:.1f}s"
        )

    def _save_checkpoint(self, epoch: int, metric: float):
        state = {
            "epoch":          epoch,
            "model_state":    self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metric":         metric,
        }
        torch.save(state, self.ckpt_dir / "latest.pt")
        if metric >= self.best_val_acc:
            self.best_val_acc = metric
            torch.save(state, self.ckpt_dir / "best.pt")

    def load_checkpoint(self, path: Optional[Path] = None, best: bool = True):
        """Load weights from checkpoint (best or latest)."""
        if path is None:
            path = self.ckpt_dir / ("best.pt" if best else "latest.pt")
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        print(f"[{self.name}] Loaded checkpoint from {path} (metric={state['metric']:.4f})")

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[Trainer] Using GPU: {name}")
            return torch.device("cuda")
        print("[Trainer] CUDA not available, using CPU.")
        return torch.device("cpu")
