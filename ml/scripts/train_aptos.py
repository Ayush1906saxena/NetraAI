"""
Train DR grading model on APTOS 2019 dataset.

Uses EfficientNet-B3 backbone (pretrained ImageNet) with the full
Netra AI training pipeline: AMP, weighted sampling, label smoothing,
cosine scheduler, early stopping, QWK/AUC metrics.

RETFound weights are available but require the RETFound repo code to load.
This script uses a proven EfficientNet backbone that trains well on APTOS.

Usage:
    python -m ml.scripts.train_aptos
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.dataset import FundusDataset
from ml.training.metrics import compute_qwk, compute_auc, compute_sensitivity_specificity


class DRGrader(nn.Module):
    """EfficientNet-B3 DR Grader — strong baseline for APTOS."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device if device != "mps" else "cpu"):
            logits = model(images)
            loss = criterion(logits, labels)

        if device == "mps":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    qwk = compute_qwk(all_labels, all_preds)
    return total_loss / len(loader), qwk


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)
    all_labels = np.array(all_labels)

    qwk = compute_qwk(all_labels, all_preds)
    auc = compute_auc(all_labels, all_probs, referable_threshold=2)
    ss = compute_sensitivity_specificity(all_labels, all_probs, referable_threshold=2)

    return {
        "qwk": qwk,
        "auc": auc,
        "sensitivity": ss.get("sensitivity", 0.0),
        "specificity": ss.get("specificity", 0.0),
        "accuracy": (all_preds == all_labels).mean(),
    }


def main():
    # === Config ===
    DATA_ROOT = "data/aptos_split"
    CHECKPOINT_DIR = "checkpoints/dr_aptos"
    EPOCHS = 30
    BATCH_SIZE = 16
    LR = 3e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 8
    NUM_WORKERS = 4

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("NETRA AI — DR Grading Training on APTOS 2019")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}")

    # === Data ===
    print("\nLoading datasets...")
    train_ds = FundusDataset(DATA_ROOT, "train")
    val_ds = FundusDataset(DATA_ROOT, "val")
    test_ds = FundusDataset(DATA_ROOT, "test")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Class distribution: {train_ds.get_class_distribution()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_ds.get_weighted_sampler(),
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)

    # === Model ===
    print("\nLoading EfficientNet-B3 model...")
    model = DRGrader(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # === Training setup ===
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # === Training loop ===
    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 70)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'T-QWK':>6} | {'V-QWK':>6} | {'V-AUC':>6} | {'Sens':>6} | {'Spec':>6} | {'V-Acc':>6}")
    print("-" * 70)

    best_qwk = -1
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_qwk = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        is_best = val_metrics["qwk"] > best_qwk
        marker = ""
        if is_best:
            best_qwk = val_metrics["qwk"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_qwk": best_qwk,
                "metrics": val_metrics,
            }, f"{CHECKPOINT_DIR}/best.pth")
            marker = " *"
        else:
            epochs_no_improve += 1

        print(
            f"{epoch+1:>5} | {train_loss:>8.4f} | {train_qwk:>6.3f} | "
            f"{val_metrics['qwk']:>6.3f} | {val_metrics['auc']:>6.3f} | "
            f"{val_metrics['sensitivity']:>6.3f} | {val_metrics['specificity']:>6.3f} | "
            f"{val_metrics['accuracy']:>6.3f}{marker}"
        )

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # === Final test evaluation ===
    print("\n" + "=" * 70)
    print("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(f"{CHECKPOINT_DIR}/best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best model from epoch {ckpt['epoch']+1} (val QWK: {ckpt['best_qwk']:.4f})")

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTEST RESULTS:")
    print(f"  QWK:         {test_metrics['qwk']:.4f}")
    print(f"  AUC-ROC:     {test_metrics['auc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
