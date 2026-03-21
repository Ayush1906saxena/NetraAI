"""
Train a multi-label eye disease classifier on the ODIR-5K dataset.

8 conditions: Normal(N), Diabetes/DR(D), Glaucoma(G), Cataract(C),
              AMD(A), Hypertension(H), Myopia(M), Other(O)

Model: EfficientNet-B0 with sigmoid output (multi-label BCEWithLogitsLoss)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import timm

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "odir_full" / "full_df.csv"
TRAINING_IMG_DIR = PROJECT_ROOT / "data" / "odir_full" / "ODIR-5K" / "ODIR-5K" / "Training Images"
PREPROCESSED_IMG_DIR = PROJECT_ROOT / "data" / "odir_full" / "preprocessed_images"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "multi_disease"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]
NUM_CLASSES = len(LABEL_COLS)
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 2e-4
EPOCHS = 15
DEVICE = "mps"
NUM_WORKERS = 0
SEED = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(filename: str) -> str | None:
    """Try preprocessed first, then training images."""
    p = PREPROCESSED_IMG_DIR / filename
    if p.exists():
        return str(p)
    p = TRAINING_IMG_DIR / filename
    if p.exists():
        return str(p)
    return None


# ── Dataset ──────────────────────────────────────────────────────────────────
class ODIRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        labels = row[LABEL_COLS].values.astype(np.float32)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(labels, dtype=torch.float32)


# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray):
    """Compute per-class AUC-ROC, F1, and overall accuracy."""
    results = {}

    # Per-class AUC-ROC
    auc_scores = {}
    for i, col in enumerate(LABEL_COLS):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        auc_scores[col] = auc
    results["auc"] = auc_scores

    # Per-class F1
    f1_scores_dict = {}
    for i, col in enumerate(LABEL_COLS):
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        f1_scores_dict[col] = f1
    results["f1"] = f1_scores_dict

    # Overall exact-match accuracy
    results["accuracy"] = accuracy_score(labels, preds)

    # Mean AUC (ignoring nan)
    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    results["mean_auc"] = np.mean(valid_aucs) if valid_aucs else 0.0

    # Mean F1
    results["mean_f1"] = np.mean(list(f1_scores_dict.values()))

    return results


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Loading CSV from {CSV_PATH}")

    # ── Load and prepare data ────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")

    # Resolve image paths
    df["img_path"] = df["filename"].apply(resolve_image_path)
    missing = df["img_path"].isna().sum()
    print(f"Missing images: {missing}")
    df = df.dropna(subset=["img_path"]).reset_index(drop=True)
    print(f"Rows with images: {len(df)}")

    # ── Split 80/10/10 ──────────────────────────────────────────────────
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Label distribution
    print("\nLabel distribution (train):")
    for col in LABEL_COLS:
        print(f"  {col}: {train_df[col].sum()} / {len(train_df)} ({train_df[col].mean():.3f})")

    # ── Datasets and loaders ─────────────────────────────────────────────
    train_ds = ODIRDataset(train_df, transform=train_transform)
    val_ds = ODIRDataset(val_df, transform=val_transform)
    test_ds = ODIRDataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ── Model ────────────────────────────────────────────────────────────
    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(f"\nModel: EfficientNet-B0, params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Compute class weights for imbalanced data ────────────────────────
    pos_counts = train_df[LABEL_COLS].sum().values.astype(np.float32)
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(DEVICE)
    print(f"Pos weights: {pos_weight.cpu().numpy().round(2)}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_auc = 0.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, batch {batch_idx+1}/{len(train_loader)}, loss={loss.item():.4f}")

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy()
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs)

        val_loss /= len(val_loader)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = (all_probs >= 0.5).astype(int)

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) — "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"mean_AUC={metrics['mean_auc']:.4f}, mean_F1={metrics['mean_f1']:.4f}, "
              f"acc={metrics['accuracy']:.4f}")

        # Per-class AUC
        auc_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics["auc"].items())
        print(f"  AUC: {auc_str}")
        f1_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics["f1"].items())
        print(f"  F1:  {f1_str}")

        # Save best
        if metrics["mean_auc"] > best_val_auc:
            best_val_auc = metrics["mean_auc"]
            best_epoch = epoch + 1
            save_path = CHECKPOINT_DIR / "best.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": metrics,
                "label_cols": LABEL_COLS,
                "num_classes": NUM_CLASSES,
                "model_name": "efficientnet_b0",
            }, save_path)
            print(f"  ** Saved best model (AUC={best_val_auc:.4f}) to {save_path}")

    # ── Test evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"Loading best checkpoint for test evaluation...")

    checkpoint = torch.load(CHECKPOINT_DIR / "best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = (all_probs >= 0.5).astype(int)

    test_metrics = compute_metrics(all_labels, all_preds, all_probs)

    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Mean AUC-ROC:     {test_metrics['mean_auc']:.4f}")
    print(f"Mean F1:          {test_metrics['mean_f1']:.4f}")
    print(f"\nPer-class results:")
    print(f"{'Class':<5} {'AUC-ROC':<10} {'F1':<10}")
    print("-" * 25)
    for col in LABEL_COLS:
        auc_val = test_metrics["auc"].get(col, float("nan"))
        f1_val = test_metrics["f1"].get(col, 0.0)
        print(f"{col:<5} {auc_val:<10.4f} {f1_val:<10.4f}")

    print(f"\nTraining complete. Checkpoint saved to {CHECKPOINT_DIR / 'best.pth'}")


if __name__ == "__main__":
    main()
