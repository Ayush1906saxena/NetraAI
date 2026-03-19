#!/usr/bin/env python3
"""
Train FundusIQA model on APTOS 2019 dataset using synthetic quality labels.

Multi-task training with three heads:
  - quality score (MSE loss)
  - gradeable (BCE loss)
  - guidance (CE loss, 4 classes)

Usage:
    python ml/scripts/train_iqa_aptos.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.iqa_model import FundusIQA

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 0.01
IMG_SIZE = 224
TRAIN_SPLIT = 0.85
NUM_GUIDANCE_CLASSES = 4  # ok, too_blurry, bad_exposure, misaligned

QUALITY_WEIGHT = 1.0
GRADEABLE_WEIGHT = 1.0
GUIDANCE_WEIGHT = 0.5

DATA_DIR = PROJECT_ROOT / "data" / "aptos"
IMAGE_DIR = DATA_DIR / "train_images"
LABELS_CSV = DATA_DIR / "quality_labels.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "iqa_best.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────
class APTOSIQADataset(Dataset):
    """Dataset that loads APTOS images with synthetic quality labels."""

    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        img_path = self.image_dir / f"{image_id}.png"

        img = Image.open(img_path).convert("RGB")
        # Quick downscale before transforms to avoid operating on huge images
        if max(img.size) > 512:
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)

        # Quality score: continuous [0, 1]
        quality = torch.tensor([float(row["quality_score"])], dtype=torch.float32)

        # Gradeable: binary
        gradeable = torch.tensor([float(row["gradeable"])], dtype=torch.float32)

        # Guidance: integer class label for CE loss
        guidance = torch.tensor(int(row["guidance"]), dtype=torch.long)

        return img, quality, gradeable, guidance


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, ce_guidance):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for images, quality, gradeable, guidance in loader:
        images = images.to(device)
        quality = quality.to(device)
        gradeable = gradeable.to(device)
        guidance = guidance.to(device)

        optimizer.zero_grad(set_to_none=True)

        preds = model(images)

        # MSE for quality score
        loss_q = nn.functional.mse_loss(preds["quality"], quality)

        # BCE for gradeable
        loss_g = nn.functional.binary_cross_entropy_with_logits(
            preds["gradeable"], gradeable
        )

        # CE for guidance (preds["guidance"] has num_guidance_classes logits)
        loss_guid = ce_guidance(preds["guidance"], guidance)

        loss = QUALITY_WEIGHT * loss_q + GRADEABLE_WEIGHT * loss_g + GUIDANCE_WEIGHT * loss_guid

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / n_samples


@torch.no_grad()
def validate(model, loader, device, ce_guidance):
    model.eval()
    total_loss = 0.0
    quality_mae = 0.0
    grad_correct = 0
    guid_correct = 0
    n_samples = 0

    for images, quality, gradeable, guidance in loader:
        images = images.to(device)
        quality = quality.to(device)
        gradeable = gradeable.to(device)
        guidance = guidance.to(device)

        preds = model(images)

        loss_q = nn.functional.mse_loss(preds["quality"], quality)
        loss_g = nn.functional.binary_cross_entropy_with_logits(
            preds["gradeable"], gradeable
        )
        loss_guid = ce_guidance(preds["guidance"], guidance)
        loss = QUALITY_WEIGHT * loss_q + GRADEABLE_WEIGHT * loss_g + GUIDANCE_WEIGHT * loss_guid

        bs = images.size(0)
        total_loss += loss.item() * bs
        quality_mae += (preds["quality"] - quality).abs().sum().item()

        grad_preds = (torch.sigmoid(preds["gradeable"]) > 0.5).float()
        grad_correct += (grad_preds == gradeable).sum().item()

        guid_preds = preds["guidance"].argmax(dim=1)
        guid_correct += (guid_preds == guidance).sum().item()

        n_samples += bs

    return {
        "loss": total_loss / n_samples,
        "quality_mae": quality_mae / n_samples,
        "gradeable_acc": grad_correct / n_samples,
        "guidance_acc": guid_correct / n_samples,
    }


def main():
    set_seed(SEED)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load labels
    print(f"Loading labels from {LABELS_CSV}")
    df = pd.read_csv(LABELS_CSV)
    print(f"Total samples: {len(df)}")
    print(f"Quality grade distribution:\n{df['quality_grade'].value_counts().sort_index()}")
    print(f"Guidance distribution:\n{df['guidance'].value_counts().sort_index()}")

    # Train/val split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    n_train = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[indices[:n_train]]
    val_df = df.iloc[indices[n_train:]]
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = APTOSIQADataset(train_df, IMAGE_DIR, transform=train_transform)
    val_dataset = APTOSIQADataset(val_df, IMAGE_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Model
    print(f"\nBuilding FundusIQA (guidance classes={NUM_GUIDANCE_CLASSES})...")
    model = FundusIQA(
        num_guidance_classes=NUM_GUIDANCE_CLASSES,
        pretrained_backbone=True,
        dropout=0.3,
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Guidance CE loss
    ce_guidance = nn.CrossEntropyLoss()

    # Checkpoint dir
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    print(f"\nStarting training for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, ce_guidance)
        val_metrics = validate(model, val_loader, device, ce_guidance)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Q-MAE: {val_metrics['quality_mae']:.4f} | "
            f"Grad Acc: {val_metrics['gradeable_acc']:.4f} | "
            f"Guid Acc: {val_metrics['guidance_acc']:.4f} | "
            f"LR: {lr_now:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), BEST_CHECKPOINT)
            print(f"  -> Saved best checkpoint (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
