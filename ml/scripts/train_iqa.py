#!/usr/bin/env python3
"""
Training script for Fundus Image Quality Assessment (IQA).

Multi-task training with three heads: quality score, gradeability, guidance.

Usage:
    python -m ml.scripts.train_iqa
    python -m ml.scripts.train_iqa --config ml/configs/iqa.yaml
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.augmentations import get_train_transforms, get_val_transforms
from ml.models.iqa_model import FundusIQA
from ml.training.callbacks import EarlyStopping, LRLogger, ModelCheckpoint
from ml.training.schedulers import WarmupCosineScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class IQADataset(Dataset):
    """
    Dataset for IQA training.

    Expects a directory structure:
        data_root/images/  - fundus images
        data_root/labels.csv - CSV with columns:
            filename, quality (float 0-1), gradeable (0/1),
            too_dark, too_bright, out_of_focus, occluded, glare,
            low_contrast, off_center, artifacts (all 0/1)
    """

    GUIDANCE_COLS = [
        "too_dark", "too_bright", "out_of_focus", "occluded",
        "glare", "low_contrast", "off_center", "artifacts",
    ]

    def __init__(self, image_paths, labels_df, transform=None):
        self.image_paths = image_paths
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        if self.transform is not None:
            img_np = self.transform(image=img_np)["image"]

        if isinstance(img_np, np.ndarray):
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        else:
            img_tensor = img_np

        row = self.labels_df.iloc[idx]
        targets = {
            "quality": torch.tensor([row["quality"]], dtype=torch.float32),
            "gradeable": torch.tensor([row["gradeable"]], dtype=torch.float32),
            "guidance": torch.tensor(
                [row[c] for c in self.GUIDANCE_COLS], dtype=torch.float32
            ),
        }

        return img_tensor, targets


def build_dataloaders(config: dict) -> tuple:
    """Build train/val data loaders for IQA."""
    import pandas as pd

    data_cfg = config["data"]
    train_cfg = config["training"]

    data_root = Path(data_cfg["data_root"])
    img_size = data_cfg.get("img_size", 224)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    # Load labels
    labels_path = data_root / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. "
            "Expected CSV with columns: filename, quality, gradeable, "
            "too_dark, too_bright, out_of_focus, occluded, glare, "
            "low_contrast, off_center, artifacts"
        )

    labels_df = pd.read_csv(labels_path)
    image_dir = data_root / "images"
    image_paths = [image_dir / fn for fn in labels_df["filename"]]

    # Train/val split
    train_ratio = data_cfg.get("train_split", 0.8)
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    indices = list(range(n_total))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_paths = [image_paths[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    train_labels = labels_df.iloc[train_indices].reset_index(drop=True)
    val_labels = labels_df.iloc[val_indices].reset_index(drop=True)

    train_transform = get_train_transforms(img_size=img_size)
    val_transform = get_val_transforms(img_size=img_size)

    train_dataset = IQADataset(train_paths, train_labels, transform=train_transform)
    val_dataset = IQADataset(val_paths, val_labels, transform=val_transform)

    logger.info(f"IQA dataset: {n_train} train, {n_val} val samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_iqa(model, train_loader, val_loader, config):
    """Full training loop for IQA model."""
    train_cfg = config["training"]
    device = torch.device(train_cfg.get("device", "mps"))
    epochs = train_cfg.get("epochs", 30)
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)
    use_amp = train_cfg.get("use_amp", True)

    quality_w = train_cfg.get("quality_loss_weight", 1.0)
    gradeable_w = train_cfg.get("gradeable_loss_weight", 1.0)
    guidance_w = train_cfg.get("guidance_loss_weight", 0.5)

    output_dir = Path(config.get("output", {}).get("dir", "outputs/iqa"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_cfg = config.get("scheduler", {})
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=sched_cfg.get("warmup_epochs", 3),
        total_steps=epochs,
        min_lr=sched_cfg.get("min_lr", 1e-7),
    )

    early_stopping = EarlyStopping(
        patience=config.get("early_stopping", {}).get("patience", 8),
        mode="min",
    )

    checkpoint_cb = ModelCheckpoint(
        save_dir=str(output_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    lr_logger = LRLogger(log_file=str(output_dir / "lr_history.json"))

    # Init W&B
    log_cfg = config.get("logging", {})
    use_wandb = log_cfg.get("wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=log_cfg.get("wandb_project", "netra-iqa"),
                config=config,
                dir=str(output_dir),
            )
        except ImportError:
            use_wandb = False

    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                device_type = "cpu" if device.type == "mps" else device.type
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    preds = model(images)
                    losses = model.compute_loss(
                        preds, targets,
                        quality_weight=quality_w,
                        gradeable_weight=gradeable_w,
                        guidance_weight=guidance_w,
                    )
                    loss = losses["total"]
            else:
                preds = model(images)
                losses = model.compute_loss(
                    preds, targets,
                    quality_weight=quality_w,
                    gradeable_weight=gradeable_w,
                    guidance_weight=guidance_w,
                )
                loss = losses["total"]

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = train_loss_total / train_samples

        # --- Validate ---
        model.eval()
        val_loss_total = 0.0
        val_quality_mae = 0.0
        val_gradeable_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

                preds = model(images)
                losses = model.compute_loss(
                    preds, targets,
                    quality_weight=quality_w,
                    gradeable_weight=gradeable_w,
                    guidance_weight=guidance_w,
                )

                bs = images.size(0)
                val_loss_total += losses["total"].item() * bs

                # Quality MAE
                val_quality_mae += (
                    (preds["quality"] - targets["quality"]).abs().sum().item()
                )

                # Gradeability accuracy
                grad_preds = (torch.sigmoid(preds["gradeable"]) > 0.5).float()
                val_gradeable_correct += (grad_preds == targets["gradeable"]).sum().item()

                val_samples += bs

        avg_val_loss = val_loss_total / val_samples
        avg_quality_mae = val_quality_mae / val_samples
        gradeable_acc = val_gradeable_correct / val_samples

        scheduler.step()
        lr_logger(epoch, optimizer)

        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_quality_mae": avg_quality_mae,
            "val_gradeable_acc": gradeable_acc,
            "epoch_time": time.time() - epoch_start,
        }

        if use_wandb:
            import wandb
            wandb.log(metrics, step=epoch)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Quality MAE: {avg_quality_mae:.4f} | "
            f"Gradeable Acc: {gradeable_acc:.4f} | "
            f"Time: {metrics['epoch_time']:.1f}s"
        )

        checkpoint_cb(epoch, avg_val_loss, model, optimizer, scheduler, metrics)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if early_stopping(epoch, avg_val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if use_wandb:
        import wandb
        wandb.finish()

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train IQA model")
    parser.add_argument("--config", type=str, default="ml/configs/iqa.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = config.get("training", {}).get("device", "mps")
    if device == "mps" and not torch.backends.mps.is_available():
        config["training"]["device"] = "cpu"

    logger.info("Building FundusIQA model...")
    model_cfg = config["model"]
    model = FundusIQA(
        num_guidance_classes=model_cfg.get("num_guidance_classes", 8),
        pretrained_backbone=model_cfg.get("pretrained_backbone", True),
        dropout=model_cfg.get("dropout", 0.3),
        quality_threshold=model_cfg.get("quality_threshold", 0.5),
        gradeable_threshold=model_cfg.get("gradeable_threshold", 0.5),
    )

    logger.info("Building data loaders...")
    train_loader, val_loader = build_dataloaders(config)

    train_iqa(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
