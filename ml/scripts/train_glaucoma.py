#!/usr/bin/env python3
"""
Training script for Glaucoma optic disc/cup segmentation.

Model: GlaucomaSegmentor (U-Net + EfficientNet-B3)
Loss: Dice + Focal combined loss
Metrics: Dice coefficient, IoU, CDR error

Usage:
    python -m ml.scripts.train_glaucoma
    python -m ml.scripts.train_glaucoma --config ml/configs/glaucoma_seg.yaml
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.glaucoma_unet import GlaucomaSegmentor
from ml.training.callbacks import EarlyStopping, LRLogger, ModelCheckpoint
from ml.training.losses import DiceFocalLoss
from ml.training.schedulers import CosineWithWarmRestarts, WarmupCosineScheduler

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


class GlaucomaSegDataset(Dataset):
    """
    Dataset for glaucoma segmentation.

    Directory structure:
        data_root/images/    - fundus images (.jpg/.png)
        data_root/masks/     - segmentation masks (.png)
            Mask encoding: 0 = background, 128 = optic disc, 255 = optic cup
            OR two separate mask channels stored as RGB where
            R channel = disc mask, G channel = cup mask
    """

    def __init__(
        self,
        image_paths: list,
        mask_paths: list,
        img_size: int = 512,
        transform=None,
        is_train: bool = True,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)

        # Load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.float32)

        # Convert single-channel mask to 2-channel binary masks
        # Convention: 0=bg, 128=disc, 255=cup (cup is inside disc)
        disc_mask = (mask_np >= 100).astype(np.float32)  # disc includes cup region
        cup_mask = (mask_np >= 200).astype(np.float32)   # cup only

        if self.transform is not None:
            import albumentations as A

            # Apply same spatial transforms to image and masks
            masks_stacked = np.stack([disc_mask, cup_mask], axis=-1)
            augmented = self.transform(image=img_np, mask=masks_stacked)
            img_np = augmented["image"]
            masks_stacked = augmented["mask"]

            if isinstance(img_np, torch.Tensor):
                img_tensor = img_np
            else:
                # Normalize manually if transform didn't include ToTensorV2
                img_np = img_np / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np - mean) / std
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

            if isinstance(masks_stacked, torch.Tensor):
                mask_tensor = masks_stacked.permute(2, 0, 1).float()
            else:
                mask_tensor = torch.from_numpy(masks_stacked.transpose(2, 0, 1)).float()
        else:
            # Default normalization
            img_np = img_np / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np - mean) / std
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(
                np.stack([disc_mask, cup_mask], axis=0)
            ).float()

        return img_tensor, mask_tensor


def build_dataloaders(config: dict) -> tuple:
    """Build train/val data loaders for glaucoma segmentation."""
    data_cfg = config["data"]
    train_cfg = config["training"]

    data_root = Path(data_cfg["data_root"])
    img_size = data_cfg.get("img_size", 512)
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)

    image_dir = data_root / "images"
    mask_dir = data_root / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # Collect paired images and masks
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    mask_paths = []
    valid_image_paths = []
    for img_path in image_paths:
        # Try multiple mask extensions
        mask_found = False
        for ext in [".png", ".bmp", ".tiff", img_path.suffix]:
            mask_path = mask_dir / (img_path.stem + ext)
            if mask_path.exists():
                mask_paths.append(mask_path)
                valid_image_paths.append(img_path)
                mask_found = True
                break
        if not mask_found:
            logger.warning(f"No mask found for {img_path.name}, skipping")

    image_paths = valid_image_paths

    if len(image_paths) == 0:
        raise RuntimeError(f"No valid image-mask pairs found in {data_root}")

    # Train/val split
    n_total = len(image_paths)
    train_ratio = data_cfg.get("train_split", 0.8)
    n_train = int(n_total * train_ratio)

    indices = list(range(n_total))
    np.random.shuffle(indices)

    train_img = [image_paths[i] for i in indices[:n_train]]
    train_mask = [mask_paths[i] for i in indices[:n_train]]
    val_img = [image_paths[i] for i in indices[n_train:]]
    val_mask = [mask_paths[i] for i in indices[n_train:]]

    logger.info(f"Glaucoma dataset: {n_train} train, {n_total - n_train} val samples")

    # Build augmentation for segmentation (spatial transforms applied to both)
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                border_mode=0, p=0.5,
            ),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    except ImportError:
        train_transform = None
        val_transform = None

    train_dataset = GlaucomaSegDataset(
        train_img, train_mask, img_size=img_size,
        transform=train_transform, is_train=True,
    )
    val_dataset = GlaucomaSegDataset(
        val_img, val_mask, img_size=img_size,
        transform=val_transform, is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Dice coefficient for binary segmentation masks."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    )


def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Intersection over Union."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def train_glaucoma(model, train_loader, val_loader, config):
    """Full training loop for glaucoma segmentation."""
    train_cfg = config["training"]
    loss_cfg = config.get("loss", {})
    device = torch.device(train_cfg.get("device", "mps"))
    epochs = train_cfg.get("epochs", 80)
    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)
    use_amp = train_cfg.get("use_amp", True)

    output_dir = Path(config.get("output", {}).get("dir", "outputs/glaucoma_seg"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    criterion = DiceFocalLoss(
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        focal_weight=loss_cfg.get("focal_weight", 1.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        smooth=loss_cfg.get("smooth", 1.0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sched_cfg = config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_warm_restarts")
    warmup_epochs = sched_cfg.get("warmup_epochs", 5)

    if sched_type == "cosine_warm_restarts":
        scheduler = CosineWithWarmRestarts(
            optimizer,
            warmup_steps=warmup_epochs,
            T_0=sched_cfg.get("T_0", 15),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("min_lr", 1e-7),
        )
    else:
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_epochs,
            total_steps=epochs,
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

    es_cfg = config.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode="max",  # monitor Dice
    )

    checkpoint_cb = ModelCheckpoint(
        save_dir=str(output_dir / "checkpoints"),
        monitor="val_dice",
        mode="max",
        save_top_k=3,
    )

    lr_logger = LRLogger(log_file=str(output_dir / "lr_history.json"))

    # W&B
    log_cfg = config.get("logging", {})
    use_wandb = log_cfg.get("wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=log_cfg.get("wandb_project", "netra-glaucoma"),
                config=config,
                dir=str(output_dir),
            )
        except ImportError:
            use_wandb = False

    best_dice = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_samples = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                device_type = "cpu" if device.type == "mps" else device.type
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)

        avg_train_loss = train_loss / train_samples

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_dice_disc = 0.0
        val_dice_cup = 0.0
        val_iou_disc = 0.0
        val_iou_cup = 0.0
        val_cdr_errors = []
        val_samples = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, masks)
                bs = images.size(0)
                val_loss += loss.item() * bs

                # Binary predictions
                probs = torch.sigmoid(logits)
                preds_binary = (probs > 0.5).float()

                # Per-sample metrics
                for i in range(bs):
                    # Disc metrics (channel 0)
                    val_dice_disc += compute_dice(preds_binary[i, 0], masks[i, 0])
                    val_iou_disc += compute_iou(preds_binary[i, 0], masks[i, 0])

                    # Cup metrics (channel 1)
                    val_dice_cup += compute_dice(preds_binary[i, 1], masks[i, 1])
                    val_iou_cup += compute_iou(preds_binary[i, 1], masks[i, 1])

                    # CDR error
                    pred_disc_np = preds_binary[i, 0].cpu().numpy()
                    pred_cup_np = preds_binary[i, 1].cpu().numpy()
                    gt_disc_np = masks[i, 0].cpu().numpy()
                    gt_cup_np = masks[i, 1].cpu().numpy()

                    pred_cdr = GlaucomaSegmentor.compute_cdr(pred_disc_np, pred_cup_np)
                    gt_cdr = GlaucomaSegmentor.compute_cdr(gt_disc_np, gt_cup_np)

                    if pred_cdr is not None and gt_cdr is not None:
                        val_cdr_errors.append(abs(pred_cdr - gt_cdr))

                val_samples += bs

        avg_val_loss = val_loss / val_samples
        avg_dice_disc = val_dice_disc / val_samples
        avg_dice_cup = val_dice_cup / val_samples
        avg_dice = (avg_dice_disc + avg_dice_cup) / 2.0
        avg_iou_disc = val_iou_disc / val_samples
        avg_iou_cup = val_iou_cup / val_samples
        avg_cdr_mae = np.mean(val_cdr_errors) if val_cdr_errors else 0.0

        scheduler.step()
        lr_logger(epoch, optimizer)

        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_dice,
            "val_dice_disc": avg_dice_disc,
            "val_dice_cup": avg_dice_cup,
            "val_iou_disc": avg_iou_disc,
            "val_iou_cup": avg_iou_cup,
            "val_cdr_mae": avg_cdr_mae,
            "epoch_time": time.time() - epoch_start,
        }

        if use_wandb:
            import wandb
            wandb.log(metrics, step=epoch)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Dice(disc/cup/avg): {avg_dice_disc:.4f}/{avg_dice_cup:.4f}/{avg_dice:.4f} | "
            f"CDR MAE: {avg_cdr_mae:.4f} | "
            f"Time: {metrics['epoch_time']:.1f}s"
        )

        checkpoint_cb(epoch, avg_dice, model, optimizer, scheduler, metrics)

        if avg_dice > best_dice:
            best_dice = avg_dice

        if early_stopping(epoch, avg_dice):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if use_wandb:
        import wandb
        wandb.finish()

    logger.info(f"Training complete. Best avg Dice: {best_dice:.4f}")
    return best_dice


def main():
    parser = argparse.ArgumentParser(description="Train glaucoma segmentation model")
    parser.add_argument("--config", type=str, default="ml/configs/glaucoma_seg.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    device = config.get("training", {}).get("device", "mps")
    if device == "mps" and not torch.backends.mps.is_available():
        config["training"]["device"] = "cpu"

    logger.info("Building GlaucomaSegmentor model...")
    model_cfg = config["model"]
    model = GlaucomaSegmentor(
        encoder_name=model_cfg.get("encoder_name", "efficientnet-b3"),
        encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
        in_channels=model_cfg.get("in_channels", 3),
        num_classes=model_cfg.get("num_classes", 2),
        decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32, 16])),
        dropout=model_cfg.get("dropout", 0.2),
    )

    logger.info("Building data loaders...")
    train_loader, val_loader = build_dataloaders(config)

    train_glaucoma(model, train_loader, val_loader, config)


if __name__ == "__main__":
    main()
