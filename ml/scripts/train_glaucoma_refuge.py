#!/usr/bin/env python3
"""
Training script for Glaucoma segmentation on REFUGE2 dataset.

The REFUGE2 dataset provides ROI-cropped fundus images (460x460) with glaucoma
classification labels but no segmentation masks. We generate pseudo disc/cup
masks using unsupervised color-based segmentation (Otsu thresholding on the
red/green channels), then train GlaucomaSegmentor to learn refined segmentation.

Model: GlaucomaSegmentor (U-Net + EfficientNet-B2)
Loss: Dice + Focal combined loss
Metrics: Per-class Dice, CDR MAE

Usage:
    .venv/bin/python ml/scripts/train_glaucoma_refuge.py
"""

import logging
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.glaucoma_unet import DiceFocalLoss, GlaucomaSegmentor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pseudo-mask generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_pseudo_masks(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo disc and cup masks from an ROI-cropped fundus image.

    Uses morphological operations and Otsu thresholding on color channels.
    The optic disc is the bright elliptical region; the cup is the brighter
    central area within the disc.

    Args:
        image_np: (H, W, 3) uint8 RGB image, already ROI-cropped around disc.

    Returns:
        disc_mask: (H, W) binary uint8, 1 = optic disc
        cup_mask:  (H, W) binary uint8, 1 = optic cup
    """
    h, w = image_np.shape[:2]

    # Convert to different color spaces
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    # --- Optic Disc: bright region in L channel ---
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    # Gaussian blur to reduce noise
    l_blur = cv2.GaussianBlur(l_enhanced, (15, 15), 0)

    # Otsu threshold for disc
    _, disc_binary = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    disc_binary = cv2.morphologyEx(disc_binary, cv2.MORPH_CLOSE, kernel_large)
    disc_binary = cv2.morphologyEx(disc_binary, cv2.MORPH_OPEN, kernel_large)

    # Keep only the largest connected component (the disc)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disc_binary, connectivity=8)
    if num_labels > 1:
        # Label 0 is background
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        disc_binary = ((labels == largest) * 255).astype(np.uint8)

    # Fill holes
    contours, _ = cv2.findContours(disc_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    disc_filled = np.zeros_like(disc_binary)
    if contours:
        cv2.drawContours(disc_filled, contours, -1, 255, -1)
        # Fit ellipse for smoother boundary if enough points
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            disc_filled = np.zeros_like(disc_binary)
            cv2.ellipse(disc_filled, ellipse, 255, -1)

    disc_mask = (disc_filled > 0).astype(np.uint8)

    # --- Optic Cup: brightest region within the disc ---
    # Mask the image to only look within the disc
    red_channel = image_np[:, :, 0].astype(np.float32)
    green_channel = image_np[:, :, 1].astype(np.float32)

    # Cup is brightest in red channel, and has low saturation
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Combine: high value + low saturation = cup (pale/white region)
    cup_score = value.astype(np.float32) - 0.5 * saturation.astype(np.float32)
    cup_score = np.clip(cup_score, 0, 255).astype(np.uint8)

    # Only consider pixels within the disc
    cup_score_masked = cup_score.copy()
    cup_score_masked[disc_mask == 0] = 0

    # Otsu threshold within disc region
    disc_pixels = cup_score[disc_mask > 0]
    if len(disc_pixels) > 100:
        threshold, _ = cv2.threshold(disc_pixels.astype(np.uint8), 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Use a slightly higher threshold to be conservative about cup
        threshold = min(threshold + 15, 250)
        cup_binary = ((cup_score_masked > threshold) & (disc_mask > 0)).astype(np.uint8) * 255
    else:
        # Fallback: cup is central 30% of disc
        cup_binary = np.zeros_like(disc_binary)

    # Morphological cleanup for cup
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cup_binary = cv2.morphologyEx(cup_binary, cv2.MORPH_CLOSE, kernel_med)
    cup_binary = cv2.morphologyEx(cup_binary, cv2.MORPH_OPEN, kernel_med)

    # Keep largest component
    num_labels_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(cup_binary, connectivity=8)
    if num_labels_c > 1:
        largest_c = 1 + np.argmax(stats_c[1:, cv2.CC_STAT_AREA])
        cup_binary = ((labels_c == largest_c) * 255).astype(np.uint8)

    # Fit ellipse for cup too
    contours_c, _ = cv2.findContours(cup_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_filled = np.zeros_like(cup_binary)
    if contours_c:
        largest_c_contour = max(contours_c, key=cv2.contourArea)
        if len(largest_c_contour) >= 5 and cv2.contourArea(largest_c_contour) > 50:
            ellipse_c = cv2.fitEllipse(largest_c_contour)
            cv2.ellipse(cup_filled, ellipse_c, 255, -1)
        else:
            cv2.drawContours(cup_filled, contours_c, -1, 255, -1)

    # Ensure cup is inside disc
    cup_mask = ((cup_filled > 0) & (disc_mask > 0)).astype(np.uint8)

    # Sanity: cup should be smaller than disc
    disc_area = disc_mask.sum()
    cup_area = cup_mask.sum()
    if disc_area > 0 and cup_area / disc_area > 0.85:
        # Cup too large — shrink it by eroding
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cup_mask = cv2.erode(cup_mask, erode_k, iterations=2)
    if disc_area > 0 and cup_area / disc_area < 0.02:
        # Cup too small — use central 25% of disc as fallback
        cy, cx = np.where(disc_mask > 0)
        if len(cy) > 0:
            center_y, center_x = cy.mean(), cx.mean()
            Y, X = np.ogrid[:h, :w]
            radius = np.sqrt(disc_area / np.pi) * 0.35
            cup_mask = (((Y - center_y)**2 + (X - center_x)**2) < radius**2).astype(np.uint8)
            cup_mask = cup_mask & disc_mask

    return disc_mask, cup_mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class REFUGE2SegDataset(Dataset):
    """
    REFUGE2 dataset for disc/cup segmentation with pseudo-masks.
    Images are ROI-cropped around the optic disc (460x460).
    Masks are generated on-the-fly using color-based segmentation.
    """

    def __init__(
        self,
        image_paths: list[Path],
        img_size: int = 512,
        transform=None,
        precompute_masks: bool = True,
    ):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transform

        # Precompute pseudo-masks to avoid recomputation each epoch
        self.masks_disc = []
        self.masks_cup = []
        if precompute_masks:
            logger.info(f"Generating pseudo-masks for {len(image_paths)} images...")
            for i, p in enumerate(image_paths):
                img = np.array(Image.open(p).convert("RGB"))
                disc, cup = generate_pseudo_masks(img)
                self.masks_disc.append(disc)
                self.masks_cup.append(cup)
                if (i + 1) % 200 == 0:
                    logger.info(f"  Generated {i + 1}/{len(image_paths)} masks")
            logger.info("Pseudo-mask generation complete.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32)

        # Get masks
        disc_mask = self.masks_disc[idx]
        cup_mask = self.masks_cup[idx]

        # Resize masks
        disc_mask = cv2.resize(disc_mask, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)
        cup_mask = cv2.resize(cup_mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        disc_mask = disc_mask.astype(np.float32)
        cup_mask = cup_mask.astype(np.float32)

        if self.transform is not None:
            import albumentations as A
            masks_stacked = np.stack([disc_mask, cup_mask], axis=-1)
            augmented = self.transform(image=img_np, mask=masks_stacked)
            img_np = augmented["image"]
            masks_stacked = augmented["mask"]

            if isinstance(img_np, torch.Tensor):
                img_tensor = img_np
            else:
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
            img_np = img_np / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np - mean) / std
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
            mask_tensor = torch.from_numpy(
                np.stack([disc_mask, cup_mask], axis=0)
            ).float()

        return img_tensor, mask_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return float(
        (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)

    # ── Config ──
    DATA_ROOT = PROJECT_ROOT / "data" / "refuge2" / "Refuge2_data"
    IMG_SIZE = 512
    BATCH_SIZE = 4
    EPOCHS = 30
    LR = 3e-4
    WEIGHT_DECAY = 0.01
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CKPT = CHECKPOINT_DIR / "glaucoma_best.pth"

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data root: {DATA_ROOT}")

    # ── Collect image paths ──
    train_dir = DATA_ROOT / "train_rgb_roi"
    val_dir = DATA_ROOT / "val_rgb_roi"

    train_paths = sorted(list(train_dir.glob("*.jpg")))
    val_paths = sorted(list(val_dir.glob("*.jpg")))

    logger.info(f"Found {len(train_paths)} training images, {len(val_paths)} validation images")

    # Subsample for efficiency: use 800 train, 200 val (still substantial)
    if len(train_paths) > 800:
        random.shuffle(train_paths)
        train_paths = train_paths[:800]
    if len(val_paths) > 200:
        random.shuffle(val_paths)
        val_paths = val_paths[:200]

    logger.info(f"Using {len(train_paths)} train, {len(val_paths)} val images")

    # ── Augmentations ──
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=20,
            border_mode=0, p=0.4,
        ),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # ── Datasets & Loaders ──
    train_ds = REFUGE2SegDataset(train_paths, img_size=IMG_SIZE, transform=train_transform)
    val_ds = REFUGE2SegDataset(val_paths, img_size=IMG_SIZE, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Quick pseudo-mask quality check ──
    _check_pseudo_masks(train_ds)

    # ── Model ──
    model = GlaucomaSegmentor(
        encoder_name="efficientnet-b2",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=2,
        decoder_channels=(256, 128, 64, 32, 16),
        dropout=0.2,
    ).to(DEVICE)

    criterion = DiceFocalLoss(dice_weight=1.0, focal_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warmup
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6
    )

    # ── Training loop ──
    best_dice = 0.0
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)

    for epoch in range(EPOCHS):
        t0 = time.time()

        # Warmup LR
        if epoch < warmup_epochs:
            warmup_lr = LR * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # ── Train ──
        model.train()
        train_loss = 0.0
        n_train = 0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            n_train += images.size(0)

        avg_train_loss = train_loss / n_train

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        dice_disc_sum = 0.0
        dice_cup_sum = 0.0
        cdr_errors = []
        n_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, masks)
                bs = images.size(0)
                val_loss += loss.item() * bs

                probs = torch.sigmoid(logits)
                preds_bin = (probs > 0.5).float()

                for i in range(bs):
                    dice_disc_sum += compute_dice(preds_bin[i, 0], masks[i, 0])
                    dice_cup_sum += compute_dice(preds_bin[i, 1], masks[i, 1])

                    # CDR
                    pd_np = preds_bin[i, 0].cpu().numpy()
                    pc_np = preds_bin[i, 1].cpu().numpy()
                    gd_np = masks[i, 0].cpu().numpy()
                    gc_np = masks[i, 1].cpu().numpy()

                    pred_cdr = GlaucomaSegmentor.compute_cdr(pd_np, pc_np)
                    gt_cdr = GlaucomaSegmentor.compute_cdr(gd_np, gc_np)
                    if pred_cdr is not None and gt_cdr is not None:
                        cdr_errors.append(abs(pred_cdr - gt_cdr))

                n_val += bs

        avg_val_loss = val_loss / n_val
        avg_dice_disc = dice_disc_sum / n_val
        avg_dice_cup = dice_cup_sum / n_val
        avg_dice = (avg_dice_disc + avg_dice_cup) / 2.0
        avg_cdr_mae = np.mean(cdr_errors) if cdr_errors else 0.0

        # Step scheduler after warmup
        if epoch >= warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Dice(disc/cup/avg): {avg_dice_disc:.4f}/{avg_dice_cup:.4f}/{avg_dice:.4f} | "
            f"CDR MAE: {avg_cdr_mae:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), BEST_CKPT)
            logger.info(f"  >> New best model saved (Dice={best_dice:.4f})")

    logger.info("=" * 70)
    logger.info(f"Training complete.")
    logger.info(f"Best avg Dice: {best_dice:.4f} at epoch {best_epoch}")
    logger.info(f"Checkpoint saved to: {BEST_CKPT}")
    logger.info("=" * 70)


def _check_pseudo_masks(dataset: REFUGE2SegDataset):
    """Log statistics about the generated pseudo-masks."""
    disc_areas = []
    cup_areas = []
    cdrs = []
    for i in range(min(50, len(dataset.masks_disc))):
        d = dataset.masks_disc[i]
        c = dataset.masks_cup[i]
        da = d.sum()
        ca = c.sum()
        disc_areas.append(da)
        cup_areas.append(ca)
        cdr = GlaucomaSegmentor.compute_cdr(d, c)
        if cdr is not None:
            cdrs.append(cdr)

    logger.info(f"Pseudo-mask stats (first 50 images):")
    logger.info(f"  Disc area: mean={np.mean(disc_areas):.0f}, "
                f"min={np.min(disc_areas):.0f}, max={np.max(disc_areas):.0f}")
    logger.info(f"  Cup area:  mean={np.mean(cup_areas):.0f}, "
                f"min={np.min(cup_areas):.0f}, max={np.max(cup_areas):.0f}")
    if cdrs:
        logger.info(f"  CDR:       mean={np.mean(cdrs):.3f}, "
                    f"min={np.min(cdrs):.3f}, max={np.max(cdrs):.3f}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
