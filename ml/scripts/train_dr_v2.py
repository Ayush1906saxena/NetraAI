#!/usr/bin/env python3
"""
DR Grading v2: ViT-Base/Large with combined APTOS+IDRiD data, Mixup/CutMix,
progressive resizing, cosine annealing with warmup.

Target: beat QWK 0.892 on APTOS test set.

Usage:
    python -m ml.scripts.train_dr_v2
"""

import csv
import logging
import os
import random
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("train_dr_v2")
# Ensure log output is flushed immediately
for handler in logging.root.handlers:
    handler.flush = lambda: sys.stderr.flush()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 5
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ────────────────────────────────────────────────────────────
class DRDataset(Dataset):
    """Combined DR dataset: list of (image_path, label) tuples."""

    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path_str, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # fallback: try PIL
            img = np.array(Image.open(path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


# ── Data loading helpers ───────────────────────────────────────────────
def load_aptos_split(split_dir):
    """Load from class-folder structure. Returns list of (path, label)."""
    samples = []
    split_dir = Path(split_dir)
    for cls in range(NUM_CLASSES):
        cls_dir = split_dir / str(cls)
        if not cls_dir.exists():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                samples.append((str(f), cls))
    return samples


def load_idrid(data_root):
    """Load IDRiD dataset. Returns train_samples, test_samples."""
    data_root = Path(data_root)
    csv_path = data_root / "idrid_labels.csv"
    img_dir = data_root / "Imagenes" / "Imagenes"

    # Parse CSV
    id_to_label = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_code = row["id_code"].strip()
            diag = int(row["diagnosis"].strip())
            id_to_label[id_code] = diag

    train_samples = []
    test_samples = []
    for img_path in sorted(img_dir.glob("IDRiD_*.jpg")):
        fname = img_path.stem  # e.g. IDRiD_001 or IDRiD_001test
        if fname.endswith("test"):
            # test image — look up base id
            base_id = fname.replace("test", "")
            if base_id in id_to_label:
                test_samples.append((str(img_path), id_to_label[base_id]))
        else:
            if fname in id_to_label:
                train_samples.append((str(img_path), id_to_label[fname]))

    return train_samples, test_samples


# ── Transforms ─────────────────────────────────────────────────────────
def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-180, 180), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Mixup / CutMix ────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    """Mixup: blend two samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut and paste patches between samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    # Adjust lambda to actual area ratio
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Weighted sampler ───────────────────────────────────────────────────
def make_weighted_sampler(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    class_weights = {c: total / (NUM_CLASSES * cnt) for c, cnt in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ── Evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, dataset_name=""):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # QWK
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    # Per-class accuracy, sensitivity, specificity
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per_class_acc = []
    sensitivities = []
    specificities = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc_c = tp / cm[c, :].sum() if cm[c, :].sum() > 0 else 0
        per_class_acc.append(acc_c)
        sensitivities.append(sens)
        specificities.append(spec)

    overall_acc = (all_preds == all_labels).mean()

    log.info(f"  [{dataset_name}] QWK={qwk:.4f}  AUC={auc:.4f}  Acc={overall_acc:.4f}")
    for c in range(NUM_CLASSES):
        log.info(f"    Class {c}: Acc={per_class_acc[c]:.3f}  Sens={sensitivities[c]:.3f}  Spec={specificities[c]:.3f}")

    return {
        "qwk": qwk,
        "auc": auc,
        "accuracy": overall_acc,
        "per_class_acc": per_class_acc,
        "sensitivity": sensitivities,
        "specificity": specificities,
        "confusion_matrix": cm,
    }


# ── Training ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    scaler=None, grad_accum=1, use_mixup=True):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Mixup / CutMix with probability 0.5
        apply_mix = use_mixup and random.random() < 0.5
        if apply_mix:
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=1.0)

        logits = model(imgs)

        if apply_mix:
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, labels)

        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum * imgs.size(0)
        if not apply_mix:
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


# ── Cosine schedule with warmup ───────────────────────────────────────
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ── Resize positional embeddings for progressive resizing ──────────────
def resize_pos_embed(model, new_img_size, patch_size=16):
    """Resize positional embeddings for a timm ViT model to a new image size."""
    pos_embed = model.pos_embed  # (1, num_tokens, embed_dim)
    num_tokens = pos_embed.shape[1]
    embed_dim = pos_embed.shape[2]

    # ViT has 1 cls token + grid tokens
    cls_token = pos_embed[:, :1, :]
    grid_embed = pos_embed[:, 1:, :]

    old_grid_size = int(grid_embed.shape[1] ** 0.5)
    new_grid_size = new_img_size // patch_size

    if old_grid_size == new_grid_size:
        return  # no resize needed

    log.info(f"Resizing pos_embed from {old_grid_size}x{old_grid_size} to {new_grid_size}x{new_grid_size}")
    grid_embed = grid_embed.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
    grid_embed = F.interpolate(grid_embed, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False)
    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, embed_dim)

    new_pos_embed = torch.cat([cls_token, grid_embed], dim=1)
    model.pos_embed = nn.Parameter(new_pos_embed)

    # Also update the patch_embed's img_size so timm doesn't assert
    if hasattr(model, "patch_embed"):
        model.patch_embed.img_size = (new_img_size, new_img_size)
        model.patch_embed.grid_size = (new_grid_size, new_grid_size)
        model.patch_embed.num_patches = new_grid_size * new_grid_size


# ── Main ───────────────────────────────────────────────────────────────
def main():
    set_seed()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────
    aptos_root = PROJECT_ROOT / "data" / "aptos_split"
    idrid_root = PROJECT_ROOT / "data" / "idrid"

    aptos_train = load_aptos_split(aptos_root / "train")
    aptos_val = load_aptos_split(aptos_root / "val")
    aptos_test = load_aptos_split(aptos_root / "test")

    idrid_train, idrid_test = load_idrid(idrid_root)

    # Split IDRiD train 80/20
    random.shuffle(idrid_train)
    split_idx = int(0.8 * len(idrid_train))
    idrid_train_split = idrid_train[:split_idx]
    idrid_val_split = idrid_train[split_idx:]

    # Combine
    train_samples = aptos_train + idrid_train_split
    val_samples = aptos_val + idrid_val_split

    random.shuffle(train_samples)

    log.info(f"Training samples: {len(train_samples)} (APTOS: {len(aptos_train)}, IDRiD: {len(idrid_train_split)})")
    log.info(f"Validation samples: {len(val_samples)} (APTOS: {len(aptos_val)}, IDRiD: {len(idrid_val_split)})")
    log.info(f"APTOS test samples: {len(aptos_test)}")
    log.info(f"IDRiD test samples: {len(idrid_test)}")

    # Class distribution
    train_labels = [s[1] for s in train_samples]
    log.info(f"Train class distribution: {Counter(train_labels)}")

    # ── Create model ───────────────────────────────────────────────────
    batch_size = 16

    log.info("Creating ViT-Base (ImageNet-21k pretrained)...")
    model = timm.create_model(
        "vit_base_patch16_224.augreg_in21k_ft_in1k",
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
    )
    model = model.to(device)
    model_name = "ViT-Base"
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"Model: {model_name}, params: {total_params:.1f}M (trainable: {trainable_params:.1f}M)")

    # ── Training config ────────────────────────────────────────────────
    lr = 1e-4
    weight_decay = 0.05
    label_smoothing = 0.1
    warmup_epochs = 3
    phase1_epochs = 12  # 224px
    phase2_epochs = 8   # 384px
    grad_accum = max(1, 32 // batch_size)  # effective batch = 32

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "dr_v2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_qwk = 0.0
    best_epoch = -1

    # ── Phase 1: 224px ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1: Training at 224x224")
    log.info("=" * 60)

    img_size = 224
    train_ds = DRDataset(train_samples, transform=get_train_transform(img_size))
    val_ds = DRDataset(val_samples, transform=get_val_transform(img_size))
    test_ds = DRDataset(aptos_test, transform=get_val_transform(img_size))
    idrid_test_ds = DRDataset(idrid_test, transform=get_val_transform(img_size))

    sampler = make_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=0, pin_memory=False)
    idrid_test_loader = DataLoader(idrid_test_ds, batch_size=batch_size * 2, shuffle=False,
                                   num_workers=0, pin_memory=False)

    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs, phase1_epochs)

    for epoch in range(phase1_epochs):
        t0 = time.time()
        scheduler.step(epoch)
        cur_lr = scheduler.get_lr()[0]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_accum=grad_accum, use_mixup=True,
        )

        val_metrics = evaluate(model, val_loader, device, "Val")
        dt = time.time() - t0

        log.info(f"Epoch {epoch+1:02d}/{phase1_epochs} (224px) | "
                 f"loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                 f"val_qwk={val_metrics['qwk']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
                 f"lr={cur_lr:.2e} | {dt:.0f}s")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "qwk": best_qwk,
                "phase": 1,
                "model_name": model_name,
            }, ckpt_dir / "best.pth")
            log.info(f"  >> New best QWK: {best_qwk:.4f} (saved)")

    # ── Phase 2: 384px ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2: Training at 384x384")
    log.info("=" * 60)

    # Load best Phase 1 checkpoint before resizing
    phase1_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(phase1_ckpt["model_state_dict"])
    log.info(f"Loaded Phase 1 best (QWK={phase1_ckpt.get('qwk', '?')})")

    # Resize positional embeddings + patch embed for 384px
    resize_pos_embed(model, 384, patch_size=16)
    model = model.to(device)

    # Reduce batch size for larger images
    batch_size_384 = max(2, batch_size // 2)
    grad_accum_384 = max(1, 32 // batch_size_384)
    img_size = 384

    train_ds = DRDataset(train_samples, transform=get_train_transform(img_size))
    val_ds = DRDataset(val_samples, transform=get_val_transform(img_size))
    test_ds = DRDataset(aptos_test, transform=get_val_transform(img_size))
    idrid_test_ds = DRDataset(idrid_test, transform=get_val_transform(img_size))

    sampler = make_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size_384, sampler=sampler,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_384 * 2, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size_384 * 2, shuffle=False,
                             num_workers=0, pin_memory=False)
    idrid_test_loader = DataLoader(idrid_test_ds, batch_size=batch_size_384 * 2, shuffle=False,
                                   num_workers=0, pin_memory=False)

    # Reset optimizer with lower LR for fine-tuning phase
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr * 0.5,  # lower LR for phase 2
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=2, total_epochs=phase2_epochs)

    for epoch in range(phase2_epochs):
        t0 = time.time()
        scheduler.step(epoch)
        cur_lr = scheduler.get_lr()[0]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            grad_accum=grad_accum_384, use_mixup=True,
        )

        val_metrics = evaluate(model, val_loader, device, "Val")
        dt = time.time() - t0

        global_epoch = phase1_epochs + epoch + 1
        log.info(f"Epoch {global_epoch:02d}/{phase1_epochs + phase2_epochs} (384px) | "
                 f"loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                 f"val_qwk={val_metrics['qwk']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
                 f"lr={cur_lr:.2e} | {dt:.0f}s")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            best_epoch = global_epoch
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "qwk": best_qwk,
                "phase": 2,
                "model_name": model_name,
                "img_size": 384,
            }, ckpt_dir / "best.pth")
            log.info(f"  >> New best QWK: {best_qwk:.4f} (saved)")

    # ── Final evaluation ───────────────────────────────────────────────
    log.info("=" * 60)
    log.info("FINAL EVALUATION")
    log.info("=" * 60)

    # Load best checkpoint
    ckpt = torch.load(ckpt_dir / "best.pth", weights_only=False, map_location=device)
    best_img_size = ckpt.get("img_size", 224)
    log.info(f"Loading best checkpoint from epoch {ckpt['epoch']} (QWK={ckpt['qwk']:.4f}, img_size={best_img_size})")

    # Rebuild model for correct img_size
    if best_img_size == 384:
        # Model already has 384 pos embeddings from phase 2 state dict
        pass
    else:
        # Need to resize back to 224
        resize_pos_embed(model, 224, patch_size=16)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # Rebuild test loaders with correct img_size
    test_ds = DRDataset(aptos_test, transform=get_val_transform(best_img_size))
    idrid_test_ds = DRDataset(idrid_test, transform=get_val_transform(best_img_size))
    eval_bs = batch_size_384 * 2 if best_img_size == 384 else batch_size * 2
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
    idrid_test_loader = DataLoader(idrid_test_ds, batch_size=eval_bs, shuffle=False, num_workers=0)

    log.info("\n--- APTOS Test Set ---")
    aptos_results = evaluate(model, test_loader, device, "APTOS-Test")

    log.info("\n--- IDRiD Test Set ---")
    idrid_results = evaluate(model, idrid_test_loader, device, "IDRiD-Test")

    # ── Summary ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Model: {model_name}")
    log.info(f"Best validation QWK: {best_qwk:.4f} at epoch {best_epoch}")
    log.info(f"")
    log.info(f"APTOS Test:  QWK={aptos_results['qwk']:.4f}  AUC={aptos_results['auc']:.4f}  Acc={aptos_results['accuracy']:.4f}")
    log.info(f"IDRiD Test:  QWK={idrid_results['qwk']:.4f}  AUC={idrid_results['auc']:.4f}  Acc={idrid_results['accuracy']:.4f}")
    log.info(f"")
    log.info(f"Previous best (EfficientNet-B3): QWK=0.892")
    log.info(f"This model ({model_name}):        QWK={aptos_results['qwk']:.4f}  ({'IMPROVED' if aptos_results['qwk'] > 0.892 else 'not improved'})")
    log.info(f"")
    log.info(f"Checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
