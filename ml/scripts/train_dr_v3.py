"""
DR Grading v3: Train on EyePACS (35K images) + APTOS + IDRiD.
Target: QWK > 0.92, Accuracy > 85% on APTOS test set.

Usage:
    python -m ml.scripts.train_dr_v3
"""

import os, sys, time, random, logging
os.environ["PYTHONUNBUFFERED"] = "1"

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", force=True)
log = logging.getLogger("train_dr_v3")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


# ── Dataset ────────────────────────────────────────────────────────────

class DRDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        transformed = self.transform(image=img)
        return transformed["image"], label


def get_train_transform(img_size=224):
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


def get_val_transform(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Data Loading ───────────────────────────────────────────────────────

def load_eyepacs():
    """Load EyePACS resized cropped dataset."""
    csv_path = PROJECT_ROOT / "data/eyepacs_resized/trainLabels_cropped.csv"
    img_dir = PROJECT_ROOT / "data/eyepacs_resized/resized_train_cropped/resized_train_cropped"

    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        img_path = img_dir / f"{row['image']}.jpeg"
        if img_path.exists():
            samples.append((str(img_path), int(row["level"])))
    return samples


def load_aptos_split(split):
    """Load APTOS split from class folders."""
    samples = []
    root = PROJECT_ROOT / f"data/aptos_split/{split}"
    if not root.exists():
        return samples
    for grade_dir in sorted(root.iterdir()):
        if not grade_dir.is_dir():
            continue
        grade = int(grade_dir.name)
        for img_path in sorted(grade_dir.glob("*.png")):
            samples.append((str(img_path), grade))
    return samples


def load_idrid():
    """Load IDRiD dataset."""
    csv_path = PROJECT_ROOT / "data/idrid/idrid_labels.csv"
    img_dir = PROJECT_ROOT / "data/idrid/Imagenes/Imagenes"
    samples = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        img_path = img_dir / f"{row['id_code']}.jpg"
        if img_path.exists() and "test" not in str(img_path).lower():
            samples.append((str(img_path), int(row["diagnosis"])))
    return samples


def make_weighted_sampler(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    weights = [total / (n_classes * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Mixup ──────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and random.random() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels)
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, name="Val"):
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

    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc = (all_preds == all_labels).mean()

    ref_labels = (all_labels >= 2).astype(int)
    ref_probs = all_probs[:, 2:].sum(axis=1)
    auc = roc_auc_score(ref_labels, ref_probs) if ref_labels.sum() > 0 else 0

    fpr, tpr, _ = roc_curve(ref_labels, ref_probs)
    idx = np.where(tpr >= 0.90)[0]
    sens = tpr[idx[0]] if len(idx) > 0 else 0
    spec = 1 - fpr[idx[0]] if len(idx) > 0 else 0

    log.info(f"  [{name}] QWK={qwk:.4f}  AUC={auc:.4f}  Acc={acc:.4f}  Sens={sens:.3f}  Spec={spec:.3f}")
    for g in range(5):
        mask = all_labels == g
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == g).mean()
            log.info(f"    {GRADE_NAMES[g]:20s}: {class_acc:.3f} ({int(class_acc*mask.sum())}/{mask.sum()})")

    return {"qwk": qwk, "auc": auc, "accuracy": acc, "sensitivity": sens, "specificity": spec}


# ── Main ───────────────────────────────────────────────────────────────

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load all data
    eyepacs = load_eyepacs()
    aptos_train = load_aptos_split("train")
    aptos_val = load_aptos_split("val")
    aptos_test = load_aptos_split("test")
    idrid = load_idrid()

    log.info(f"EyePACS: {len(eyepacs)}")
    log.info(f"APTOS train: {len(aptos_train)}")
    log.info(f"APTOS val: {len(aptos_val)}")
    log.info(f"APTOS test: {len(aptos_test)}")
    log.info(f"IDRiD: {len(idrid)}")

    # Split EyePACS 90/10 for train/val
    random.shuffle(eyepacs)
    ep_split = int(len(eyepacs) * 0.9)
    eyepacs_train = eyepacs[:ep_split]
    eyepacs_val = eyepacs[ep_split:]

    # Combine all training data
    train_samples = eyepacs_train + aptos_train + idrid
    val_samples = eyepacs_val + aptos_val
    random.shuffle(train_samples)

    log.info(f"\nTotal train: {len(train_samples)}")
    log.info(f"Total val: {len(val_samples)}")
    log.info(f"Train distribution: {Counter([s[1] for s in train_samples])}")

    # Config
    img_size = 224
    batch_size = 32
    epochs = 15
    lr = 2e-4
    weight_decay = 0.05
    ckpt_dir = PROJECT_ROOT / "checkpoints/dr_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Model: EfficientNet-B3 (fast, proven)
    log.info("\nCreating EfficientNet-B3...")
    model = timm.create_model("efficientnet_b3.ra2_in1k", pretrained=True, num_classes=5)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {params/1e6:.1f}M")

    # Data
    train_ds = DRDataset(train_samples, get_train_transform(img_size))
    val_ds = DRDataset(val_samples, get_val_transform(img_size))
    test_ds = DRDataset(aptos_test, get_val_transform(img_size))

    sampler = make_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Training
    best_qwk = 0
    log.info(f"\nTraining for {epochs} epochs on {len(train_samples)} images...")
    log.info("=" * 70)

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, "Val")
        scheduler.step()
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        log.info(f"Epoch {epoch+1:02d}/{epochs} | loss={train_loss:.4f} acc={train_acc:.3f} | "
                 f"val_qwk={val_metrics['qwk']:.4f} val_acc={val_metrics['accuracy']:.3f} | "
                 f"lr={cur_lr:.2e} | {dt:.0f}s")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "qwk": best_qwk,
                "metrics": val_metrics,
            }, ckpt_dir / "best.pth")
            log.info(f"  >> New best QWK: {best_qwk:.4f} (saved)")

        # Save periodic checkpoints for ensemble
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pth")

    # Final evaluation
    log.info("\n" + "=" * 70)
    log.info("Loading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Best model from epoch {ckpt['epoch']} (val QWK: {ckpt['qwk']:.4f})")

    log.info("\n--- APTOS Test Set (550 images) ---")
    test_metrics = evaluate(model, test_loader, device, "APTOS-Test")

    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info(f"  Training data: {len(train_samples)} images (EyePACS + APTOS + IDRiD)")
    log.info(f"  Best val QWK: {ckpt['qwk']:.4f}")
    log.info(f"  APTOS test QWK: {test_metrics['qwk']:.4f}")
    log.info(f"  APTOS test AUC: {test_metrics['auc']:.4f}")
    log.info(f"  APTOS test Acc: {test_metrics['accuracy']:.4f}")
    log.info(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {test_metrics['specificity']:.4f}")
    log.info(f"  Previous best (v1 EfficientNet): QWK=0.892, Acc=80.7%")
    log.info(f"  Previous best (v2 ViT-Base):     QWK=0.894, Acc=80.9%")
    log.info(f"  Previous best (ensemble):        QWK=0.902, Acc=82.4%")
    log.info(f"  Checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
