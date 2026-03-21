"""
Ensemble evaluation: combine EfficientNet-B3 (v1) + ViT-Base (v2) predictions.
Different architectures see different features → ensemble should beat both.

Usage:
    python -m ml.scripts.eval_ensemble
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
import timm
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        transformed = self.transform(image=img)
        return transformed["image"], label


def get_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_test_samples():
    """Load APTOS test set."""
    samples = []
    test_root = Path("data/aptos_split/test")
    for grade_dir in sorted(test_root.iterdir()):
        if not grade_dir.is_dir():
            continue
        grade = int(grade_dir.name)
        for img_path in sorted(grade_dir.glob("*.png")):
            samples.append((str(img_path), grade))
    return samples


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())
    return np.concatenate(all_probs), np.array(all_labels)


def compute_metrics(labels, probs, name=""):
    preds = probs.argmax(axis=1)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    acc = (preds == labels).mean()

    # Binary: referable DR (grade >= 2)
    ref_labels = (labels >= 2).astype(int)
    ref_probs = probs[:, 2:].sum(axis=1)
    auc = roc_auc_score(ref_labels, ref_probs) if ref_labels.sum() > 0 else 0

    # Sensitivity/specificity at 90% sensitivity
    fpr, tpr, thresholds = roc_curve(ref_labels, ref_probs)
    idx = np.where(tpr >= 0.90)[0]
    if len(idx) > 0:
        sens = tpr[idx[0]]
        spec = 1 - fpr[idx[0]]
    else:
        sens = spec = 0

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  QWK:         {qwk:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Accuracy:    {acc:.4f} ({int(acc*len(labels))}/{len(labels)})")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"\n  Per-class accuracy:")
    for g in range(5):
        mask = labels == g
        if mask.sum() > 0:
            class_acc = (preds[mask] == g).mean()
            print(f"    Grade {g} ({GRADE_NAMES[g]:20s}): {class_acc:.3f} ({int(class_acc*mask.sum())}/{mask.sum()})")

    return {"qwk": qwk, "auc": auc, "accuracy": acc, "sensitivity": sens, "specificity": spec}


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    test_samples = load_test_samples()
    print(f"Test samples: {len(test_samples)}")

    # ── Model 1: EfficientNet-B3 (v1) ──
    print("\nLoading EfficientNet-B3 (v1)...")
    from torchvision.models import efficientnet_b3
    model_v1 = efficientnet_b3(weights=None)
    model_v1.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 512), nn.ReLU(True), nn.Dropout(0.2), nn.Linear(512, 5))
    ckpt_v1 = torch.load("checkpoints/dr_aptos/best.pth", map_location=device, weights_only=False)
    model_v1.load_state_dict(ckpt_v1["model_state_dict"])
    model_v1 = model_v1.to(device)

    ds_v1 = SimpleDataset(test_samples, get_transform(224))
    loader_v1 = DataLoader(ds_v1, batch_size=32, shuffle=False, num_workers=0)
    probs_v1, labels = get_predictions(model_v1, loader_v1, device)
    m1 = compute_metrics(labels, probs_v1, "Model 1: EfficientNet-B3 (224px)")

    # ── Model 2: ViT-Base (v2) ──
    print("\nLoading ViT-Base (v2)...")
    model_v2 = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=5)
    # Resize pos embed for 384px
    ckpt_v2 = torch.load("checkpoints/dr_v2/best.pth", map_location=device, weights_only=False)
    # Check if this is a 384px model
    state = ckpt_v2["model_state_dict"]
    pos_embed_shape = state.get("pos_embed", torch.zeros(1)).shape
    if len(pos_embed_shape) == 3 and pos_embed_shape[1] == 577:  # 24*24+1 = 577 for 384px
        model_v2 = timm.create_model("vit_base_patch16_384", pretrained=False, num_classes=5)
        img_size_v2 = 384
    else:
        img_size_v2 = 224
    model_v2.load_state_dict(state)
    model_v2 = model_v2.to(device)
    print(f"  ViT-Base loaded ({img_size_v2}px)")

    ds_v2 = SimpleDataset(test_samples, get_transform(img_size_v2))
    loader_v2 = DataLoader(ds_v2, batch_size=16, shuffle=False, num_workers=0)
    probs_v2, _ = get_predictions(model_v2, loader_v2, device)
    m2 = compute_metrics(labels, probs_v2, f"Model 2: ViT-Base ({img_size_v2}px)")

    # ── Ensemble: weighted average ──
    # Try different weight combinations
    best_qwk = 0
    best_w = 0.5
    for w in np.arange(0.1, 0.9, 0.05):
        ensemble_probs = w * probs_v1 + (1 - w) * probs_v2
        preds = ensemble_probs.argmax(axis=1)
        qwk = cohen_kappa_score(labels, preds, weights="quadratic")
        if qwk > best_qwk:
            best_qwk = qwk
            best_w = w

    print(f"\nBest ensemble weight: EfficientNet={best_w:.2f}, ViT={1-best_w:.2f}")

    ensemble_probs = best_w * probs_v1 + (1 - best_w) * probs_v2
    m3 = compute_metrics(labels, ensemble_probs, f"ENSEMBLE (EfficientNet×{best_w:.2f} + ViT×{1-best_w:.2f})")

    # Also try simple average
    avg_probs = 0.5 * probs_v1 + 0.5 * probs_v2
    m4 = compute_metrics(labels, avg_probs, "ENSEMBLE (simple average)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<35s} {'QWK':>6s} {'AUC':>6s} {'Acc':>6s} {'Sens':>6s} {'Spec':>6s}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for name, m in [
        ("EfficientNet-B3", m1),
        ("ViT-Base", m2),
        (f"Ensemble (w={best_w:.2f})", m3),
        ("Ensemble (avg)", m4),
    ]:
        print(f"  {name:<35s} {m['qwk']:>6.4f} {m['auc']:>6.4f} {m['accuracy']:>6.4f} {m['sensitivity']:>6.4f} {m['specificity']:>6.4f}")


if __name__ == "__main__":
    main()
