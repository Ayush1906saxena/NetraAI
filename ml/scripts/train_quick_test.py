"""
Quick training test with synthetic data + EfficientNet backbone.
Validates the full pipeline works end-to-end without needing
Kaggle datasets or RETFound weights.

Usage:
    python -m ml.scripts.train_quick_test
"""

import os
import sys
import shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import cohen_kappa_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.dataset import FundusDataset
from ml.data.preprocess import FundusPreprocessor
from ml.training.metrics import compute_qwk, compute_auc


def create_synthetic_dataset(root: Path, num_images_per_class: int = 20):
    """Create synthetic fundus-like images for pipeline testing."""
    print(f"Creating synthetic dataset at {root}...")
    preprocessor = FundusPreprocessor()

    for split in ["train", "val", "test"]:
        n = num_images_per_class if split == "train" else num_images_per_class // 2
        for grade in range(5):
            class_dir = root / split / str(grade)
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                # Create a circular fundus-like image
                img = np.zeros((256, 256, 3), dtype=np.uint8)

                # Dark background with circular bright region (fundus)
                center = (128, 128)
                radius = 100
                cv2.circle(img, center, radius, (80 + grade * 30, 60 + grade * 20, 40 + grade * 15), -1)

                # Add some random "vessel-like" lines
                np.random.seed(grade * 100 + i)
                for _ in range(5 + grade * 3):
                    pt1 = (
                        np.random.randint(60, 196),
                        np.random.randint(60, 196),
                    )
                    pt2 = (
                        np.random.randint(60, 196),
                        np.random.randint(60, 196),
                    )
                    color = (
                        np.random.randint(100, 200),
                        np.random.randint(20, 80),
                        np.random.randint(20, 60),
                    )
                    cv2.line(img, pt1, pt2, color, 1)

                # Add "lesion-like" spots for higher grades
                for _ in range(grade * 4):
                    cx = np.random.randint(60, 196)
                    cy = np.random.randint(60, 196)
                    r = np.random.randint(2, 8)
                    cv2.circle(img, (cx, cy), r, (200, 50, 50), -1)

                # Save
                path = class_dir / f"synth_{split}_{grade}_{i:03d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    total = sum(1 for _ in root.rglob("*.png"))
    print(f"Created {total} synthetic images")
    return root


class SimpleEfficientNetDR(nn.Module):
    """Lightweight DR grader using EfficientNet-B0 for pipeline testing."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        backbone = efficientnet_b0(weights=None)  # No download needed for pipeline test
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    qwk = compute_qwk(np.array(all_labels), np.array(all_preds))
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

    return {
        "qwk": qwk,
        "auc": auc,
        "accuracy": (all_preds == all_labels).mean(),
    }


def main():
    print("=" * 60)
    print("NETRA AI — Quick Pipeline Validation Test")
    print("=" * 60)

    # Config
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    epochs = 5
    batch_size = 8
    lr = 1e-3
    data_root = Path("data/synthetic_test")

    # 1. Create synthetic dataset
    if data_root.exists():
        shutil.rmtree(data_root)
    create_synthetic_dataset(data_root, num_images_per_class=30)

    # 2. Create datasets and loaders
    print("\nLoading datasets...")
    train_ds = FundusDataset(data_root, "train")
    val_ds = FundusDataset(data_root, "val")
    test_ds = FundusDataset(data_root, "test")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_ds.get_weighted_sampler(),
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    # 3. Create model
    print("\nCreating EfficientNet-B0 model...")
    model = SimpleEfficientNetDR(num_classes=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    # 4. Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5. Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    best_qwk = -1
    for epoch in range(epochs):
        train_loss, train_qwk = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        is_best = val_metrics["qwk"] > best_qwk
        if is_best:
            best_qwk = val_metrics["qwk"]
            # Save best checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/quick_test_best.pth")

        print(
            f"  Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train QWK: {train_qwk:.3f} | "
            f"Val QWK: {val_metrics['qwk']:.3f} | "
            f"Val AUC: {val_metrics['auc']:.3f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f}"
            f"{' *' if is_best else ''}"
        )

    # 6. Final test evaluation
    print("-" * 60)
    model.load_state_dict(torch.load("checkpoints/quick_test_best.pth", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTEST RESULTS:")
    print(f"  QWK:      {test_metrics['qwk']:.4f}")
    print(f"  AUC:      {test_metrics['auc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    # 7. Test GradCAM
    print("\nTesting GradCAM generation...")
    from ml.evaluation.gradcam import generate_gradcam
    sample_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    sample_tensor = torch.randn(1, 3, 224, 224).to(device)
    gradcam_bytes = generate_gradcam(model, sample_tensor, sample_img)
    print(f"  GradCAM output: {len(gradcam_bytes)} bytes (PNG)")

    # 8. Test preprocessing pipeline
    print("\nTesting preprocessing pipeline...")
    preprocessor = FundusPreprocessor()
    test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    processed = preprocessor.process(test_img, target_size=224)
    print(f"  Input: {test_img.shape} -> Output: {processed.shape}")

    print("\n" + "=" * 60)
    print("Pipeline validation COMPLETE")
    print("=" * 60)

    # Cleanup
    shutil.rmtree(data_root)
    print("Cleaned up synthetic data.")


if __name__ == "__main__":
    main()
