"""
Cross-dataset validation: Evaluate the APTOS-trained DR model on the IDRiD dataset.

IDRiD (Indian Diabetic Retinopathy Image Dataset) uses the same 0-4 DR grading scale
as APTOS, making it ideal for testing model generalization to unseen clinical data.

Usage:
    .venv/bin/python ml/scripts/validate_cross_dataset.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

from server.services.inference_v2 import DRGrader
from ml.data.augmentations import get_val_transforms

# ── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "dr_aptos" / "best.pth"
IDRID_DATA_DIR = PROJECT_ROOT / "data" / "idrid"
IDRID_IMAGES_DIR = IDRID_DATA_DIR / "Imagenes" / "Imagenes"
IDRID_LABELS_CSV = IDRID_DATA_DIR / "idrid_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "evaluation" / "cross_dataset"

DR_GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
NUM_CLASSES = 5


# ── Dataset ──────────────────────────────────────────────────────────────────
class IDRiDDataset(Dataset):
    """
    IDRiD dataset loader.
    Loads images and labels from the IDRiD CSV + image directory.
    Supports filtering to 'test' images (those with 'test' suffix) or 'all'.
    """

    def __init__(self, images_dir: Path, labels_csv: Path, split: str = "all",
                 transform=None):
        self.images_dir = images_dir
        self.transform = transform or get_val_transforms(224)
        self.samples = []

        # Parse CSV labels
        label_map = {}
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_code = row["id_code"].strip()
                diagnosis = int(row["diagnosis"])
                label_map[id_code] = diagnosis

        # Match images to labels
        available_images = sorted(images_dir.glob("*.jpg"))
        for img_path in available_images:
            stem = img_path.stem  # e.g. IDRiD_001 or IDRiD_001test
            is_test = stem.endswith("test")

            if split == "test" and not is_test:
                continue
            if split == "train" and is_test:
                continue

            # Map back to label ID
            label_id = stem.replace("test", "")
            if label_id in label_map:
                self.samples.append((img_path, label_map[label_id]))

        print(f"IDRiD [{split}]: {len(self.samples)} images loaded")
        # Print class distribution
        dist = defaultdict(int)
        for _, g in self.samples:
            dist[g] += 1
        for g in range(NUM_CLASSES):
            print(f"  Grade {g} ({DR_GRADE_NAMES[g]}): {dist[g]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, grade = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        transformed = self.transform(image=img_np)
        img_tensor = transformed["image"]
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()
        return img_tensor, grade, str(img_path.name)


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_probs):
    """Compute QWK, AUC, sensitivity, specificity, and per-class metrics."""
    results = {}

    # Quadratic Weighted Kappa
    results["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # AUC-ROC (one-vs-rest, macro)
    try:
        results["auc"] = roc_auc_score(
            y_true, y_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        results["auc"] = float("nan")

    # Binary referable DR metrics (grade >= 2)
    y_true_bin = (np.array(y_true) >= 2).astype(int)
    y_pred_bin = (np.array(y_pred) >= 2).astype(int)

    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    results["sensitivity"] = tp / max(tp + fn, 1)
    results["specificity"] = tn / max(tn + fp, 1)
    results["accuracy"] = (tp + tn) / max(tp + tn + fp + fn, 1)

    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class_acc = {}
    for i in range(NUM_CLASSES):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc[DR_GRADE_NAMES[i]] = correct / max(total, 1)
    results["per_class_accuracy"] = per_class_acc

    return results


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(DR_GRADE_NAMES, fontsize=9)
    ax.set_yticklabels(DR_GRADE_NAMES, fontsize=9)
    ax.set_xlabel("Predicted Grade", fontsize=11)
    ax.set_ylabel("True Grade", fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")


def plot_error_analysis(y_true, y_pred, y_probs, output_path: Path):
    """Plot per-class confidence distribution and error patterns."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confidence distribution for correct vs incorrect
    correct_mask = y_true == y_pred
    correct_conf = y_probs[np.arange(len(y_pred)), y_pred][correct_mask]
    incorrect_conf = y_probs[np.arange(len(y_pred)), y_pred][~correct_mask]

    ax = axes[0]
    bins = np.linspace(0, 1, 25)
    if len(correct_conf) > 0:
        ax.hist(correct_conf, bins=bins, alpha=0.7, label=f"Correct (n={len(correct_conf)})",
                color="#2ecc71", edgecolor="white")
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=bins, alpha=0.7, label=f"Incorrect (n={len(incorrect_conf)})",
                color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()

    # 2. Per-class accuracy bar chart
    ax = axes[1]
    per_class_correct = []
    per_class_total = []
    for g in range(NUM_CLASSES):
        mask = y_true == g
        total = mask.sum()
        correct = (y_pred[mask] == g).sum() if total > 0 else 0
        per_class_correct.append(correct)
        per_class_total.append(total)

    accs = [c / max(t, 1) for c, t in zip(per_class_correct, per_class_total)]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    bars = ax.bar(range(NUM_CLASSES), accs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, acc, total in zip(bars, accs, per_class_total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}\n(n={total})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(DR_GRADE_NAMES, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    ax.set_ylim(0, 1.15)

    # 3. Off-by-one vs larger errors
    ax = axes[2]
    errors = np.abs(y_true - y_pred)
    error_counts = [np.sum(errors == i) for i in range(5)]
    labels_err = ["Exact", "Off-by-1", "Off-by-2", "Off-by-3", "Off-by-4"]
    colors_err = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
    nonzero = [(l, c, col) for l, c, col in zip(labels_err, error_counts, colors_err) if c > 0]
    if nonzero:
        ax.pie([x[1] for x in nonzero], labels=[x[0] for x in nonzero],
               colors=[x[2] for x in nonzero], autopct="%1.1f%%", startangle=90)
    ax.set_title("Error Magnitude Distribution")

    fig.suptitle("Cross-Dataset Error Analysis (APTOS model on IDRiD)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error analysis: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {CHECKPOINT_PATH}...")
    model = DRGrader(num_classes=NUM_CLASSES)
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        best_qwk = checkpoint.get("best_qwk", "?")
        print(f"Loaded best checkpoint from epoch {epoch} (train QWK: {best_qwk})")
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load IDRiD dataset — use ALL images for maximum evaluation coverage
    print(f"\nLoading IDRiD dataset from {IDRID_DATA_DIR}...")
    dataset = IDRiDDataset(
        images_dir=IDRID_IMAGES_DIR,
        labels_csv=IDRID_LABELS_CSV,
        split="all",  # Use all 455 images
        transform=get_val_transforms(224),
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Run inference
    print(f"\nRunning inference on {len(dataset)} IDRiD images...")
    all_preds = []
    all_labels = []
    all_probs = []
    all_names = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels, names) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_names.extend(names)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * loader.batch_size}/{len(dataset)} images...")

    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s ({elapsed / len(dataset) * 1000:.0f}ms/image)")

    all_probs = np.concatenate(all_probs, axis=0)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Print results
    print("\n" + "=" * 65)
    print("  CROSS-DATASET VALIDATION: APTOS-trained model on IDRiD")
    print("=" * 65)
    print(f"  Dataset:       IDRiD ({len(dataset)} images)")
    print(f"  Model:         EfficientNet-B3 (trained on APTOS 2019)")
    print(f"  Device:        {device}")
    print(f"  Inference:     {elapsed:.1f}s total, {elapsed / len(dataset) * 1000:.0f}ms/image")
    print("-" * 65)
    print(f"  QWK:           {metrics['qwk']:.4f}")
    print(f"  AUC-ROC:       {metrics['auc']:.4f}")
    print(f"  Sensitivity:   {metrics['sensitivity']:.4f}  (referable DR: grade >= 2)")
    print(f"  Specificity:   {metrics['specificity']:.4f}  (referable DR: grade >= 2)")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}  (referable DR binary)")
    print("-" * 65)
    print("  Per-class accuracy:")
    for name, acc in metrics["per_class_accuracy"].items():
        print(f"    {name:12s}: {acc:.4f}")
    print("=" * 65)

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=DR_GRADE_NAMES, zero_division=0
    ))

    # Generate plots
    plot_confusion_matrix(
        all_labels, all_preds,
        OUTPUT_DIR / "confusion_matrix_idrid.png",
        "APTOS Model on IDRiD - Confusion Matrix"
    )
    plot_error_analysis(
        all_labels, all_preds, all_probs,
        OUTPUT_DIR / "error_analysis_idrid.png",
    )

    # Save results JSON
    results = {
        "dataset": "IDRiD",
        "num_images": len(dataset),
        "model": "EfficientNet-B3 (APTOS-trained)",
        "checkpoint": str(CHECKPOINT_PATH),
        "device": device,
        "inference_time_s": round(elapsed, 2),
        "metrics": {
            "qwk": round(metrics["qwk"], 4),
            "auc_roc": round(metrics["auc"], 4),
            "sensitivity": round(metrics["sensitivity"], 4),
            "specificity": round(metrics["specificity"], 4),
            "accuracy": round(metrics["accuracy"], 4),
            "per_class_accuracy": {
                k: round(v, 4) for k, v in metrics["per_class_accuracy"].items()
            },
        },
    }

    results_path = OUTPUT_DIR / "cross_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results JSON: {results_path}")

    # Save misclassified images list for further analysis
    misclassified = []
    for i, (name, true_g, pred_g) in enumerate(zip(all_names, all_labels, all_preds)):
        if true_g != pred_g:
            misclassified.append({
                "image": name,
                "true_grade": true_g,
                "true_label": DR_GRADE_NAMES[true_g],
                "pred_grade": pred_g,
                "pred_label": DR_GRADE_NAMES[pred_g],
                "confidence": round(float(all_probs[i, pred_g]), 4),
                "off_by": abs(true_g - pred_g),
            })

    misclassified.sort(key=lambda x: -x["off_by"])
    misc_path = OUTPUT_DIR / "misclassified_idrid.json"
    with open(misc_path, "w") as f:
        json.dump(misclassified, f, indent=2)
    print(f"Saved {len(misclassified)} misclassified cases: {misc_path}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
