"""
Full evaluation pipeline for DR grading models.

Loads a checkpoint, runs inference on a test set, computes all clinical
metrics (QWK, AUC, sensitivity, specificity, per-class accuracy),
generates confusion matrix, and saves results to disk and optionally MLflow.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ml.data.augmentations import get_val_transforms
from ml.data.dataset import FundusDataset
from ml.evaluation.calibration import expected_calibration_error
from ml.evaluation.confusion import plot_confusion_matrix, per_class_metrics
from ml.evaluation.tta import TTAPredictor


DR_GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def load_model(checkpoint_path: str, device: str = "cpu", num_classes: int = 5):
    """
    Load a DR grading model from a checkpoint file.

    Supports both full state-dict checkpoints and training checkpoint dicts
    that contain 'model_state_dict' and 'config' keys.
    """
    from ml.models.retfound_wrapper import RETFoundDRGrader

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        config = ckpt.get("config", {})
        model = RETFoundDRGrader(
            num_classes=config.get("num_classes", num_classes),
            use_lora=False,
        )
        model.backbone.load_state_dict(ckpt["model_state_dict"])
    else:
        model = RETFoundDRGrader(num_classes=num_classes, use_lora=False)
        model.backbone.load_state_dict(ckpt)

    model.eval()
    return model.to(device)


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    use_tta: bool = False,
    tta_folds: int = 8,
) -> dict[str, np.ndarray]:
    """
    Run model inference on the entire dataloader.

    Returns:
        dict with keys:
            'logits': (N, C) raw logits
            'probs':  (N, C) softmax probabilities
            'preds':  (N,) predicted class indices
            'labels': (N,) ground-truth labels
    """
    all_logits = []
    all_labels = []

    tta_predictor = TTAPredictor(model, device=device, n_folds=tta_folds) if use_tta else None

    with torch.no_grad():
        for images, labels in dataloader:
            if use_tta and tta_predictor is not None:
                # TTA expects individual images as numpy arrays; batch workaround
                batch_logits = tta_predictor.predict_batch(images)
                all_logits.append(batch_logits)
            else:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())

            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Softmax probabilities
    probs = _softmax(logits)
    preds = np.argmax(probs, axis=1)

    return {
        "logits": logits,
        "probs": probs,
        "preds": preds,
        "labels": labels,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over axis=-1."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 5,
) -> dict[str, Any]:
    """
    Compute all clinically relevant metrics.

    Returns dict with:
        - qwk: quadratic weighted kappa
        - accuracy: overall accuracy
        - auc_macro: macro-averaged AUC (one-vs-rest)
        - auc_per_class: per-class AUC values
        - sensitivity: per-class sensitivity (recall)
        - specificity: per-class specificity
        - per_class_accuracy: per-class accuracy
        - referable_sensitivity: sensitivity for referable DR (grade >= 2)
        - referable_specificity: specificity for referable DR
        - ece: expected calibration error
        - classification_report: sklearn text report
    """
    # Quadratic Weighted Kappa (primary metric for ordinal DR grading)
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    # Overall accuracy
    acc = accuracy_score(labels, preds)

    # Per-class AUC (one-vs-rest)
    auc_per_class = {}
    try:
        for c in range(num_classes):
            binary_labels = (labels == c).astype(int)
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                auc_per_class[DR_GRADE_NAMES[c]] = float(
                    roc_auc_score(binary_labels, probs[:, c])
                )
            else:
                auc_per_class[DR_GRADE_NAMES[c]] = float("nan")
        auc_macro = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        auc_macro = float("nan")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Per-class sensitivity and specificity
    sensitivity = {}
    specificity = {}
    per_class_acc = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        class_acc = tp / cm[c, :].sum() if cm[c, :].sum() > 0 else 0.0

        sensitivity[DR_GRADE_NAMES[c]] = float(sens)
        specificity[DR_GRADE_NAMES[c]] = float(spec)
        per_class_acc[DR_GRADE_NAMES[c]] = float(class_acc)

    # Referable DR: grade >= 2 (moderate NPDR or worse)
    ref_labels = (labels >= 2).astype(int)
    ref_preds = (preds >= 2).astype(int)
    ref_tp = ((ref_labels == 1) & (ref_preds == 1)).sum()
    ref_fn = ((ref_labels == 1) & (ref_preds == 0)).sum()
    ref_fp = ((ref_labels == 0) & (ref_preds == 1)).sum()
    ref_tn = ((ref_labels == 0) & (ref_preds == 0)).sum()

    ref_sens = float(ref_tp / (ref_tp + ref_fn)) if (ref_tp + ref_fn) > 0 else 0.0
    ref_spec = float(ref_tn / (ref_tn + ref_fp)) if (ref_tn + ref_fp) > 0 else 0.0

    # Expected Calibration Error
    max_probs = probs.max(axis=1)
    correct = (preds == labels).astype(float)
    ece = expected_calibration_error(max_probs, correct)

    # Classification report
    report = classification_report(
        labels, preds, target_names=DR_GRADE_NAMES[:num_classes], digits=4
    )

    return {
        "qwk": float(qwk),
        "accuracy": float(acc),
        "auc_macro": float(auc_macro),
        "auc_per_class": auc_per_class,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "per_class_accuracy": per_class_acc,
        "referable_sensitivity": ref_sens,
        "referable_specificity": ref_spec,
        "ece": float(ece),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "num_samples": int(len(labels)),
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    num_classes: int = 5,
    use_tta: bool = False,
    log_to_mlflow: bool = False,
) -> dict[str, Any]:
    """
    Full evaluation pipeline: load model, run on test set, compute metrics,
    generate plots, save everything.

    Args:
        checkpoint_path: Path to model checkpoint.
        data_dir: Root data directory with test/{grade}/ structure.
        output_dir: Where to save results (metrics.json, confusion_matrix.png, etc.).
        device: 'cpu', 'cuda', or 'mps'.
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        img_size: Input image size.
        num_classes: Number of DR grades.
        use_tta: Whether to use test-time augmentation.
        log_to_mlflow: If True, log metrics and artifacts to MLflow.

    Returns:
        Full metrics dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path} ...")
    model = load_model(checkpoint_path, device=device, num_classes=num_classes)

    print(f"Loading test data from {data_dir} ...")
    transform = get_val_transforms(img_size)
    dataset = FundusDataset(root=data_dir, split="test", transform=transform, num_classes=num_classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    print(f"Test set: {len(dataset)} images")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    print("Running inference ...")
    start_time = time.time()
    results = run_inference(model, dataloader, device=device, use_tta=use_tta)
    elapsed = time.time() - start_time
    print(f"Inference complete in {elapsed:.1f}s ({len(dataset) / elapsed:.1f} img/s)")

    print("Computing metrics ...")
    metrics = compute_metrics(
        results["labels"], results["preds"], results["probs"], num_classes=num_classes
    )
    metrics["inference_time_sec"] = round(elapsed, 2)
    metrics["images_per_second"] = round(len(dataset) / elapsed, 2)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["use_tta"] = use_tta

    # Save metrics JSON
    metrics_path = output_path / "metrics.json"
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Save classification report
    report_path = output_path / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])

    # Generate and save confusion matrix plot
    cm = np.array(metrics["confusion_matrix"])
    cm_path = output_path / "confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        class_names=DR_GRADE_NAMES[:num_classes],
        save_path=str(cm_path),
        title=f"DR Grading — QWK={metrics['qwk']:.4f}",
    )
    print(f"Confusion matrix saved to {cm_path}")

    # Save per-class metrics detail
    detail = per_class_metrics(results["labels"], results["preds"], num_classes=num_classes)
    detail_path = output_path / "per_class_metrics.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    # Save raw predictions for further analysis
    np.savez_compressed(
        output_path / "predictions.npz",
        logits=results["logits"],
        probs=results["probs"],
        preds=results["preds"],
        labels=results["labels"],
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  QWK:                    {metrics['qwk']:.4f}")
    print(f"  Accuracy:               {metrics['accuracy']:.4f}")
    print(f"  AUC (macro):            {metrics['auc_macro']:.4f}")
    print(f"  Referable Sensitivity:  {metrics['referable_sensitivity']:.4f}")
    print(f"  Referable Specificity:  {metrics['referable_specificity']:.4f}")
    print(f"  ECE:                    {metrics['ece']:.4f}")
    print("=" * 60)
    print(metrics["classification_report"])

    # Optionally log to MLflow
    if log_to_mlflow:
        _log_to_mlflow(metrics, output_path)

    return metrics


def _log_to_mlflow(metrics: dict, artifact_dir: Path) -> None:
    """Log evaluation metrics and artifacts to MLflow."""
    try:
        import mlflow

        with mlflow.start_run(run_name="evaluation"):
            # Log scalar metrics
            for key in [
                "qwk",
                "accuracy",
                "auc_macro",
                "referable_sensitivity",
                "referable_specificity",
                "ece",
                "inference_time_sec",
                "images_per_second",
            ]:
                if key in metrics and not isinstance(metrics[key], (dict, list, str)):
                    mlflow.log_metric(key, metrics[key])

            # Log per-class metrics
            for name, val in metrics.get("sensitivity", {}).items():
                mlflow.log_metric(f"sensitivity_{name}", val)
            for name, val in metrics.get("specificity", {}).items():
                mlflow.log_metric(f"specificity_{name}", val)

            # Log artifacts
            for fpath in artifact_dir.iterdir():
                if fpath.is_file():
                    mlflow.log_artifact(str(fpath))

            print("Logged results to MLflow.")
    except ImportError:
        print("MLflow not installed; skipping MLflow logging.")
    except Exception as e:
        print(f"MLflow logging failed: {e}")
