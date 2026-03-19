"""
Confusion matrix plotting, error analysis, and per-class metrics
for DR grading evaluation.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


DR_GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot a confusion matrix with both counts and percentages.

    Args:
        cm: (C, C) confusion matrix.
        class_names: Class labels for axes.
        save_path: Where to save the figure. None to skip saving.
        title: Plot title.
        figsize: Figure size.
        cmap: Matplotlib colormap.
        normalize: If True, normalize rows to show percentages.

    Returns:
        matplotlib Figure.
    """
    if class_names is None:
        class_names = DR_GRADE_NAMES[: cm.shape[0]]

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(float) / row_sums
        fmt = ".2%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    # Build annotation labels: count + percentage
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    percentages = cm.astype(float) / row_sums_safe * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({percentages[i, j]:.1f}%)"

    sns.heatmap(
        cm_display if normalize else cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        ax=ax,
        vmax=vmax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicted Grade", fontsize=12)
    ax.set_ylabel("True Grade", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_normalized_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Normalized Confusion Matrix",
) -> plt.Figure:
    """Plot row-normalized confusion matrix (shows recall per class)."""
    return plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=save_path,
        title=title,
        normalize=True,
        cmap="YlOrRd",
    )


def per_class_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    num_classes: int = 5,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute detailed per-class metrics.

    Returns dict with per-class precision, recall, f1, support,
    and additional clinical metrics.
    """
    if class_names is None:
        class_names = DR_GRADE_NAMES[:num_classes]

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    metrics = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        metrics[class_names[c]] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "support": int(support[c]),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    return metrics


def error_analysis(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    image_paths: Optional[list[str]] = None,
    num_classes: int = 5,
    top_k: int = 20,
) -> dict:
    """
    Analyze the most confident errors and common confusion patterns.

    Returns dict with:
        - 'confusion_pairs': most common (true, pred) error pairs
        - 'high_confidence_errors': errors with highest model confidence
        - 'off_by_more_than_one': cases where pred is 2+ grades off
        - 'clinical_misses': false negatives for referable DR (grade >= 2)
    """
    errors_mask = labels != preds
    error_indices = np.where(errors_mask)[0]

    # Most common confusion pairs
    confusion_pairs = {}
    for idx in error_indices:
        pair = (int(labels[idx]), int(preds[idx]))
        key = f"true={DR_GRADE_NAMES[pair[0]]}, pred={DR_GRADE_NAMES[pair[1]]}"
        confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
    confusion_pairs = dict(
        sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    )

    # Highest-confidence errors
    max_probs = probs.max(axis=1)
    error_confidences = max_probs[errors_mask]
    error_true = labels[errors_mask]
    error_pred = preds[errors_mask]

    sorted_by_conf = np.argsort(-error_confidences)[:top_k]
    high_conf_errors = []
    for rank, idx in enumerate(sorted_by_conf):
        entry = {
            "rank": rank + 1,
            "true_label": DR_GRADE_NAMES[int(error_true[idx])],
            "predicted": DR_GRADE_NAMES[int(error_pred[idx])],
            "confidence": float(error_confidences[idx]),
            "original_index": int(error_indices[idx]),
        }
        if image_paths:
            entry["image_path"] = image_paths[error_indices[idx]]
        high_conf_errors.append(entry)

    # Off by more than one grade
    grade_diff = np.abs(labels.astype(int) - preds.astype(int))
    off_by_2_plus = int((grade_diff >= 2).sum())
    off_by_3_plus = int((grade_diff >= 3).sum())

    # Clinical misses: referable DR (grade >= 2) predicted as non-referable (grade < 2)
    referable = labels >= 2
    predicted_non_referable = preds < 2
    clinical_misses_mask = referable & predicted_non_referable
    clinical_miss_count = int(clinical_misses_mask.sum())

    clinical_miss_details = []
    for idx in np.where(clinical_misses_mask)[0]:
        entry = {
            "true_label": DR_GRADE_NAMES[int(labels[idx])],
            "predicted": DR_GRADE_NAMES[int(preds[idx])],
            "confidence": float(max_probs[idx]),
            "index": int(idx),
        }
        if image_paths:
            entry["image_path"] = image_paths[idx]
        clinical_miss_details.append(entry)

    return {
        "total_errors": int(errors_mask.sum()),
        "error_rate": float(errors_mask.mean()),
        "confusion_pairs": confusion_pairs,
        "high_confidence_errors": high_conf_errors,
        "off_by_2_or_more": off_by_2_plus,
        "off_by_3_or_more": off_by_3_plus,
        "clinical_misses": {
            "count": clinical_miss_count,
            "rate": float(clinical_miss_count / referable.sum()) if referable.sum() > 0 else 0.0,
            "details": clinical_miss_details[:top_k],
        },
    }


def plot_error_distribution(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the distribution of prediction errors by grade difference.

    Shows how many predictions are exact, off by 1, off by 2, etc.
    For ordinal DR grading, being off by 1 is much less serious than
    being off by 3-4 grades.
    """
    diffs = preds.astype(int) - labels.astype(int)
    unique_diffs = sorted(set(diffs))
    min_diff = min(unique_diffs)
    max_diff = max(unique_diffs)

    bins = list(range(min_diff, max_diff + 2))
    fig, ax = plt.subplots(figsize=(10, 6))

    counts, edges, patches = ax.hist(
        diffs, bins=bins, align="left", rwidth=0.8, color="#4c72b0", edgecolor="white"
    )

    # Color-code by severity
    for patch, edge in zip(patches, edges):
        diff = int(edge)
        if diff == 0:
            patch.set_facecolor("#2ca02c")  # green for correct
        elif abs(diff) == 1:
            patch.set_facecolor("#ff7f0e")  # orange for off-by-1
        else:
            patch.set_facecolor("#d62728")  # red for off-by-2+

    # Add count labels
    for count, edge in zip(counts, edges):
        if count > 0:
            ax.text(
                edge, count + max(counts) * 0.02,
                str(int(count)),
                ha="center", va="bottom", fontsize=10,
            )

    ax.set_xlabel("Prediction Error (Predicted - True)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Grading Errors", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
