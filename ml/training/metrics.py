"""
Evaluation metrics for DR grading, binary classification, and calibration.
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Quadratic Weighted Kappa between true and predicted ordinal labels.

    Args:
        y_true: (N,) array of integer ground-truth DR grades [0..4].
        y_pred: (N,) array of integer predicted DR grades [0..4].

    Returns:
        QWK score in [-1, 1]. Higher is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referable_threshold: int = 2,
) -> float:
    """
    Compute AUC-ROC for referable DR detection (grade >= threshold).

    Args:
        y_true: (N,) integer ground-truth grades.
        y_prob: (N, C) softmax probabilities or (N,) positive-class probabilities.
                If 2-D, probabilities for classes >= threshold are summed.
        referable_threshold: grades >= this are considered referable (default 2).

    Returns:
        AUC-ROC score.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    binary_true = (y_true >= referable_threshold).astype(int)

    if y_prob.ndim == 2:
        # Sum probabilities for referable classes
        positive_prob = y_prob[:, referable_threshold:].sum(axis=1)
    else:
        positive_prob = y_prob

    return roc_auc_score(binary_true, positive_prob)


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.90,
    referable_threshold: int = 2,
) -> dict:
    """
    Find the operating point on the ROC curve that achieves at least
    `target_sensitivity` and report the corresponding specificity.

    Args:
        y_true: (N,) integer ground-truth grades.
        y_prob: (N, C) softmax probabilities or (N,) positive-class probabilities.
        target_sensitivity: minimum sensitivity to achieve (default 0.90).
        referable_threshold: grades >= this are referable.

    Returns:
        Dict with keys: sensitivity, specificity, threshold, youden_j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    binary_true = (y_true >= referable_threshold).astype(int)

    if y_prob.ndim == 2:
        positive_prob = y_prob[:, referable_threshold:].sum(axis=1)
    else:
        positive_prob = y_prob

    fpr, tpr, thresholds = roc_curve(binary_true, positive_prob)
    specificities = 1.0 - fpr

    # Find indices where sensitivity >= target
    valid_mask = tpr >= target_sensitivity
    if not valid_mask.any():
        # Fall back to the point with highest sensitivity
        idx = np.argmax(tpr)
    else:
        # Among valid points, pick the one with highest specificity
        valid_indices = np.where(valid_mask)[0]
        idx = valid_indices[np.argmax(specificities[valid_indices])]

    sensitivity = float(tpr[idx])
    specificity = float(specificities[idx])
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.0
    youden_j = sensitivity + specificity - 1.0

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "youden_j": youden_j,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Measures how well predicted probabilities match empirical accuracy,
    which is critical for clinical decision support.

    Args:
        y_true: (N,) integer ground-truth labels (class indices).
        y_prob: (N, C) softmax probability matrix.
        n_bins: number of equal-width confidence bins.

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        bin_size = in_bin.sum()

        if bin_size == 0:
            continue

        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    return float(ece)
