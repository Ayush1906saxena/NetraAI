"""
Model calibration analysis for DR grading.

Provides reliability diagrams, Expected Calibration Error (ECE),
and temperature scaling for post-hoc calibration.

Well-calibrated models are critical in clinical settings: when the model
says it is 90% confident, it should be correct ~90% of the time.
"""

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual
    accuracy, weighted by the number of samples in each bin.

    Args:
        confidences: (N,) array of model confidence (max softmax probability).
        accuracies: (N,) binary array (1 if prediction is correct, 0 otherwise).
        n_bins: Number of bins for calibration.

    Returns:
        ECE as a float in [0, 1]. Lower is better.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def maximum_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the worst-case calibration gap across all bins.
    Important for safety-critical applications where even a single
    badly calibrated confidence range is dangerous.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            continue

        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].mean()
        mce = max(mce, abs(avg_accuracy - avg_confidence))

    return float(mce)


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """
    Plot a reliability diagram showing calibration quality.

    A perfectly calibrated model would have all bars on the diagonal.
    Bars below the diagonal indicate overconfidence; bars above indicate
    underconfidence.

    Args:
        confidences: (N,) model confidence values.
        accuracies: (N,) binary correctness values.
        n_bins: Number of calibration bins.
        save_path: Where to save the figure.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        bin_counts[i] = n_in_bin

        if n_in_bin > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()

    ece = expected_calibration_error(confidences, accuracies, n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Reliability diagram
    bar_width = 1.0 / n_bins * 0.8
    bars = ax1.bar(
        bin_centers,
        bin_accuracies,
        width=bar_width,
        color="#4c72b0",
        edgecolor="white",
        label="Model",
        alpha=0.8,
    )

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Gap visualization
    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap_color = "#d62728" if bin_confidences[i] > bin_accuracies[i] else "#2ca02c"
            ax1.plot(
                [bin_centers[i], bin_centers[i]],
                [bin_accuracies[i], bin_confidences[i]],
                color=gap_color,
                linewidth=2,
                alpha=0.6,
            )

    ax1.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax1.set_ylabel("Fraction of Positives (Accuracy)", fontsize=12)
    ax1.set_title(f"{title}\nECE={ece:.4f}, MCE={mce:.4f}", fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal")

    # Histogram of predictions
    ax2.bar(
        bin_centers,
        bin_counts,
        width=bar_width,
        color="#ff7f0e",
        edgecolor="white",
        alpha=0.8,
    )
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Prediction Distribution", fontsize=11)
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class TemperatureScaling(nn.Module):
    """
    Post-hoc temperature scaling for model calibration.

    Learns a single temperature parameter T that divides the logits
    before softmax: p = softmax(z / T). This is the simplest and most
    widely used calibration method.

    Usage:
        1. Train the model as usual.
        2. On the validation set, fit the temperature:
           ts = TemperatureScaling()
           ts.fit(model, val_loader, device)
        3. At inference time, use ts.calibrate(logits) to get
           calibrated probabilities.
    """

    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor([init_temperature], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        return F.softmax(self.forward(logits), dim=-1)

    @torch.no_grad()
    def calibrate_numpy(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate numpy logits and return numpy probabilities."""
        t = torch.tensor(logits, dtype=torch.float32)
        probs = self.calibrate(t)
        return probs.numpy()

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str = "cpu",
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit the temperature parameter on a validation set using NLL loss.

        Args:
            model: Trained model (will be set to eval mode).
            val_loader: Validation DataLoader.
            device: Device for computation.
            lr: Learning rate for LBFGS optimizer.
            max_iter: Maximum LBFGS iterations.

        Returns:
            Optimal temperature value.
        """
        model.eval()
        self.to(device)

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0).to(device)
        all_labels = torch.cat(all_labels, dim=0).to(device)

        # Optimize temperature using LBFGS
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        optimal_temp = self.temperature.item()
        print(f"Optimal temperature: {optimal_temp:.4f}")

        # Compute calibrated ECE
        with torch.no_grad():
            calibrated_probs = self.calibrate(all_logits)
            max_probs = calibrated_probs.max(dim=1).values.cpu().numpy()
            preds = calibrated_probs.argmax(dim=1).cpu().numpy()
            labels_np = all_labels.cpu().numpy()
            correct = (preds == labels_np).astype(float)

            ece_before = expected_calibration_error(
                F.softmax(all_logits, dim=1).max(dim=1).values.cpu().numpy(),
                (all_logits.argmax(dim=1).cpu().numpy() == labels_np).astype(float),
            )
            ece_after = expected_calibration_error(max_probs, correct)

        print(f"ECE before temperature scaling: {ece_before:.4f}")
        print(f"ECE after temperature scaling:  {ece_after:.4f}")

        return optimal_temp

    def save(self, path: str) -> None:
        """Save the temperature parameter."""
        torch.save({"temperature": self.temperature.item()}, path)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaling":
        """Load a saved temperature parameter."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        ts = cls(init_temperature=data["temperature"])
        return ts


def plot_calibration_comparison(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side reliability diagrams before and after temperature scaling.
    """
    # Before calibration
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs_before = exp / exp.sum(axis=-1, keepdims=True)
    conf_before = probs_before.max(axis=1)
    preds_before = probs_before.argmax(axis=1)
    correct_before = (preds_before == labels).astype(float)

    # After temperature scaling
    scaled = logits / temperature
    exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    probs_after = exp_s / exp_s.sum(axis=-1, keepdims=True)
    conf_after = probs_after.max(axis=1)
    preds_after = probs_after.argmax(axis=1)
    correct_after = (preds_after == labels).astype(float)

    ece_before = expected_calibration_error(conf_before, correct_before, n_bins)
    ece_after = expected_calibration_error(conf_after, correct_after, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, conf, acc, ece_val, label in [
        (ax1, conf_before, correct_before, ece_before, "Before Calibration"),
        (ax2, conf_after, correct_after, ece_after, f"After (T={temperature:.2f})"),
    ]:
        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bar_width = 1.0 / n_bins * 0.8

        bin_accs = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accs[i] = acc[mask].mean()

        ax.bar(
            bin_centers, bin_accs, width=bar_width,
            color="#4c72b0", edgecolor="white", alpha=0.8,
        )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{label}\nECE = {ece_val:.4f}", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
