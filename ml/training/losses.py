"""
Custom loss functions for DR grading, ordinal regression, and segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Distributes (smoothing / num_classes) probability to all classes and
    (1 - smoothing) to the ground-truth class. Reduces overconfidence
    which is important for calibrated clinical predictions.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw class logits.
            target: (B,) integer class labels.

        Returns:
            Scalar loss.
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)

        loss = -(smooth_target * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class QWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa loss.

    Uses soft predictions (softmax probabilities) to construct a differentiable
    confusion matrix, then computes 1 - QWK as the loss. This directly
    optimizes the evaluation metric used for DR grading competitions.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # Pre-compute the quadratic weight matrix W[i,j] = (i-j)^2 / (C-1)^2
        weights = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw class logits.
            target: (B,) integer class labels.

        Returns:
            Scalar loss = 1 - approx_qwk. Lower is better (QWK closer to 1).
        """
        num_classes = self.num_classes
        batch_size = logits.size(0)

        # Soft predictions via softmax
        probs = F.softmax(logits, dim=-1)  # (B, C)

        # One-hot encode targets
        target_onehot = F.one_hot(target, num_classes=num_classes).float()  # (B, C)

        # Soft confusion matrix: O[i,j] = sum_b target_b[i] * pred_b[j]
        O = target_onehot.t() @ probs  # (C, C)

        # Normalize to get observed proportions
        O = O / (O.sum() + 1e-8)

        # Marginal distributions
        hist_true = O.sum(dim=1)   # (C,)
        hist_pred = O.sum(dim=0)   # (C,)

        # Expected matrix under independence
        E = hist_true.unsqueeze(1) * hist_pred.unsqueeze(0)  # (C, C)

        # QWK = 1 - sum(W * O) / sum(W * E)
        numerator = (self.weights * O).sum()
        denominator = (self.weights * E).sum() + 1e-8

        qwk = 1.0 - numerator / denominator

        # Loss = 1 - QWK (we want to maximize QWK)
        return 1.0 - qwk


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss for segmentation tasks (e.g., glaucoma
    optic disc/cup segmentation).

    Supports multi-class binary segmentation masks (B, C, H, W).
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Dice loss averaged over channels."""
        pred_flat = pred.flatten(2)       # (B, C, N)
        target_flat = target.flatten(2)   # (B, C, N)

        intersection = (pred_flat * target_flat).sum(dim=2)
        cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary focal loss computed per-pixel."""
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        alpha_factor = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        loss = alpha_factor * focal_factor * bce
        return loss.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits from segmentation model.
            target: (B, C, H, W) binary targets (one-hot encoded).

        Returns:
            Scalar combined loss.
        """
        pred = torch.sigmoid(logits)
        dice = self._dice_loss(pred, target)
        focal = self._focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal
