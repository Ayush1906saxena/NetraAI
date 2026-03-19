"""
Test-Time Augmentation (TTA) module for DR grading.

Performs 8-fold TTA by applying geometric and photometric transforms
to each input image, running inference on all variants, and averaging
the softmax predictions.

8-fold TTA set:
    0: original
    1: horizontal flip
    2: vertical flip
    3: horizontal + vertical flip
    4: rotate 90
    5: rotate 180
    6: rotate 270
    7: gamma correction (gamma=0.8)
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAPredictor:
    """
    Test-Time Augmentation predictor.

    Applies N augmentation folds to each image, runs the model on each,
    and averages the softmax outputs for a more robust prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        n_folds: int = 8,
    ):
        self.model = model
        self.device = device
        self.n_folds = min(n_folds, 8)
        self.model.eval()

    @staticmethod
    def _hflip(x: torch.Tensor) -> torch.Tensor:
        """Horizontal flip: flip along width (dim=-1)."""
        return x.flip(-1)

    @staticmethod
    def _vflip(x: torch.Tensor) -> torch.Tensor:
        """Vertical flip: flip along height (dim=-2)."""
        return x.flip(-2)

    @staticmethod
    def _hvflip(x: torch.Tensor) -> torch.Tensor:
        """Both horizontal and vertical flip."""
        return x.flip(-1).flip(-2)

    @staticmethod
    def _rot90(x: torch.Tensor) -> torch.Tensor:
        """Rotate 90 degrees counter-clockwise."""
        return torch.rot90(x, k=1, dims=[-2, -1])

    @staticmethod
    def _rot180(x: torch.Tensor) -> torch.Tensor:
        """Rotate 180 degrees."""
        return torch.rot90(x, k=2, dims=[-2, -1])

    @staticmethod
    def _rot270(x: torch.Tensor) -> torch.Tensor:
        """Rotate 270 degrees counter-clockwise."""
        return torch.rot90(x, k=3, dims=[-2, -1])

    @staticmethod
    def _gamma(x: torch.Tensor, gamma: float = 0.8) -> torch.Tensor:
        """
        Apply gamma correction.

        This works on normalized tensors: we undo ImageNet normalization,
        apply gamma, then re-normalize. For simplicity, we approximate
        by clamping to [0, 1] range, applying gamma, then rescaling.
        """
        # ImageNet normalization constants
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        # Denormalize
        denorm = x * std + mean
        denorm = denorm.clamp(0.0, 1.0)

        # Apply gamma correction
        corrected = denorm.pow(gamma)

        # Re-normalize
        result = (corrected - mean) / std
        return result

    def _get_augmentations(self):
        """Return list of (name, transform_fn) tuples."""
        augmentations = [
            ("original", lambda x: x),
            ("hflip", self._hflip),
            ("vflip", self._vflip),
            ("hvflip", self._hvflip),
            ("rot90", self._rot90),
            ("rot180", self._rot180),
            ("rot270", self._rot270),
            ("gamma_0.8", lambda x: self._gamma(x, gamma=0.8)),
        ]
        return augmentations[: self.n_folds]

    @torch.no_grad()
    def predict_single(self, image: torch.Tensor) -> np.ndarray:
        """
        Run TTA on a single image tensor.

        Args:
            image: (3, H, W) or (1, 3, H, W) tensor (already preprocessed/normalized).

        Returns:
            (C,) averaged softmax probabilities.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        augmentations = self._get_augmentations()
        all_probs = []

        for _name, aug_fn in augmentations:
            augmented = aug_fn(image)
            logits = self.model(augmented)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

        # Average softmax predictions across all augmentation folds
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return avg_probs.squeeze(0).numpy()

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Run TTA on a batch of images.

        Args:
            images: (B, 3, H, W) tensor batch.

        Returns:
            (B, C) averaged softmax probabilities as logits
                    (log of averaged probs for compatibility with evaluate.py).
        """
        images = images.to(self.device)
        augmentations = self._get_augmentations()
        all_probs = []

        for _name, aug_fn in augmentations:
            augmented = aug_fn(images)
            logits = self.model(augmented)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

        # Average softmax predictions
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)

        # Convert back to log-space (pseudo-logits) for consistency with evaluate.py
        pseudo_logits = torch.log(avg_probs + 1e-8)
        return pseudo_logits.numpy()

    @torch.no_grad()
    def predict_with_uncertainty(self, image: torch.Tensor) -> dict:
        """
        Run TTA and also return uncertainty estimates.

        The variance across TTA folds gives a model-free uncertainty
        estimate that can flag ambiguous cases for human review.

        Args:
            image: (3, H, W) or (1, 3, H, W) tensor.

        Returns:
            dict with:
                'probs': (C,) averaged softmax
                'pred': int predicted class
                'confidence': float max probability
                'uncertainty': float mean std across classes and folds
                'fold_preds': list of per-fold predictions
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        augmentations = self._get_augmentations()
        all_probs = []
        fold_preds = []

        for _name, aug_fn in augmentations:
            augmented = aug_fn(image)
            logits = self.model(augmented)
            probs = F.softmax(logits, dim=-1).cpu().squeeze(0)
            all_probs.append(probs)
            fold_preds.append(int(probs.argmax().item()))

        stacked = torch.stack(all_probs, dim=0)  # (n_folds, C)
        avg_probs = stacked.mean(dim=0)
        std_probs = stacked.std(dim=0)

        pred = int(avg_probs.argmax().item())
        confidence = float(avg_probs.max().item())
        uncertainty = float(std_probs.mean().item())

        return {
            "probs": avg_probs.numpy(),
            "pred": pred,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "fold_preds": fold_preds,
            "agreement": fold_preds.count(pred) / len(fold_preds),
        }
