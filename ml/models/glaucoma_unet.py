"""
Glaucoma optic-disc / optic-cup segmentation model.

Architecture: U-Net (via segmentation_models_pytorch) with EfficientNet-B3 encoder.
Loss: combined Dice + Focal loss.
Post-processing: compute vertical Cup-to-Disc Ratio (CDR) from predicted masks.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceFocalLoss(nn.Module):
    """
    Combined Dice loss + Focal loss for segmentation.

    Supports multi-class soft masks (B, C, H, W) with per-channel Dice.
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

    # ---- Dice component ----
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Soft Dice loss averaged over channels.

        Args:
            pred:   (B, C, H, W) probabilities after sigmoid/softmax
            target: (B, C, H, W) one-hot or soft targets
        """
        pred_flat = pred.flatten(2)       # (B, C, N)
        target_flat = target.flatten(2)   # (B, C, N)

        intersection = (pred_flat * target_flat).sum(dim=2)
        cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()

    # ---- Focal component ----
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Binary focal loss computed per-pixel, averaged.

        Args:
            pred:   (B, C, H, W) probabilities after sigmoid
            target: (B, C, H, W) binary targets
        """
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        alpha_factor = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        loss = alpha_factor * focal_factor * bce
        return loss.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits from model
            target: (B, C, H, W) binary targets (one-hot encoded)

        Returns:
            scalar combined loss
        """
        pred = torch.sigmoid(logits)
        dice = self._dice_loss(pred, target)
        focal = self._focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GlaucomaSegmentor(nn.Module):
    """
    Segments optic disc (channel 0) and optic cup (channel 1) from a
    fundus image and computes vertical CDR for glaucoma screening.

    Backbone: EfficientNet-B3 via segmentation_models_pytorch U-Net.
    """

    CDR_NORMAL_THRESHOLD = 0.65
    CDR_SUSPECT_THRESHOLD = 0.80

    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 2,  # disc, cup
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_channels=list(decoder_channels),
            decoder_attention_type="scse",
        )

        # Optional lightweight refinement conv on top of decoder output
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.Dropout2d(p=dropout),
        )

        self.loss_fn = DiceFocalLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) fundus images (recommended 512x512).

        Returns:
            logits: (B, 2, H, W) — channel 0 = disc, channel 1 = cup.
        """
        logits = self.unet(x)
        logits = logits + self.refine(logits)  # residual refinement
        return logits

    def predict_masks(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary masks (B, 2, H, W) as uint8."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).to(torch.uint8)
        return masks

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Wrapper around DiceFocalLoss."""
        return self.loss_fn(logits, targets)

    # ------------------------------------------------------------------
    # CDR computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cdr(
        disc_mask: np.ndarray,
        cup_mask: np.ndarray,
        orientation: str = "vertical",
    ) -> Optional[float]:
        """
        Compute Cup-to-Disc Ratio from binary masks.

        Args:
            disc_mask: (H, W) binary numpy array for optic disc.
            cup_mask:  (H, W) binary numpy array for optic cup.
            orientation: 'vertical' or 'horizontal'.

        Returns:
            CDR float in [0, 1], or None if disc not detected.
        """
        disc_rows, disc_cols = np.where(disc_mask > 0)
        cup_rows, cup_cols = np.where(cup_mask > 0)

        if len(disc_rows) == 0:
            return None

        if orientation == "vertical":
            disc_diameter = disc_rows.max() - disc_rows.min() + 1
            cup_diameter = (cup_rows.max() - cup_rows.min() + 1) if len(cup_rows) > 0 else 0
        else:
            disc_diameter = disc_cols.max() - disc_cols.min() + 1
            cup_diameter = (cup_cols.max() - cup_cols.min() + 1) if len(cup_cols) > 0 else 0

        if disc_diameter == 0:
            return None

        cdr = cup_diameter / disc_diameter
        return round(min(cdr, 1.0), 4)

    @staticmethod
    def compute_area_cdr(disc_mask: np.ndarray, cup_mask: np.ndarray) -> Optional[float]:
        """
        Compute area-based CDR: sqrt(cup_area / disc_area).

        Returns:
            Area CDR float, or None if disc not detected.
        """
        disc_area = np.sum(disc_mask > 0)
        cup_area = np.sum(cup_mask > 0)

        if disc_area == 0:
            return None

        return round(math.sqrt(cup_area / disc_area), 4)

    @torch.no_grad()
    def assess_glaucoma(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[dict]:
        """
        End-to-end glaucoma screening inference.

        Args:
            x: (B, 3, H, W) batch of fundus images.

        Returns:
            List of dicts per image:
                vertical_cdr:  float | None
                area_cdr:      float | None
                risk_level:    'normal' | 'suspect' | 'high_risk'
                disc_area_px:  int
                cup_area_px:   int
        """
        self.eval()
        masks = self.predict_masks(x, threshold=threshold)
        masks_np = masks.cpu().numpy()

        results = []
        for i in range(masks_np.shape[0]):
            disc = masks_np[i, 0]
            cup = masks_np[i, 1]

            vcdr = self.compute_cdr(disc, cup, orientation="vertical")
            acdr = self.compute_area_cdr(disc, cup)

            if vcdr is None:
                risk = "no_disc_detected"
            elif vcdr < self.CDR_NORMAL_THRESHOLD:
                risk = "normal"
            elif vcdr < self.CDR_SUSPECT_THRESHOLD:
                risk = "suspect"
            else:
                risk = "high_risk"

            results.append(
                {
                    "vertical_cdr": vcdr,
                    "area_cdr": acdr,
                    "risk_level": risk,
                    "disc_area_px": int(np.sum(disc > 0)),
                    "cup_area_px": int(np.sum(cup > 0)),
                }
            )

        return results

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **kwargs) -> "GlaucomaSegmentor":
        """Load from a saved state-dict checkpoint."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
