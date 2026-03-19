"""
DR Grading Ensemble with Test-Time Augmentation (TTA).

Combines MAE-based and DINOv2-based RETFound models via weighted averaging
after 8-fold geometric TTA. Includes clinical referral logic.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

from .retfound_wrapper import RETFoundDRGrader


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def _generate_tta_batch(x: torch.Tensor) -> list[torch.Tensor]:
    """
    Generate 8-fold TTA variants of an input batch:
        0: original
        1: horizontal flip
        2: vertical flip
        3: horizontal + vertical flip
        4: 90-degree rotation
        5: 90-degree rotation + horizontal flip
        6: 270-degree rotation
        7: 270-degree rotation + horizontal flip

    Args:
        x: (B, C, H, W) input tensor.

    Returns:
        List of 8 tensors, each (B, C, H, W).
    """
    x_hflip = TF.hflip(x)
    x_vflip = TF.vflip(x)
    x_hvflip = TF.vflip(x_hflip)
    x_rot90 = torch.rot90(x, k=1, dims=[2, 3])
    x_rot90_hflip = TF.hflip(x_rot90)
    x_rot270 = torch.rot90(x, k=3, dims=[2, 3])
    x_rot270_hflip = TF.hflip(x_rot270)

    return [x, x_hflip, x_vflip, x_hvflip, x_rot90, x_rot90_hflip, x_rot270, x_rot270_hflip]


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class DRGradingEnsemble(nn.Module):
    """
    Weighted ensemble of MAE and DINOv2 RETFound models with 8-fold TTA.

    DR grades (ICDR scale):
        0: No DR
        1: Mild NPDR
        2: Moderate NPDR
        3: Severe NPDR
        4: Proliferative DR

    Referral logic:
        - Grade >= 2           -> refer
        - Grade 1 + low conf   -> refer
        - Any microaneurysm uncertainty -> follow-up
    """

    GRADE_NAMES = [
        "No DR",
        "Mild NPDR",
        "Moderate NPDR",
        "Severe NPDR",
        "Proliferative DR",
    ]

    REFERRAL_THRESHOLD_GRADE = 2
    LOW_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        mae_checkpoint: Optional[str] = None,
        dinov2_checkpoint: Optional[str] = None,
        num_classes: int = 5,
        mae_weight: float = 0.55,
        dinov2_weight: float = 0.45,
        use_tta: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mae_weight = mae_weight
        self.dinov2_weight = dinov2_weight
        self.use_tta = use_tta

        # --- MAE branch ---
        self.mae_model = RETFoundDRGrader(
            num_classes=num_classes,
            model_variant="mae",
            use_lora=False,
        )
        if mae_checkpoint:
            state = torch.load(mae_checkpoint, map_location=device, weights_only=True)
            self.mae_model.backbone.load_state_dict(state, strict=False)

        # --- DINOv2 branch ---
        self.dinov2_model = RETFoundDRGrader(
            num_classes=num_classes,
            model_variant="dinov2",
            use_lora=False,
        )
        if dinov2_checkpoint:
            state = torch.load(dinov2_checkpoint, map_location=device, weights_only=True)
            self.dinov2_model.backbone.load_state_dict(state, strict=False)

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _forward_single_model(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run a single model with optional 8-fold TTA.

        Returns:
            (B, num_classes) averaged softmax probabilities.
        """
        if not self.use_tta:
            logits = model(x)
            return F.softmax(logits / self.temperature, dim=-1)

        tta_variants = _generate_tta_batch(x)
        probs_list = []
        for variant in tta_variants:
            logits = model(variant)
            probs = F.softmax(logits / self.temperature, dim=-1)
            probs_list.append(probs)

        # Average across TTA variants
        stacked = torch.stack(probs_list, dim=0)  # (8, B, C)
        return stacked.mean(dim=0)  # (B, C)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) fundus images.

        Returns:
            dict:
                probs:        (B, num_classes) fused probabilities
                mae_probs:    (B, num_classes)
                dinov2_probs: (B, num_classes)
                predicted:    (B,) predicted grade indices
                confidence:   (B,) max probability per sample
        """
        mae_probs = self._forward_single_model(self.mae_model, x)
        dinov2_probs = self._forward_single_model(self.dinov2_model, x)

        # Weighted fusion
        fused = self.mae_weight * mae_probs + self.dinov2_weight * dinov2_probs

        confidence, predicted = fused.max(dim=-1)

        return {
            "probs": fused,
            "mae_probs": mae_probs,
            "dinov2_probs": dinov2_probs,
            "predicted": predicted,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Referral logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def grade_and_refer(self, x: torch.Tensor) -> list[dict]:
        """
        Full clinical grading pipeline with referral decisions.

        Args:
            x: (B, 3, 224, 224)

        Returns:
            List[dict] per image:
                grade:        int (0-4)
                grade_name:   str
                confidence:   float
                probs:        list[float] per-class probabilities
                refer:        bool
                urgency:      'routine' | 'soon' | 'urgent'
                reason:       str explanation
                model_agreement: float (cosine similarity of MAE vs DINOv2 probs)
        """
        self.eval()
        outputs = self.forward(x)

        results = []
        batch_size = x.size(0)

        for i in range(batch_size):
            grade = outputs["predicted"][i].item()
            conf = outputs["confidence"][i].item()
            probs = outputs["probs"][i].cpu().tolist()
            mae_p = outputs["mae_probs"][i]
            dino_p = outputs["dinov2_probs"][i]

            # Model agreement via cosine similarity
            agreement = F.cosine_similarity(mae_p.unsqueeze(0), dino_p.unsqueeze(0)).item()

            # Referral decision
            refer, urgency, reason = self._referral_decision(grade, conf, agreement, probs)

            results.append(
                {
                    "grade": grade,
                    "grade_name": self.GRADE_NAMES[grade],
                    "confidence": round(conf, 4),
                    "probs": [round(p, 4) for p in probs],
                    "refer": refer,
                    "urgency": urgency,
                    "reason": reason,
                    "model_agreement": round(agreement, 4),
                }
            )

        return results

    def _referral_decision(
        self,
        grade: int,
        confidence: float,
        agreement: float,
        probs: list[float],
    ) -> tuple[bool, str, str]:
        """
        Determine referral based on grade, confidence, and model agreement.

        Returns:
            (refer: bool, urgency: str, reason: str)
        """
        # Proliferative DR -> urgent referral
        if grade == 4:
            return True, "urgent", "Proliferative DR detected - immediate ophthalmology referral required."

        # Severe NPDR -> urgent
        if grade == 3:
            return True, "urgent", "Severe NPDR detected - ophthalmology referral within 1 week."

        # Moderate NPDR -> soon
        if grade == 2:
            return True, "soon", "Moderate NPDR detected - ophthalmology referral within 1 month."

        # Mild NPDR with low confidence or disagreement -> refer cautiously
        if grade == 1:
            if confidence < self.LOW_CONFIDENCE_THRESHOLD:
                return True, "soon", (
                    f"Mild NPDR detected with low confidence ({confidence:.2f}). "
                    "Recommend confirmatory examination."
                )
            if agreement < 0.90:
                return True, "soon", (
                    f"Mild NPDR detected but models disagree (agreement={agreement:.2f}). "
                    "Recommend follow-up."
                )
            # Confident mild NPDR -> routine follow-up, no urgent referral
            return False, "routine", "Mild NPDR detected with high confidence. Annual follow-up recommended."

        # Grade 0: No DR
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            return False, "routine", (
                f"No DR detected but confidence is low ({confidence:.2f}). "
                "Consider re-screening in 6 months."
            )

        # Check if adjacent severe grades have non-trivial probability
        severe_prob = sum(probs[2:])
        if severe_prob > 0.15:
            return False, "routine", (
                f"No DR detected but {severe_prob:.1%} probability of moderate+ DR. "
                "Annual screening recommended."
            )

        return False, "routine", "No DR detected. Annual screening recommended."

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def set_weights(self, mae_weight: float, dinov2_weight: float):
        """Update ensemble weights (must sum to 1)."""
        assert abs(mae_weight + dinov2_weight - 1.0) < 1e-6, "Weights must sum to 1."
        self.mae_weight = mae_weight
        self.dinov2_weight = dinov2_weight

    def enable_tta(self):
        self.use_tta = True

    def disable_tta(self):
        self.use_tta = False

    @classmethod
    def from_checkpoints(
        cls,
        mae_path: str,
        dinov2_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "DRGradingEnsemble":
        """Convenience constructor from two checkpoint files."""
        model = cls(
            mae_checkpoint=mae_path,
            dinov2_checkpoint=dinov2_path,
            device=device,
            **kwargs,
        )
        model.eval()
        return model.to(device)
