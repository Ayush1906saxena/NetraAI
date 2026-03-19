"""
Fundus Image Quality Assessment (IQA) model.

Uses MobileNetV3-Small as a lightweight backbone with three prediction heads:
  - quality:   scalar [0, 1] overall quality score
  - gradeable: binary probability that the image is clinically gradeable
  - guidance:  multi-label logits for actionable feedback categories
               (e.g. "too dark", "out of focus", "occluded", "glare", "low contrast")
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FundusIQA(nn.Module):
    """Image-quality gatekeeper that runs before any diagnostic model."""

    GUIDANCE_LABELS = [
        "too_dark",
        "too_bright",
        "out_of_focus",
        "occluded",
        "glare",
        "low_contrast",
        "off_center",
        "artifacts",
    ]

    def __init__(
        self,
        num_guidance_classes: int = 8,
        pretrained_backbone: bool = True,
        dropout: float = 0.3,
        quality_threshold: float = 0.5,
        gradeable_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_guidance_classes = num_guidance_classes
        self.quality_threshold = quality_threshold
        self.gradeable_threshold = gradeable_threshold

        # --- backbone ---
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[0].in_features  # 576

        # --- shared neck ---
        self.neck = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
        )

        # --- head 1: quality score (regression, sigmoid-bounded) ---
        self.quality_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # --- head 2: gradeable (binary classification) ---
        self.gradeable_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, 1),
        )

        # --- head 3: guidance (multi-label classification) ---
        self.guidance_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_guidance_classes),
        )

        self._init_heads()

    def _init_heads(self):
        """Kaiming initialization for all head layers."""
        for module in [self.neck, self.quality_head, self.gradeable_head, self.guidance_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the shared 256-d feature vector."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) fundus image tensor, expected 224x224.

        Returns:
            dict with keys:
                quality:   (B, 1) float in [0, 1]
                gradeable: (B, 1) logit
                guidance:  (B, num_guidance_classes) logits
        """
        feats = self.extract_features(x)
        return {
            "quality": self.quality_head(feats),
            "gradeable": self.gradeable_head(feats),
            "guidance": self.guidance_head(feats),
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        quality_weight: float = 1.0,
        gradeable_weight: float = 1.0,
        guidance_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined multi-task loss.

        Args:
            predictions: output from forward()
            targets: dict with keys:
                quality:   (B, 1) float targets in [0, 1]
                gradeable: (B, 1) float binary targets
                guidance:  (B, num_guidance_classes) float binary targets

        Returns:
            dict with 'total', 'quality', 'gradeable', 'guidance' losses.
        """
        loss_quality = nn.functional.mse_loss(predictions["quality"], targets["quality"])
        loss_gradeable = nn.functional.binary_cross_entropy_with_logits(
            predictions["gradeable"], targets["gradeable"]
        )
        loss_guidance = nn.functional.binary_cross_entropy_with_logits(
            predictions["guidance"], targets["guidance"]
        )

        total = (
            quality_weight * loss_quality
            + gradeable_weight * loss_gradeable
            + guidance_weight * loss_guidance
        )

        return {
            "total": total,
            "quality": loss_quality,
            "gradeable": loss_gradeable,
            "guidance": loss_guidance,
        }

    @torch.no_grad()
    def assess(self, x: torch.Tensor) -> dict:
        """
        High-level inference helper.

        Returns a human-readable assessment dict:
            quality_score:  float
            is_gradeable:   bool
            issues:         list[str]  (triggered guidance labels)
            accept:         bool       (True if quality AND gradeable pass thresholds)
        """
        self.eval()
        preds = self.forward(x)

        quality_score = preds["quality"].squeeze(-1).cpu().float()
        gradeable_prob = torch.sigmoid(preds["gradeable"]).squeeze(-1).cpu().float()
        guidance_probs = torch.sigmoid(preds["guidance"]).cpu().float()

        results = []
        batch_size = x.size(0)
        for i in range(batch_size):
            issues = [
                self.GUIDANCE_LABELS[j]
                for j in range(self.num_guidance_classes)
                if guidance_probs[i, j].item() > 0.5
            ]
            q = quality_score[i].item()
            g = gradeable_prob[i].item()
            results.append(
                {
                    "quality_score": round(q, 4),
                    "is_gradeable": g >= self.gradeable_threshold,
                    "issues": issues,
                    "accept": q >= self.quality_threshold and g >= self.gradeable_threshold,
                }
            )

        return results[0] if batch_size == 1 else results

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **kwargs) -> "FundusIQA":
        """Load a trained FundusIQA model from a state-dict checkpoint."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model.to(device)
