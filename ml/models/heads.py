"""
Reusable classification, regression, and multi-task head modules.

These heads attach on top of a frozen or fine-tuned backbone and provide
flexible output layers for different downstream tasks in retinal imaging.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRClassificationHead(nn.Module):
    """
    Classification head for DR grading or similar ordinal/categorical tasks.

    Supports optional ordinal regression mode (CORN / cumulative logits)
    where the model predicts P(Y > k) for k = 0..K-2, which is better
    suited for ordinal labels like DR severity.

    Architecture: Linear -> GELU -> Dropout -> Linear [-> optional ordinal]
    """

    def __init__(
        self,
        in_features: int = 1024,
        num_classes: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        ordinal: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ordinal = ordinal

        out_dim = num_classes - 1 if ordinal else num_classes

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, in_features) from backbone.

        Returns:
            If ordinal=False: (B, num_classes) logits.
            If ordinal=True:  (B, num_classes-1) cumulative logits.
        """
        return self.head(features)

    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convenience method returning predicted class and probabilities.

        Returns:
            dict with 'logits', 'probs', 'predicted' keys.
        """
        logits = self.forward(features)

        if self.ordinal:
            # Convert cumulative logits to class probabilities
            cum_probs = torch.sigmoid(logits)  # P(Y > k) for k=0..K-2
            probs = self._cumulative_to_class_probs(cum_probs)
        else:
            probs = F.softmax(logits, dim=-1)

        confidence, predicted = probs.max(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "predicted": predicted,
            "confidence": confidence,
        }

    def _cumulative_to_class_probs(self, cum_probs: torch.Tensor) -> torch.Tensor:
        """
        Convert P(Y > k) cumulative probabilities to per-class probabilities.

        P(Y = 0) = 1 - P(Y > 0)
        P(Y = k) = P(Y > k-1) - P(Y > k)   for k = 1..K-2
        P(Y = K-1) = P(Y > K-2)
        """
        # cum_probs: (B, K-1) where K = num_classes
        p0 = 1.0 - cum_probs[:, 0:1]  # (B, 1)
        p_last = cum_probs[:, -1:]     # (B, 1)

        if self.num_classes > 2:
            p_mid = cum_probs[:, :-1] - cum_probs[:, 1:]  # (B, K-2)
            class_probs = torch.cat([p0, p_mid, p_last], dim=-1)  # (B, K)
        else:
            class_probs = torch.cat([p0, p_last], dim=-1)

        # Clamp to avoid numerical issues
        class_probs = class_probs.clamp(min=0.0)
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)
        return class_probs

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute loss appropriate for the head mode.

        Args:
            logits: output of forward()
            targets: (B,) integer class labels

        Returns:
            scalar loss
        """
        if self.ordinal:
            return self._ordinal_loss(logits, targets)
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

    def _ordinal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy on cumulative indicators."""
        # Create cumulative targets: for class k, indicators are 1 for j < k
        device = logits.device
        K = self.num_classes - 1
        # targets: (B,), indicators: (B, K) where indicator[i,j] = 1 if targets[i] > j
        indicators = (targets.unsqueeze(1) > torch.arange(K, device=device).unsqueeze(0)).float()
        return F.binary_cross_entropy_with_logits(logits, indicators)


class RegressionHead(nn.Module):
    """
    Regression head for continuous-valued predictions.

    Use cases:
        - Cup-to-Disc Ratio estimation
        - Foveal thickness prediction
        - Image quality score
        - Vessel tortuosity index

    Supports optional output clamping and activation.
    """

    def __init__(
        self,
        in_features: int = 1024,
        out_features: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        output_activation: Optional[str] = None,  # 'sigmoid', 'relu', None
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
    ):
        super().__init__()
        self.output_min = output_min
        self.output_max = output_max

        layers = [
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_features),
        ]

        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            layers.append(nn.ReLU())

        self.head = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, in_features)

        Returns:
            (B, out_features) predictions, optionally clamped.
        """
        out = self.head(features)
        if self.output_min is not None or self.output_max is not None:
            out = torch.clamp(out, min=self.output_min, max=self.output_max)
        return out

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "smooth_l1",
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, out_features)
            targets: (B, out_features)
            loss_type: 'mse', 'smooth_l1', or 'huber'

        Returns:
            scalar loss
        """
        if loss_type == "mse":
            return F.mse_loss(predictions, targets)
        elif loss_type == "smooth_l1":
            return F.smooth_l1_loss(predictions, targets)
        elif loss_type == "huber":
            return F.huber_loss(predictions, targets, delta=1.0)
        raise ValueError(f"Unknown loss type: {loss_type}")


class MultiTaskHead(nn.Module):
    """
    Multi-task head that bundles multiple sub-heads sharing the same backbone
    features. Handles mixed classification + regression tasks with per-task
    loss weighting.

    Example task config:
        tasks = {
            "dr_grade": {"type": "classification", "num_classes": 5, "weight": 1.0},
            "dme_risk":  {"type": "classification", "num_classes": 2, "weight": 0.5},
            "cdr":       {"type": "regression", "out_features": 1, "weight": 0.3,
                          "output_activation": "sigmoid"},
        }
    """

    def __init__(
        self,
        in_features: int = 1024,
        tasks: dict = None,
        shared_hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        if tasks is None:
            tasks = {
                "dr_grade": {"type": "classification", "num_classes": 5, "weight": 1.0},
            }

        self.task_configs = tasks

        # Shared feature projection
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # Per-task heads
        self.task_heads = nn.ModuleDict()
        self.task_weights = {}

        for name, cfg in tasks.items():
            task_type = cfg["type"]
            weight = cfg.get("weight", 1.0)
            self.task_weights[name] = weight

            if task_type == "classification":
                num_classes = cfg.get("num_classes", 2)
                ordinal = cfg.get("ordinal", False)
                self.task_heads[name] = DRClassificationHead(
                    in_features=shared_hidden_dim,
                    num_classes=num_classes,
                    hidden_dim=cfg.get("hidden_dim", 128),
                    dropout=dropout * 0.5,
                    ordinal=ordinal,
                )
            elif task_type == "regression":
                self.task_heads[name] = RegressionHead(
                    in_features=shared_hidden_dim,
                    out_features=cfg.get("out_features", 1),
                    hidden_dim=cfg.get("hidden_dim", 64),
                    dropout=dropout * 0.5,
                    output_activation=cfg.get("output_activation"),
                    output_min=cfg.get("output_min"),
                    output_max=cfg.get("output_max"),
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_features) backbone features.

        Returns:
            Dict mapping task name to output tensor.
        """
        shared = self.shared_proj(features)
        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(shared)
        return outputs

    def compute_loss(
        self,
        features: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            features: (B, in_features) backbone features.
            targets: dict mapping task name to target tensor.

        Returns:
            Dict with per-task losses and 'total'.
        """
        outputs = self.forward(features)
        losses = {}
        total = torch.tensor(0.0, device=features.device)

        for name, head in self.task_heads.items():
            if name not in targets:
                continue

            task_cfg = self.task_configs[name]
            task_type = task_cfg["type"]

            if task_type == "classification":
                loss = head.compute_loss(outputs[name], targets[name])
            else:
                loss_type = task_cfg.get("loss_type", "smooth_l1")
                loss = head.compute_loss(outputs[name], targets[name], loss_type=loss_type)

            weight = self.task_weights[name]
            losses[name] = loss
            total = total + weight * loss

        losses["total"] = total
        return losses

    def predict(self, features: torch.Tensor) -> dict[str, dict]:
        """
        Run all heads and return structured predictions.

        Returns:
            Dict[task_name, prediction_dict]
        """
        shared = self.shared_proj(features)
        results = {}
        for name, head in self.task_heads.items():
            if isinstance(head, DRClassificationHead):
                results[name] = head.predict(shared)
            else:
                results[name] = {"value": head(shared)}
        return results
