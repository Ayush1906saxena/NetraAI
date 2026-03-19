"""
Prediction drift detection for deployed DR grading models.

Monitors the distribution of model predictions over time and raises
alerts when the distribution shifts significantly from a reference
baseline. This catches:

1. Data drift: camera hardware changes, new patient demographics,
   image quality degradation.
2. Model degradation: performance decay over time.
3. Operational anomalies: single-class dominance (e.g., model always
   predicts "No DR") indicating a broken pipeline.
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class PredictionDriftDetector:
    """
    Sliding-window drift detector for DR grade predictions.

    Maintains a rolling window of recent predictions and periodically
    compares their distribution against a reference baseline using
    chi-squared tests and simple heuristic checks.

    Usage:
        detector = PredictionDriftDetector(
            reference_distribution=[0.40, 0.15, 0.25, 0.12, 0.08],
            window_size=500,
        )
        # For each prediction:
        alert = detector.add_prediction(predicted_grade=2, confidence=0.85)
        if alert:
            print(f"DRIFT ALERT: {alert}")
    """

    def __init__(
        self,
        reference_distribution: list[float],
        num_classes: int = 5,
        window_size: int = 500,
        chi2_significance: float = 0.01,
        dominance_threshold: float = 0.85,
        min_samples_for_test: int = 50,
        check_interval: int = 50,
        confidence_drop_threshold: float = 0.15,
    ):
        """
        Args:
            reference_distribution: Expected class distribution from validation
                                    set, e.g., [0.40, 0.15, 0.25, 0.12, 0.08].
                                    Must sum to ~1.0.
            num_classes: Number of DR grades (default 5).
            window_size: Number of recent predictions to keep in the sliding window.
            chi2_significance: P-value threshold for chi-squared test.
                               Below this triggers a drift alert.
            dominance_threshold: If any single class exceeds this fraction
                                 of predictions, trigger a dominance alert.
            min_samples_for_test: Minimum predictions before running tests.
            check_interval: Run drift checks every N predictions.
            confidence_drop_threshold: Alert if mean confidence drops by this
                                       much compared to reference.
        """
        self.num_classes = num_classes
        self.reference = np.array(reference_distribution, dtype=np.float64)
        assert len(self.reference) == num_classes, (
            f"Reference distribution has {len(self.reference)} classes, "
            f"expected {num_classes}"
        )
        assert abs(self.reference.sum() - 1.0) < 0.01, (
            f"Reference distribution sums to {self.reference.sum():.3f}, expected ~1.0"
        )

        self.window_size = window_size
        self.chi2_significance = chi2_significance
        self.dominance_threshold = dominance_threshold
        self.min_samples = min_samples_for_test
        self.check_interval = check_interval
        self.confidence_drop_threshold = confidence_drop_threshold

        # Sliding windows
        self._predictions: deque[int] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)

        # State tracking
        self._total_predictions = 0
        self._alerts: list[dict] = []
        self._reference_mean_confidence: Optional[float] = None

    @property
    def current_count(self) -> int:
        return len(self._predictions)

    def set_reference_confidence(self, mean_confidence: float) -> None:
        """Set the baseline mean confidence from validation set."""
        self._reference_mean_confidence = mean_confidence

    def add_prediction(
        self,
        predicted_grade: int,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Add a new prediction to the sliding window.

        Args:
            predicted_grade: Predicted DR grade (0-4).
            confidence: Model confidence (max softmax probability).
            timestamp: Unix timestamp. Defaults to current time.

        Returns:
            Alert dict if drift is detected, None otherwise.
        """
        if timestamp is None:
            timestamp = time.time()

        self._predictions.append(predicted_grade)
        self._confidences.append(confidence)
        self._timestamps.append(timestamp)
        self._total_predictions += 1

        # Run checks at the configured interval
        if (
            self._total_predictions % self.check_interval == 0
            and self.current_count >= self.min_samples
        ):
            return self._run_checks()

        return None

    def _run_checks(self) -> Optional[dict]:
        """Run all drift detection checks."""
        alerts = []

        # Check 1: Single-class dominance
        dominance_alert = self._check_dominance()
        if dominance_alert:
            alerts.append(dominance_alert)

        # Check 2: Chi-squared distribution shift
        chi2_alert = self._check_chi2_shift()
        if chi2_alert:
            alerts.append(chi2_alert)

        # Check 3: Confidence degradation
        conf_alert = self._check_confidence_drop()
        if conf_alert:
            alerts.append(conf_alert)

        if alerts:
            combined = {
                "timestamp": datetime.utcnow().isoformat(),
                "window_size": self.current_count,
                "total_predictions": self._total_predictions,
                "alerts": alerts,
                "current_distribution": self._get_current_distribution().tolist(),
                "reference_distribution": self.reference.tolist(),
            }
            self._alerts.append(combined)
            logger.warning(f"Drift alert: {json.dumps(combined, indent=2)}")
            return combined

        return None

    def _check_dominance(self) -> Optional[dict]:
        """
        Check if a single class dominates the prediction window.

        This catches degenerate model behavior where the model always
        predicts the same class (e.g., always "No DR").
        """
        dist = self._get_current_distribution()
        max_fraction = dist.max()
        dominant_class = int(dist.argmax())

        if max_fraction >= self.dominance_threshold:
            return {
                "type": "single_class_dominance",
                "severity": "critical",
                "message": (
                    f"Class {dominant_class} dominates with "
                    f"{max_fraction:.1%} of predictions "
                    f"(threshold: {self.dominance_threshold:.1%})"
                ),
                "dominant_class": dominant_class,
                "fraction": float(max_fraction),
            }
        return None

    def _check_chi2_shift(self) -> Optional[dict]:
        """
        Chi-squared goodness-of-fit test comparing current distribution
        to the reference distribution.

        A significant result indicates the model's prediction distribution
        has shifted from what was observed during validation.
        """
        observed_counts = self._get_current_counts()
        n = observed_counts.sum()
        expected_counts = self.reference * n

        # Mask out classes with zero expected count (chi2 requirement)
        mask = expected_counts > 0
        if mask.sum() < 2:
            return None

        chi2_stat, p_value = stats.chisquare(observed_counts[mask], expected_counts[mask])

        if p_value < self.chi2_significance:
            return {
                "type": "distribution_shift",
                "severity": "warning",
                "message": (
                    f"Chi-squared test detects distribution shift "
                    f"(chi2={chi2_stat:.2f}, p={p_value:.6f}, "
                    f"threshold={self.chi2_significance})"
                ),
                "chi2_statistic": float(chi2_stat),
                "p_value": float(p_value),
            }
        return None

    def _check_confidence_drop(self) -> Optional[dict]:
        """
        Check if mean model confidence has dropped significantly.

        A sudden drop in confidence often indicates the model is seeing
        out-of-distribution data.
        """
        if self._reference_mean_confidence is None:
            return None

        current_mean = np.mean(list(self._confidences))
        drop = self._reference_mean_confidence - current_mean

        if drop >= self.confidence_drop_threshold:
            return {
                "type": "confidence_degradation",
                "severity": "warning",
                "message": (
                    f"Mean confidence dropped by {drop:.3f} "
                    f"(reference={self._reference_mean_confidence:.3f}, "
                    f"current={current_mean:.3f})"
                ),
                "reference_confidence": float(self._reference_mean_confidence),
                "current_confidence": float(current_mean),
                "drop": float(drop),
            }
        return None

    def _get_current_counts(self) -> np.ndarray:
        """Get per-class prediction counts from the sliding window."""
        counts = np.zeros(self.num_classes, dtype=np.float64)
        for pred in self._predictions:
            if 0 <= pred < self.num_classes:
                counts[pred] += 1
        return counts

    def _get_current_distribution(self) -> np.ndarray:
        """Get normalized prediction distribution from the sliding window."""
        counts = self._get_current_counts()
        total = counts.sum()
        if total == 0:
            return np.zeros(self.num_classes)
        return counts / total

    def get_status(self) -> dict:
        """Get current detector status for monitoring dashboards."""
        dist = self._get_current_distribution()
        confidences = list(self._confidences)

        return {
            "total_predictions": self._total_predictions,
            "window_size": self.current_count,
            "current_distribution": dist.tolist(),
            "reference_distribution": self.reference.tolist(),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "std_confidence": float(np.std(confidences)) if confidences else 0.0,
            "num_alerts": len(self._alerts),
            "last_alert": self._alerts[-1] if self._alerts else None,
        }

    def get_alerts(self, since: Optional[datetime] = None) -> list[dict]:
        """Get all alerts, optionally filtered by time."""
        if since is None:
            return list(self._alerts)
        since_str = since.isoformat()
        return [a for a in self._alerts if a["timestamp"] >= since_str]

    def reset(self) -> None:
        """Clear the sliding window and alert history."""
        self._predictions.clear()
        self._confidences.clear()
        self._timestamps.clear()
        self._total_predictions = 0
        self._alerts.clear()

    def save_state(self, path: str) -> None:
        """Persist detector state to JSON."""
        state = {
            "reference": self.reference.tolist(),
            "window_size": self.window_size,
            "chi2_significance": self.chi2_significance,
            "dominance_threshold": self.dominance_threshold,
            "total_predictions": self._total_predictions,
            "predictions": list(self._predictions),
            "confidences": list(self._confidences),
            "timestamps": list(self._timestamps),
            "alerts": self._alerts,
            "reference_mean_confidence": self._reference_mean_confidence,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "PredictionDriftDetector":
        """Restore detector from saved JSON state."""
        with open(path) as f:
            state = json.load(f)

        detector = cls(
            reference_distribution=state["reference"],
            window_size=state["window_size"],
            chi2_significance=state["chi2_significance"],
            dominance_threshold=state["dominance_threshold"],
        )
        detector._total_predictions = state["total_predictions"]
        detector._reference_mean_confidence = state.get("reference_mean_confidence")
        detector._alerts = state.get("alerts", [])

        for pred, conf, ts in zip(
            state["predictions"], state["confidences"], state["timestamps"]
        ):
            detector._predictions.append(pred)
            detector._confidences.append(conf)
            detector._timestamps.append(ts)

        return detector
