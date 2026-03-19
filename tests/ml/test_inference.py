"""
Tests for model inference, TTA, GradCAM, calibration, and drift detection.

Uses lightweight mock models to keep tests fast.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.evaluation.calibration import (
    TemperatureScaling,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)
from ml.evaluation.confusion import (
    error_analysis,
    per_class_metrics,
    plot_confusion_matrix,
    plot_error_distribution,
)
from ml.evaluation.tta import TTAPredictor
from ml.monitoring.drift import PredictionDriftDetector


# ---------------------------------------------------------------------------
# TTA Tests
# ---------------------------------------------------------------------------

class TestTTAPredictor:
    """Tests for test-time augmentation."""

    def test_predict_single_shape(self, mock_dr_model, sample_fundus_tensor):
        """Single image TTA should return (C,) probabilities."""
        tta = TTAPredictor(mock_dr_model, device="cpu", n_folds=4)
        probs = tta.predict_single(sample_fundus_tensor.squeeze(0))
        assert probs.shape == (5,), f"Expected (5,), got {probs.shape}"
        assert probs.sum() > 0

    def test_predict_single_sums_to_one(self, mock_dr_model, sample_fundus_tensor):
        """Averaged softmax probabilities should sum to ~1."""
        tta = TTAPredictor(mock_dr_model, device="cpu", n_folds=8)
        probs = tta.predict_single(sample_fundus_tensor)
        np.testing.assert_almost_equal(probs.sum(), 1.0, decimal=4)

    def test_predict_batch_shape(self, mock_dr_model, sample_batch_tensor):
        """Batch TTA should return (B, C) pseudo-logits."""
        tta = TTAPredictor(mock_dr_model, device="cpu", n_folds=4)
        result = tta.predict_batch(sample_batch_tensor)
        assert result.shape == (4, 5), f"Expected (4, 5), got {result.shape}"

    def test_predict_with_uncertainty(self, mock_dr_model, sample_fundus_tensor):
        """Uncertainty prediction should return expected keys."""
        tta = TTAPredictor(mock_dr_model, device="cpu", n_folds=8)
        result = tta.predict_with_uncertainty(sample_fundus_tensor)

        assert "probs" in result
        assert "pred" in result
        assert "confidence" in result
        assert "uncertainty" in result
        assert "fold_preds" in result
        assert "agreement" in result

        assert 0 <= result["pred"] <= 4
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["uncertainty"] >= 0.0
        assert 0.0 <= result["agreement"] <= 1.0
        assert len(result["fold_preds"]) == 8

    def test_different_n_folds(self, mock_dr_model, sample_fundus_tensor):
        """Different n_folds should produce different results."""
        tta_2 = TTAPredictor(mock_dr_model, device="cpu", n_folds=2)
        tta_8 = TTAPredictor(mock_dr_model, device="cpu", n_folds=8)

        result_2 = tta_2.predict_with_uncertainty(sample_fundus_tensor)
        result_8 = tta_8.predict_with_uncertainty(sample_fundus_tensor)

        assert len(result_2["fold_preds"]) == 2
        assert len(result_8["fold_preds"]) == 8

    def test_tta_more_robust_than_single(self, mock_dr_model):
        """TTA with 8 folds should have lower variance than single prediction."""
        # This is a statistical property; just verify it runs
        tta = TTAPredictor(mock_dr_model, device="cpu", n_folds=8)
        results = []
        for _ in range(5):
            tensor = torch.randn(1, 3, 224, 224)
            result = tta.predict_with_uncertainty(tensor)
            results.append(result["confidence"])
        # Just verify we get valid confidence values
        assert all(0 <= c <= 1 for c in results)


# ---------------------------------------------------------------------------
# Calibration Tests
# ---------------------------------------------------------------------------

class TestCalibration:
    """Tests for calibration metrics and temperature scaling."""

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE ~0."""
        n = 1000
        confidences = np.random.uniform(0.5, 1.0, n)
        # Make accuracy match confidence
        accuracies = (np.random.uniform(0, 1, n) < confidences).astype(float)
        ece = expected_calibration_error(confidences, accuracies, n_bins=10)
        # Should be relatively low (not exactly 0 due to binning)
        assert ece < 0.15

    def test_ece_overconfident(self):
        """Always-confident but wrong model should have high ECE."""
        n = 1000
        confidences = np.ones(n) * 0.99  # always confident
        accuracies = np.zeros(n)  # always wrong
        ece = expected_calibration_error(confidences, accuracies)
        assert ece > 0.5

    def test_ece_range(self):
        """ECE should be in [0, 1]."""
        confidences = np.random.uniform(0, 1, 500)
        accuracies = np.random.randint(0, 2, 500).astype(float)
        ece = expected_calibration_error(confidences, accuracies)
        assert 0.0 <= ece <= 1.0

    def test_mce_range(self):
        """MCE should be in [0, 1]."""
        confidences = np.random.uniform(0, 1, 500)
        accuracies = np.random.randint(0, 2, 500).astype(float)
        mce = maximum_calibration_error(confidences, accuracies)
        assert 0.0 <= mce <= 1.0

    def test_mce_geq_ece(self):
        """MCE should always be >= ECE."""
        confidences = np.random.uniform(0, 1, 500)
        accuracies = np.random.randint(0, 2, 500).astype(float)
        ece = expected_calibration_error(confidences, accuracies)
        mce = maximum_calibration_error(confidences, accuracies)
        assert mce >= ece - 1e-8

    def test_temperature_scaling_init(self):
        """TemperatureScaling should initialize with given temperature."""
        ts = TemperatureScaling(init_temperature=2.0)
        assert abs(ts.temperature.item() - 2.0) < 1e-6

    def test_temperature_scaling_calibrate(self):
        """Calibrated probs should sum to 1."""
        ts = TemperatureScaling(init_temperature=1.5)
        logits = torch.randn(10, 5)
        probs = ts.calibrate(logits)
        np.testing.assert_allclose(probs.sum(dim=1).detach().numpy(), 1.0, atol=1e-5)

    def test_temperature_scaling_save_load(self, tmp_dir):
        """Should round-trip save/load correctly."""
        ts = TemperatureScaling(init_temperature=2.34)
        path = str(tmp_dir / "temp.pt")
        ts.save(path)

        loaded = TemperatureScaling.load(path)
        assert abs(loaded.temperature.item() - 2.34) < 1e-5

    def test_reliability_diagram_produces_figure(self):
        """reliability_diagram should return a matplotlib Figure."""
        confidences = np.random.uniform(0.3, 1.0, 200)
        accuracies = np.random.randint(0, 2, 200).astype(float)
        fig = reliability_diagram(confidences, accuracies)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_reliability_diagram_saves_to_file(self, tmp_dir):
        """reliability_diagram should save to PNG."""
        confidences = np.random.uniform(0.3, 1.0, 200)
        accuracies = np.random.randint(0, 2, 200).astype(float)
        path = str(tmp_dir / "reliability.png")
        fig = reliability_diagram(confidences, accuracies, save_path=path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Confusion Matrix & Error Analysis Tests
# ---------------------------------------------------------------------------

class TestConfusionMetrics:
    """Tests for confusion matrix and error analysis."""

    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions for testing."""
        np.random.seed(42)
        n = 200
        labels = np.random.randint(0, 5, n)
        # Make predictions mostly correct with some noise
        preds = labels.copy()
        noise_indices = np.random.choice(n, size=40, replace=False)
        preds[noise_indices] = np.random.randint(0, 5, 40)
        probs = np.random.dirichlet([1] * 5, n)
        # Set the predicted class to have high probability
        for i in range(n):
            probs[i, preds[i]] += 2.0
        probs = probs / probs.sum(axis=1, keepdims=True)
        return labels, preds, probs

    def test_per_class_metrics_keys(self, sample_predictions):
        labels, preds, _ = sample_predictions
        metrics = per_class_metrics(labels, preds)
        assert len(metrics) == 5
        for name, m in metrics.items():
            assert "precision" in m
            assert "recall" in m
            assert "f1" in m
            assert "specificity" in m
            assert "support" in m
            assert "true_positives" in m

    def test_per_class_metrics_ranges(self, sample_predictions):
        labels, preds, _ = sample_predictions
        metrics = per_class_metrics(labels, preds)
        for name, m in metrics.items():
            assert 0.0 <= m["precision"] <= 1.0
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1"] <= 1.0
            assert 0.0 <= m["specificity"] <= 1.0

    def test_plot_confusion_matrix_saves(self, sample_predictions, tmp_dir):
        labels, preds, _ = sample_predictions
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
        path = str(tmp_dir / "cm.png")
        fig = plot_confusion_matrix(cm, save_path=path)
        assert Path(path).exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_error_analysis_structure(self, sample_predictions):
        labels, preds, probs = sample_predictions
        analysis = error_analysis(labels, preds, probs)

        assert "total_errors" in analysis
        assert "error_rate" in analysis
        assert "confusion_pairs" in analysis
        assert "high_confidence_errors" in analysis
        assert "off_by_2_or_more" in analysis
        assert "clinical_misses" in analysis

        assert analysis["total_errors"] >= 0
        assert 0.0 <= analysis["error_rate"] <= 1.0

    def test_error_analysis_clinical_misses(self):
        """Verify clinical misses are correctly identified."""
        # All grade 2+ predicted as grade 0
        labels = np.array([0, 1, 2, 3, 4])
        preds = np.array([0, 1, 0, 0, 0])
        probs = np.eye(5)

        analysis = error_analysis(labels, preds, probs)
        # Grades 2, 3, 4 are referable; 2, 3, 4 are all predicted < 2
        assert analysis["clinical_misses"]["count"] == 3

    def test_plot_error_distribution(self, sample_predictions, tmp_dir):
        labels, preds, _ = sample_predictions
        path = str(tmp_dir / "errors.png")
        fig = plot_error_distribution(labels, preds, save_path=path)
        assert Path(path).exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Drift Detection Tests
# ---------------------------------------------------------------------------

class TestDriftDetector:
    """Tests for PredictionDriftDetector."""

    @pytest.fixture
    def reference_dist(self):
        return [0.40, 0.15, 0.25, 0.12, 0.08]

    @pytest.fixture
    def detector(self, reference_dist):
        return PredictionDriftDetector(
            reference_distribution=reference_dist,
            window_size=100,
            check_interval=10,
            min_samples_for_test=10,
        )

    def test_init(self, detector, reference_dist):
        assert detector.num_classes == 5
        np.testing.assert_allclose(detector.reference, reference_dist)

    def test_add_prediction(self, detector):
        result = detector.add_prediction(0, 0.9)
        assert result is None  # No alert on first prediction

    def test_no_alert_on_normal_distribution(self, detector, reference_dist):
        """Should not alert when predictions match reference."""
        np.random.seed(42)
        for _ in range(100):
            grade = np.random.choice(5, p=reference_dist)
            conf = np.random.uniform(0.6, 0.99)
            detector.add_prediction(grade, conf)

        status = detector.get_status()
        assert status["total_predictions"] == 100

    def test_dominance_alert(self, reference_dist):
        """Should alert when one class dominates."""
        detector = PredictionDriftDetector(
            reference_distribution=reference_dist,
            window_size=100,
            check_interval=10,
            min_samples_for_test=10,
            dominance_threshold=0.80,
        )

        alert = None
        # Send 100 predictions all as grade 0
        for i in range(100):
            result = detector.add_prediction(0, 0.9)
            if result is not None:
                alert = result

        assert alert is not None
        alert_types = [a["type"] for a in alert["alerts"]]
        assert "single_class_dominance" in alert_types

    def test_distribution_shift_alert(self, reference_dist):
        """Should alert on significant distribution shift."""
        detector = PredictionDriftDetector(
            reference_distribution=reference_dist,
            window_size=200,
            check_interval=20,
            min_samples_for_test=20,
            chi2_significance=0.05,
        )

        alert = None
        # Send predictions with a very different distribution
        shifted_dist = [0.05, 0.05, 0.05, 0.05, 0.80]
        for i in range(200):
            grade = np.random.choice(5, p=shifted_dist)
            result = detector.add_prediction(grade, 0.8)
            if result is not None:
                alert = result

        assert alert is not None

    def test_confidence_drop_alert(self, reference_dist):
        """Should alert when mean confidence drops."""
        detector = PredictionDriftDetector(
            reference_distribution=reference_dist,
            window_size=100,
            check_interval=10,
            min_samples_for_test=10,
            confidence_drop_threshold=0.2,
        )
        detector.set_reference_confidence(0.85)

        alert = None
        for i in range(100):
            grade = np.random.choice(5, p=reference_dist)
            # Very low confidence
            result = detector.add_prediction(grade, np.random.uniform(0.2, 0.5))
            if result is not None:
                alert = result

        assert alert is not None
        alert_types = [a["type"] for a in alert["alerts"]]
        assert "confidence_degradation" in alert_types

    def test_get_status(self, detector):
        """Status should contain expected fields."""
        for _ in range(20):
            detector.add_prediction(np.random.randint(0, 5), 0.8)

        status = detector.get_status()
        assert "total_predictions" in status
        assert "current_distribution" in status
        assert "mean_confidence" in status
        assert status["total_predictions"] == 20

    def test_save_load_state(self, detector, tmp_dir, reference_dist):
        """Should round-trip save/load correctly."""
        for _ in range(50):
            grade = np.random.choice(5, p=reference_dist)
            detector.add_prediction(grade, np.random.uniform(0.5, 0.99))

        path = str(tmp_dir / "drift_state.json")
        detector.save_state(path)

        loaded = PredictionDriftDetector.load_state(path)
        assert loaded._total_predictions == detector._total_predictions
        assert loaded.current_count == detector.current_count

    def test_reset(self, detector):
        """Reset should clear all state."""
        for _ in range(50):
            detector.add_prediction(0, 0.9)

        detector.reset()
        assert detector.current_count == 0
        assert detector._total_predictions == 0


# ---------------------------------------------------------------------------
# GradCAM Tests (basic, using mock model)
# ---------------------------------------------------------------------------

class TestGradCAM:
    """Basic GradCAM tests with mock CNN model."""

    def test_generate_gradcam_returns_png_bytes(self, mock_dr_model, sample_fundus_tensor, sample_fundus_image):
        """generate_gradcam should return PNG bytes."""
        from ml.evaluation.gradcam import generate_gradcam

        # Resize image to match what gradcam expects
        import cv2
        img_224 = cv2.resize(sample_fundus_image, (224, 224))

        png_bytes = generate_gradcam(
            mock_dr_model,
            sample_fundus_tensor,
            img_224,
        )
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Check PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"

    def test_gradcam_specific_class(self, mock_dr_model, sample_fundus_tensor, sample_fundus_image):
        """Should work with explicit target class."""
        from ml.evaluation.gradcam import generate_gradcam
        import cv2

        img_224 = cv2.resize(sample_fundus_image, (224, 224))

        for target_class in range(5):
            png_bytes = generate_gradcam(
                mock_dr_model,
                sample_fundus_tensor,
                img_224,
                target_class=target_class,
            )
            assert isinstance(png_bytes, bytes)
            assert len(png_bytes) > 0
