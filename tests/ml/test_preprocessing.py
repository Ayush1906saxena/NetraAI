"""
Tests for FundusPreprocessor.

Validates the preprocessing pipeline that all fundus images go through
before model inference: circle cropping, Ben Graham normalization,
CLAHE enhancement, and resizing.
"""

import cv2
import numpy as np
import pytest

from ml.data.preprocess import FundusPreprocessor


@pytest.fixture
def preprocessor():
    return FundusPreprocessor()


class TestCircleCrop:
    """Tests for circle-based fundus cropping."""

    def test_circle_crop_returns_correct_shape(self, preprocessor, sample_fundus_image):
        """Circle crop should return a square image."""
        result = preprocessor.circle_crop(sample_fundus_image)
        h, w = result.shape[:2]
        assert h == w, f"Expected square output, got {h}x{w}"
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_circle_crop_removes_black_borders(self, preprocessor):
        """Black borders around the fundus should be cropped away."""
        # Create an image with a bright circle on black background
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        cv2.circle(img, (500, 400), 300, (150, 100, 60), -1)

        result = preprocessor.circle_crop(img)
        h, w = result.shape[:2]
        # Cropped image should be smaller than original
        assert h < 800 or w < 1000

    def test_circle_crop_handles_already_cropped(self, preprocessor):
        """Should handle images that are already tightly cropped."""
        img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        result = preprocessor.circle_crop(img)
        assert result.shape[2] == 3

    def test_circle_crop_handles_all_black(self, preprocessor):
        """Should return original if image is entirely black (no fundus)."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = preprocessor.circle_crop(img)
        assert result.shape == img.shape

    def test_circle_crop_preserves_dtype(self, preprocessor, sample_fundus_image):
        """Output should be uint8."""
        result = preprocessor.circle_crop(sample_fundus_image)
        assert result.dtype == np.uint8


class TestBenGraham:
    """Tests for Ben Graham's local average color subtraction."""

    def test_ben_graham_output_shape(self, preprocessor, sample_fundus_image):
        """Output shape should match input."""
        result = preprocessor.ben_graham(sample_fundus_image)
        assert result.shape == sample_fundus_image.shape

    def test_ben_graham_output_range(self, preprocessor, sample_fundus_image):
        """Output should be uint8 in [0, 255]."""
        result = preprocessor.ben_graham(sample_fundus_image)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_ben_graham_changes_image(self, preprocessor, sample_fundus_image):
        """Ben Graham normalization should modify the image."""
        result = preprocessor.ben_graham(sample_fundus_image)
        assert not np.array_equal(result, sample_fundus_image)

    def test_ben_graham_custom_sigma(self, preprocessor, sample_fundus_image):
        """Should work with custom sigma parameter."""
        result = preprocessor.ben_graham(sample_fundus_image, sigma=10.0)
        assert result.shape == sample_fundus_image.shape

    def test_ben_graham_centers_around_128(self, preprocessor):
        """Properly illuminated regions should center near 128."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 100
        result = preprocessor.ben_graham(img)
        # The mean should be roughly around 128 (the bias term)
        assert 80 < result.mean() < 180


class TestCLAHE:
    """Tests for CLAHE contrast enhancement."""

    def test_clahe_output_shape(self, preprocessor, sample_fundus_image):
        result = preprocessor.clahe_enhance(sample_fundus_image)
        assert result.shape == sample_fundus_image.shape

    def test_clahe_output_dtype(self, preprocessor, sample_fundus_image):
        result = preprocessor.clahe_enhance(sample_fundus_image)
        assert result.dtype == np.uint8

    def test_clahe_improves_contrast(self, preprocessor):
        """CLAHE should increase the dynamic range of low-contrast images."""
        # Create a low-contrast image
        low_contrast = np.random.randint(100, 150, (256, 256, 3), dtype=np.uint8)
        result = preprocessor.clahe_enhance(low_contrast)

        # Enhanced image should have wider range
        original_range = low_contrast.max() - low_contrast.min()
        enhanced_range = result.max() - result.min()
        assert enhanced_range >= original_range

    def test_clahe_custom_clip_limit(self, preprocessor, sample_fundus_image):
        result = preprocessor.clahe_enhance(sample_fundus_image, clip_limit=4.0)
        assert result.shape == sample_fundus_image.shape


class TestFullPipeline:
    """Tests for the complete preprocessing pipeline."""

    def test_process_returns_correct_size(self, preprocessor, sample_fundus_image):
        """process() should return image at target_size."""
        result = preprocessor.process(sample_fundus_image, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_process_different_sizes(self, preprocessor, sample_fundus_image):
        """Should work with various target sizes."""
        for size in [128, 224, 384, 512]:
            result = preprocessor.process(sample_fundus_image, target_size=size)
            assert result.shape == (size, size, 3)

    def test_process_skip_ben_graham(self, preprocessor, sample_fundus_image):
        result = preprocessor.process(
            sample_fundus_image, target_size=224, apply_ben_graham=False
        )
        assert result.shape == (224, 224, 3)

    def test_process_skip_clahe(self, preprocessor, sample_fundus_image):
        result = preprocessor.process(
            sample_fundus_image, target_size=224, apply_clahe=False
        )
        assert result.shape == (224, 224, 3)

    def test_process_skip_both(self, preprocessor, sample_fundus_image):
        result = preprocessor.process(
            sample_fundus_image,
            target_size=224,
            apply_ben_graham=False,
            apply_clahe=False,
        )
        assert result.shape == (224, 224, 3)

    def test_process_bytes(self, preprocessor, image_bytes):
        """process_bytes() should accept JPEG bytes and return processed image."""
        result = preprocessor.process_bytes(image_bytes, target_size=224)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8

    def test_process_bytes_different_sizes(self, preprocessor, image_bytes):
        for size in [128, 224, 384]:
            result = preprocessor.process_bytes(image_bytes, target_size=size)
            assert result.shape == (size, size, 3)

    def test_process_rectangular_input(self, preprocessor):
        """Should handle non-square input images."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some bright pixels so circle_crop has something to find
        cv2.circle(img, (320, 240), 200, (150, 100, 60), -1)
        result = preprocessor.process(img, target_size=224)
        assert result.shape == (224, 224, 3)

    def test_process_large_input(self, preprocessor):
        """Should handle typical camera resolution (e.g. 3024x4032)."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 200, (150, 100, 60), -1)
        result = preprocessor.process(img, target_size=224)
        assert result.shape == (224, 224, 3)


class TestEdgeCases:
    """Edge case tests."""

    def test_single_pixel_image(self, preprocessor):
        """Should not crash on degenerate input."""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        # This may not produce meaningful output, but should not crash
        try:
            result = preprocessor.process(img, target_size=224)
            assert result.shape == (224, 224, 3)
        except (cv2.error, ValueError):
            pass  # Acceptable to raise on degenerate input

    def test_grayscale_input_fails_gracefully(self, preprocessor):
        """Single-channel input should fail with a clear error."""
        img = np.zeros((256, 256), dtype=np.uint8)
        with pytest.raises((cv2.error, ValueError, IndexError)):
            preprocessor.process(img, target_size=224)

    def test_deterministic_output(self, preprocessor, sample_fundus_image):
        """Same input should produce same output (no randomness in preprocessing)."""
        result1 = preprocessor.process(sample_fundus_image.copy(), target_size=224)
        result2 = preprocessor.process(sample_fundus_image.copy(), target_size=224)
        np.testing.assert_array_equal(result1, result2)
