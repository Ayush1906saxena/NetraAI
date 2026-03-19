"""
Root conftest.py for the Netra AI test suite.

Provides shared fixtures used across ML and server tests:
- Temporary directories
- Sample image generation
- Model fixtures (lightweight mocks)
- Device detection
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Device fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device() -> str:
    """Return the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def cpu_device() -> str:
    """Always return CPU device (for deterministic tests)."""
    return "cpu"


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path


@pytest.fixture
def output_dir(tmp_path):
    """Provide a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Sample image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fundus_image() -> np.ndarray:
    """
    Generate a synthetic fundus-like image for testing.

    Creates a 512x512 RGB image with a bright circular region on a dark
    background, roughly mimicking a fundus photograph.
    """
    size = 512
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Draw a bright circle (simulating the fundus disc area)
    center = size // 2
    radius = size // 3
    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Add some color variation
    img[mask, 0] = np.random.randint(120, 200)  # Red channel
    img[mask, 1] = np.random.randint(60, 120)   # Green channel
    img[mask, 2] = np.random.randint(30, 80)    # Blue channel

    # Add some noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def sample_fundus_tensor() -> torch.Tensor:
    """
    Generate a normalized fundus-like tensor for model input.

    Returns (1, 3, 224, 224) tensor with ImageNet normalization.
    """
    # Random tensor simulating a preprocessed fundus image
    tensor = torch.randn(1, 3, 224, 224)
    return tensor


@pytest.fixture
def sample_batch_tensor() -> torch.Tensor:
    """Generate a batch of 4 normalized tensors."""
    return torch.randn(4, 3, 224, 224)


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dataset_dir(tmp_path) -> Path:
    """
    Create a mock dataset directory with the expected structure:
        tmp_path/{train,val,test}/{0,1,2,3,4}/image_NNN.jpg

    Creates 5 images per class per split (75 total).
    """
    from PIL import Image

    splits = ["train", "val", "test"]
    num_classes = 5
    images_per_class = 5

    for split in splits:
        for grade in range(num_classes):
            grade_dir = tmp_path / split / str(grade)
            grade_dir.mkdir(parents=True)

            for i in range(images_per_class):
                # Create a small random image
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(grade_dir / f"img_{split}_{grade}_{i:03d}.jpg")

    return tmp_path


# ---------------------------------------------------------------------------
# Model fixtures (lightweight for fast tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_dr_model():
    """
    Create a lightweight mock DR grading model for testing.

    Uses a simple linear model instead of the full RETFound backbone,
    so tests run in seconds rather than minutes.
    """

    class MockDRModel(torch.nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            )
            self.head = torch.nn.Linear(16, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.head(x)

    model = MockDRModel()
    model.eval()
    return model


@pytest.fixture
def mock_iqa_model():
    """Create a lightweight mock IQA model for testing."""
    from ml.models.iqa_model import FundusIQA

    model = FundusIQA(pretrained_backbone=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def image_bytes(sample_fundus_image) -> bytes:
    """Encode a sample fundus image as JPEG bytes."""
    import cv2

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(sample_fundus_image, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()
