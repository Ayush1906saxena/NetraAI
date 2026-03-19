"""
Synthetic Image Quality Assessment (IQA) label generator.

Generates quality labels for fundus images based on measurable image statistics.
Used to bootstrap an IQA model when manual quality annotations are unavailable.

Quality grades:
    0 - Good:     Sharp, well-exposed, centered fundus
    1 - Usable:   Minor issues (slight blur, mild exposure offset)
    2 - Reject:   Severe blur, extreme under/overexposure, heavy artifacts
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ImageQualityMetrics:
    """Raw quality metrics extracted from a fundus image."""
    blur_score: float        # Laplacian variance (higher = sharper)
    brightness: float        # Mean pixel intensity [0, 255]
    contrast: float          # Std of pixel intensity
    saturation: float        # Mean saturation in HSV space
    foreground_ratio: float  # Fraction of non-black pixels (fundus vs border)
    entropy: float           # Shannon entropy of grayscale histogram


def compute_quality_metrics(img: np.ndarray) -> ImageQualityMetrics:
    """
    Compute quality metrics from an RGB fundus image.

    Args:
        img: Input image in RGB format, shape (H, W, 3), dtype uint8.

    Returns:
        ImageQualityMetrics with all fields populated.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Blur: Laplacian variance. Low = blurry, high = sharp.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # Brightness: mean intensity of the grayscale image
    brightness = float(gray.mean())

    # Contrast: standard deviation of grayscale pixel values
    contrast = float(gray.std())

    # Saturation: mean of the S channel in HSV
    saturation = float(hsv[:, :, 1].mean())

    # Foreground ratio: fraction of pixels above a dark threshold
    # Fundus images have large black borders; good images have more fundus
    foreground_mask = gray > 15
    foreground_ratio = float(foreground_mask.sum()) / gray.size

    # Entropy: Shannon entropy of the grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return ImageQualityMetrics(
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        foreground_ratio=foreground_ratio,
        entropy=entropy,
    )


def classify_quality(
    metrics: ImageQualityMetrics,
    blur_thresholds: Tuple[float, float] = (50.0, 200.0),
    brightness_range: Tuple[float, float] = (40.0, 200.0),
    contrast_thresholds: Tuple[float, float] = (20.0, 45.0),
    foreground_min: Tuple[float, float] = (0.15, 0.30),
    entropy_thresholds: Tuple[float, float] = (5.0, 6.5),
) -> int:
    """
    Assign a quality grade based on image statistics.

    Uses a rule-based scoring system: each metric contributes penalty points.
    Total penalty determines the final grade.

    Args:
        metrics: Computed image quality metrics.
        blur_thresholds: (reject_below, good_above) for Laplacian variance.
        brightness_range: (too_dark, too_bright) acceptable range.
        contrast_thresholds: (reject_below, good_above) for grayscale std.
        foreground_min: (reject_below, good_above) for foreground ratio.
        entropy_thresholds: (reject_below, good_above) for entropy.

    Returns:
        Quality grade: 0 (good), 1 (usable), 2 (reject).
    """
    penalty = 0.0

    # Blur assessment
    if metrics.blur_score < blur_thresholds[0]:
        penalty += 3.0  # Severe blur -> likely reject
    elif metrics.blur_score < blur_thresholds[1]:
        penalty += 1.0  # Mild blur

    # Brightness assessment
    if metrics.brightness < brightness_range[0] or metrics.brightness > brightness_range[1]:
        penalty += 2.5  # Extreme exposure
    elif metrics.brightness < 60 or metrics.brightness > 180:
        penalty += 1.0  # Mild exposure issue

    # Contrast assessment
    if metrics.contrast < contrast_thresholds[0]:
        penalty += 2.0  # Very low contrast (washed out or uniform)
    elif metrics.contrast < contrast_thresholds[1]:
        penalty += 0.5

    # Foreground ratio: too little fundus visible
    if metrics.foreground_ratio < foreground_min[0]:
        penalty += 3.0  # Mostly black, fundus barely visible
    elif metrics.foreground_ratio < foreground_min[1]:
        penalty += 1.0

    # Entropy: low entropy means low information content
    if metrics.entropy < entropy_thresholds[0]:
        penalty += 2.0
    elif metrics.entropy < entropy_thresholds[1]:
        penalty += 0.5

    # Map penalty to grade
    if penalty >= 4.0:
        return 2  # Reject
    elif penalty >= 1.5:
        return 1  # Usable
    else:
        return 0  # Good


def generate_quality_labels(
    image_dir: str,
    output_csv: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"),
    recursive: bool = True,
) -> Dict[str, int]:
    """
    Scan a directory of fundus images and generate synthetic quality labels.

    Writes a CSV with columns: [image_path, quality_grade, blur, brightness,
    contrast, saturation, foreground_ratio, entropy].

    Args:
        image_dir: Directory containing fundus images.
        output_csv: Path to write the output CSV.
        extensions: Accepted image file extensions.
        recursive: Whether to search subdirectories.

    Returns:
        Summary dict: {"good": N, "usable": N, "reject": N, "failed": N}
    """
    image_dir = Path(image_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths: List[Path] = []
    pattern = "**/*" if recursive else "*"
    for ext in extensions:
        image_paths.extend(image_dir.glob(f"{pattern}{ext}"))
    image_paths.sort()

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    summary = {"good": 0, "usable": 0, "reject": 0, "failed": 0}
    grade_names = {0: "good", 1: "usable", 2: "reject"}

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "quality_grade", "quality_label",
            "blur", "brightness", "contrast",
            "saturation", "foreground_ratio", "entropy",
        ])

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    summary["failed"] += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                metrics = compute_quality_metrics(img)
                grade = classify_quality(metrics)
                label = grade_names[grade]
                summary[label] += 1

                rel_path = img_path.relative_to(image_dir)
                writer.writerow([
                    str(rel_path),
                    grade,
                    label,
                    f"{metrics.blur_score:.2f}",
                    f"{metrics.brightness:.2f}",
                    f"{metrics.contrast:.2f}",
                    f"{metrics.saturation:.2f}",
                    f"{metrics.foreground_ratio:.4f}",
                    f"{metrics.entropy:.4f}",
                ])
            except Exception as e:
                summary["failed"] += 1
                print(f"WARNING: Failed to process {img_path}: {e}")

    total = sum(summary.values())
    print(f"Quality labels generated for {total} images -> {output_path}")
    print(f"  Good: {summary['good']}, Usable: {summary['usable']}, "
          f"Reject: {summary['reject']}, Failed: {summary['failed']}")

    return summary


def load_quality_labels(csv_path: str) -> Dict[str, int]:
    """
    Load quality labels from a previously generated CSV.

    Args:
        csv_path: Path to the quality labels CSV.

    Returns:
        Dict mapping relative image path -> quality grade (0, 1, 2).
    """
    labels = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["image_path"]] = int(row["quality_grade"])
    return labels
