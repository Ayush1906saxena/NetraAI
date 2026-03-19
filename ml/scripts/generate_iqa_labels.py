#!/usr/bin/env python3
"""
Generate synthetic IQA labels from APTOS 2019 dataset.

Computes image quality metrics and assigns:
  - quality_grade: 0=good, 1=usable, 2=reject
  - quality_score: float in [0, 1] (1=best)
  - gradeable: 0 or 1
  - guidance: 0=ok, 1=too_blurry, 2=bad_exposure, 3=misaligned

Saves results to data/aptos/quality_labels.csv
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.quality_labels import compute_quality_metrics


def classify_quality_custom(metrics):
    """
    Classify quality using thresholds adapted to APTOS fundus image statistics.

    Fundus images naturally have lower Laplacian variance due to large dark
    borders and soft retinal tissue. Thresholds are calibrated to the actual
    distribution (median blur ~15, 75th ~42).

    Thresholds:
        blur < 5    -> reject (very blurry)
        blur < 10   -> usable (slightly blurry)
        brightness < 30 or > 120 -> bad exposure (reject if extreme)
        foreground_ratio < 0.15 -> misaligned / too much border

    Returns:
        (quality_grade, quality_score, gradeable, guidance)
    """
    blur = metrics.blur_score
    brightness = metrics.brightness
    fg_ratio = metrics.foreground_ratio
    contrast = metrics.contrast

    # Quality grade: 0=good, 1=usable, 2=reject
    if blur < 5:
        quality_grade = 2  # reject (very blurry)
    elif blur < 10:
        quality_grade = 1  # usable (slightly blurry)
    elif brightness < 30 or brightness > 120:
        quality_grade = 2  # reject (bad exposure)
    elif fg_ratio < 0.15:
        quality_grade = 2  # reject (mostly black / misaligned)
    elif contrast < 15:
        quality_grade = 1  # usable (low contrast)
    elif blur < 20:
        quality_grade = 1  # usable (moderate blur)
    else:
        quality_grade = 0  # good

    # Quality score: continuous [0, 1] (higher = better)
    blur_score = min(blur / 80.0, 1.0)
    bright_score = 1.0 - abs(brightness - 70.0) / 70.0
    bright_score = max(bright_score, 0.0)
    fg_score = min(fg_ratio / 0.5, 1.0)
    quality_score = 0.4 * blur_score + 0.3 * bright_score + 0.3 * fg_score
    quality_score = max(0.0, min(1.0, quality_score))

    # Gradeable: 1 if not reject, 0 if reject
    gradeable = 0 if quality_grade == 2 else 1

    # Guidance: 0=ok, 1=too_blurry, 2=bad_exposure, 3=misaligned
    if blur < 5:
        guidance = 1  # too_blurry
    elif brightness < 30 or brightness > 120:
        guidance = 2  # bad_exposure
    elif fg_ratio < 0.15:
        guidance = 3  # misaligned
    else:
        guidance = 0  # ok

    return quality_grade, quality_score, gradeable, guidance


def main():
    image_dir = PROJECT_ROOT / "data" / "aptos" / "train_images"
    output_csv = PROJECT_ROOT / "data" / "aptos" / "quality_labels.csv"

    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    # Collect all PNG images
    image_paths = sorted(image_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        print("No images found!")
        sys.exit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    counts = {"good": 0, "usable": 0, "reject": 0, "failed": 0}
    guidance_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "quality_grade", "quality_score", "gradeable",
            "guidance", "blur", "brightness", "contrast",
            "saturation", "foreground_ratio",
        ])

        for i, img_path in enumerate(image_paths):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    counts["failed"] += 1
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                metrics = compute_quality_metrics(img_rgb)
                grade, score, gradeable, guidance = classify_quality_custom(metrics)

                grade_name = {0: "good", 1: "usable", 2: "reject"}[grade]
                counts[grade_name] += 1
                guidance_counts[guidance] += 1

                image_id = img_path.stem  # filename without extension

                writer.writerow([
                    image_id,
                    grade,
                    f"{score:.4f}",
                    gradeable,
                    guidance,
                    f"{metrics.blur_score:.2f}",
                    f"{metrics.brightness:.2f}",
                    f"{metrics.contrast:.2f}",
                    f"{metrics.saturation:.2f}",
                    f"{metrics.foreground_ratio:.4f}",
                ])

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images...")

            except Exception as e:
                counts["failed"] += 1
                print(f"WARNING: Failed to process {img_path.name}: {e}")

    total = sum(counts.values())
    print(f"\nQuality labels saved to {output_csv}")
    print(f"Total: {total} images")
    print(f"  Good:   {counts['good']} ({100*counts['good']/max(total,1):.1f}%)")
    print(f"  Usable: {counts['usable']} ({100*counts['usable']/max(total,1):.1f}%)")
    print(f"  Reject: {counts['reject']} ({100*counts['reject']/max(total,1):.1f}%)")
    print(f"  Failed: {counts['failed']}")
    print(f"\nGuidance distribution:")
    guidance_names = {0: "ok", 1: "too_blurry", 2: "bad_exposure", 3: "misaligned"}
    for k, v in guidance_counts.items():
        print(f"  {guidance_names[k]}: {v}")


if __name__ == "__main__":
    main()
