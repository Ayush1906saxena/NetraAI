import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from filename.
    Handles common naming conventions:
        - APTOS: {id}.png  (numeric id)
        - EyePACS: {patient_id}_{left|right}.jpeg
        - IDRiD: IDRiD_{number}.jpg
        - Generic: take everything before the last underscore or dot
    """
    stem = Path(filename).stem

    # EyePACS pattern: patientid_left / patientid_right
    match = re.match(r"^(\d+)_(left|right)", stem)
    if match:
        return match.group(1)

    # IDRiD pattern: IDRiD_XXX
    match = re.match(r"^(IDRiD_\d+)", stem)
    if match:
        return match.group(1)

    # Generic: split on underscore, take first part as patient ID
    parts = stem.split("_")
    if len(parts) > 1:
        return parts[0]

    # Fallback: entire stem is the patient ID
    return stem


def create_stratified_split(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    csv_path: Optional[str] = None,
) -> Dict[str, int]:
    """
    Create stratified train/val/test splits by PATIENT, not by image.

    This is critical for medical imaging: if a patient has multiple images
    (e.g., left eye, right eye, multiple visits), ALL of that patient's
    images must land in the same split. Otherwise you get data leakage
    and inflated metrics.

    Args:
        data_dir: Source directory with images. Can be flat or class-folder.
        output_dir: Destination directory. Creates {train,val,test}/{grade}/ subfolders.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        csv_path: Optional CSV with columns [image, label]. If None, assumes
                  class-folder structure: data_dir/{grade}/image.ext

    Returns:
        Dict with split counts: {"train": N, "val": N, "test": N}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Collect all (image_path, label, patient_id) tuples
    records: List[Tuple[Path, int, str]] = []

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        assert "image" in df.columns and "label" in df.columns, \
            "CSV must have 'image' and 'label' columns"
        for _, row in df.iterrows():
            img_path = data_dir / str(row["image"])
            if img_path.exists():
                patient_id = extract_patient_id(row["image"])
                records.append((img_path, int(row["label"]), patient_id))
    else:
        # Assume class-folder structure: data_dir/{grade}/image.ext
        for grade_dir in sorted(data_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            try:
                grade = int(grade_dir.name)
            except ValueError:
                continue
            for img_path in sorted(grade_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
                    patient_id = extract_patient_id(img_path.name)
                    records.append((img_path, grade, patient_id))

    if not records:
        raise RuntimeError(f"No images found in {data_dir}")

    # Build patient-level dataframe for stratified splitting
    df = pd.DataFrame(records, columns=["path", "label", "patient_id"])

    # Get the majority label for each patient (for stratification)
    patient_labels = (
        df.groupby("patient_id")["label"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    patient_labels.columns = ["patient_id", "stratify_label"]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    patients_train, patients_valtest = train_test_split(
        patient_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=patient_labels["stratify_label"],
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    patients_val, patients_test = train_test_split(
        patients_valtest,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=patients_valtest["stratify_label"],
    )

    # Map patient IDs to splits
    split_map = {}
    for pid in patients_train["patient_id"]:
        split_map[pid] = "train"
    for pid in patients_val["patient_id"]:
        split_map[pid] = "val"
    for pid in patients_test["patient_id"]:
        split_map[pid] = "test"

    # Copy images into split directories
    counts = {"train": 0, "val": 0, "test": 0}
    for img_path, label, patient_id in records:
        split = split_map[patient_id]
        dest_dir = output_dir / split / str(label)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / img_path.name
        shutil.copy2(str(img_path), str(dest_file))
        counts[split] += 1

    # Save split metadata for reproducibility
    meta_rows = []
    for img_path, label, patient_id in records:
        meta_rows.append({
            "image": img_path.name,
            "label": label,
            "patient_id": patient_id,
            "split": split_map[patient_id],
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(output_dir / "split_metadata.csv", index=False)

    print(f"Split complete: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"Patients: train={len(patients_train)}, val={len(patients_val)}, test={len(patients_test)}")
    print(f"Metadata saved to {output_dir / 'split_metadata.csv'}")

    return counts
