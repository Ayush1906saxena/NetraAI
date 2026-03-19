import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class FundusDataset(Dataset):
    """
    PyTorch Dataset for fundus images organized in class-folder structure:
        root/{split}/{grade}/image.jpg

    Supports weighted sampling to handle severe class imbalance typical
    in DR datasets (grade 0 dominates, grade 4 is rare).
    """

    GRADE_NAMES = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        num_classes: int = 5,
    ):
        self.root = Path(root) / split
        self.split = split
        self.num_classes = num_classes

        # Default transforms if none provided
        if transform is not None:
            self.transform = transform
        else:
            from ml.data.augmentations import get_train_transforms, get_val_transforms
            self.transform = get_train_transforms() if split == "train" else get_val_transforms()

        self.samples: List[Tuple[Path, int]] = []
        self.class_counts: Dict[int, int] = {i: 0 for i in range(num_classes)}

        if not self.root.exists():
            raise FileNotFoundError(f"Split directory not found: {self.root}")

        for grade_dir in sorted(self.root.iterdir()):
            if not grade_dir.is_dir():
                continue
            try:
                grade = int(grade_dir.name)
            except ValueError:
                continue
            if grade < 0 or grade >= num_classes:
                continue

            for img_path in sorted(grade_dir.iterdir()):
                suffix = img_path.suffix.lower()
                if suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
                    self.samples.append((img_path, grade))
                    self.class_counts[grade] += 1

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root}. Expected structure: "
                f"{self.root}/{{0,1,2,3,4}}/image.jpg"
            )

        self.labels = np.array([s[1] for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, grade = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        transformed = self.transform(image=img_np)
        img_tensor = transformed["image"]

        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()

        return img_tensor, grade

    def get_sample_weights(self) -> np.ndarray:
        """
        Compute per-sample weights inversely proportional to class frequency.
        This ensures rare grades (3, 4) are sampled more often during training.
        """
        total = len(self.samples)
        class_weights = {}
        for cls, count in self.class_counts.items():
            if count > 0:
                class_weights[cls] = total / (self.num_classes * count)
            else:
                class_weights[cls] = 0.0

        sample_weights = np.array(
            [class_weights[label] for label in self.labels], dtype=np.float64
        )
        return sample_weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Return a WeightedRandomSampler for use with DataLoader.
        Balances class distribution during training.
        """
        weights = self.get_sample_weights()
        return WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        )

    def get_class_distribution(self) -> Dict[str, int]:
        """Return human-readable class distribution."""
        return {
            f"{grade} ({self.GRADE_NAMES.get(grade, 'Unknown')})": count
            for grade, count in sorted(self.class_counts.items())
        }

    def __repr__(self) -> str:
        dist = ", ".join(
            f"G{g}:{c}" for g, c in sorted(self.class_counts.items())
        )
        return (
            f"FundusDataset(split={self.split}, samples={len(self)}, "
            f"distribution=[{dist}])"
        )
