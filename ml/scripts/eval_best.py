"""Evaluate the best trained DR model on the test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from ml.scripts.train_aptos import DRGrader, evaluate
from ml.data.dataset import FundusDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DRGrader(num_classes=5).to(device)
    ckpt = torch.load("checkpoints/dr_aptos/best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from epoch {ckpt['epoch']+1} (val QWK: {ckpt['best_qwk']:.4f})")

    test_ds = FundusDataset("data/aptos_split", "test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTEST RESULTS (APTOS 2019 — {len(test_ds)} images):")
    print(f"  QWK:         {test_metrics['qwk']:.4f}")
    print(f"  AUC-ROC:     {test_metrics['auc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
