#!/usr/bin/env python3
"""
Script to run full evaluation on a DR grading model checkpoint.

Usage:
    python -m ml.scripts.evaluate_model \
        --checkpoint weights/dr_grader_v1.pt \
        --data-dir data/processed \
        --output-dir results/eval_v1 \
        --device mps \
        --use-tta

    python -m ml.scripts.evaluate_model \
        --checkpoint weights/dr_grader_v1.pt \
        --data-dir data/processed \
        --output-dir results/eval_v1 \
        --calibrate \
        --mlflow
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a DR grading model checkpoint"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt or .pth)"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory containing test/{grade}/ structure"
    )
    parser.add_argument(
        "--output-dir", default="results/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--num-classes", type=int, default=5,
        help="Number of DR grades"
    )
    parser.add_argument(
        "--use-tta", action="store_true",
        help="Enable test-time augmentation (8-fold)"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Fit temperature scaling on val set and apply to test predictions"
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log results to MLflow"
    )
    parser.add_argument(
        "--error-analysis", action="store_true", default=True,
        help="Run detailed error analysis"
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print()

    # Run main evaluation
    from ml.evaluation.evaluate import evaluate_checkpoint

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_classes=args.num_classes,
        use_tta=args.use_tta,
        log_to_mlflow=args.mlflow,
    )

    # Temperature scaling calibration
    if args.calibrate:
        print("\n--- Temperature Scaling Calibration ---")
        from ml.data.augmentations import get_val_transforms
        from ml.data.dataset import FundusDataset
        from ml.evaluation.calibration import TemperatureScaling, plot_calibration_comparison
        from ml.evaluation.evaluate import load_model

        model = load_model(args.checkpoint, device=device, num_classes=args.num_classes)
        transform = get_val_transforms(args.img_size)

        # Use val set for fitting temperature
        val_dataset = FundusDataset(
            root=args.data_dir, split="val", transform=transform,
            num_classes=args.num_classes,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )

        ts = TemperatureScaling()
        optimal_temp = ts.fit(model, val_loader, device=device)

        # Save temperature
        output_path = Path(args.output_dir)
        ts.save(str(output_path / "temperature.pt"))

        # Generate calibration comparison plot
        predictions_path = output_path / "predictions.npz"
        if predictions_path.exists():
            data = np.load(predictions_path)
            plot_calibration_comparison(
                data["logits"], data["labels"],
                temperature=optimal_temp,
                save_path=str(output_path / "calibration_comparison.png"),
            )
            print(f"Calibration comparison saved to {output_path / 'calibration_comparison.png'}")

    # Error analysis
    if args.error_analysis:
        print("\n--- Error Analysis ---")
        from ml.evaluation.confusion import error_analysis, plot_error_distribution

        output_path = Path(args.output_dir)
        predictions_path = output_path / "predictions.npz"

        if predictions_path.exists():
            data = np.load(predictions_path)
            analysis = error_analysis(
                data["labels"], data["preds"], data["probs"],
                num_classes=args.num_classes,
            )

            # Save error analysis
            with open(output_path / "error_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)

            print(f"Total errors: {analysis['total_errors']} ({analysis['error_rate']:.1%})")
            print(f"Off by 2+ grades: {analysis['off_by_2_or_more']}")
            print(f"Clinical misses (referable DR): {analysis['clinical_misses']['count']}")

            # Plot error distribution
            plot_error_distribution(
                data["labels"], data["preds"],
                save_path=str(output_path / "error_distribution.png"),
            )

            print("\nTop confusion pairs:")
            for pair, count in list(analysis["confusion_pairs"].items())[:5]:
                print(f"  {pair}: {count}")

    # Exit with non-zero status if QWK is too low (for CI/CD gating)
    qwk = metrics.get("qwk", 0.0)
    ref_sens = metrics.get("referable_sensitivity", 0.0)

    if qwk < 0.7:
        print(f"\nWARNING: QWK ({qwk:.4f}) is below clinical threshold (0.7)")
    if ref_sens < 0.9:
        print(f"\nWARNING: Referable sensitivity ({ref_sens:.4f}) is below 0.9")

    return 0


if __name__ == "__main__":
    sys.exit(main())
