#!/usr/bin/env python3
"""
Script to export DR and IQA models to all deployment formats.

Exports:
    - ONNX (server inference via ONNX Runtime)
    - CoreML (iOS/macOS on-device inference)
    - TFLite (Android on-device IQA)

Usage:
    python -m ml.scripts.export_all \
        --dr-checkpoint weights/dr_grader_v1.pt \
        --iqa-checkpoint weights/iqa_v1.pt \
        --output-dir exports/v1

    # Export only DR model to ONNX:
    python -m ml.scripts.export_all \
        --dr-checkpoint weights/dr_grader_v1.pt \
        --output-dir exports/v1 \
        --formats onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export models to all deployment formats"
    )
    parser.add_argument(
        "--dr-checkpoint",
        help="Path to DR grading model checkpoint"
    )
    parser.add_argument(
        "--iqa-checkpoint",
        help="Path to IQA model checkpoint"
    )
    parser.add_argument(
        "--output-dir", default="exports",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--formats", nargs="+",
        default=["onnx", "coreml", "tflite"],
        choices=["onnx", "coreml", "tflite"],
        help="Export formats to generate"
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
        "--device", default="cpu",
        help="Device for export tracing"
    )
    parser.add_argument(
        "--tflite-quantize", default="float16",
        choices=["none", "float16", "int8"],
        help="TFLite quantization mode"
    )

    args = parser.parse_args()

    if not args.dr_checkpoint and not args.iqa_checkpoint:
        parser.error("At least one of --dr-checkpoint or --iqa-checkpoint is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_start = time.time()

    # -----------------------------------------------------------------------
    # DR Model exports
    # -----------------------------------------------------------------------
    if args.dr_checkpoint:
        print("=" * 60)
        print("EXPORTING DR GRADING MODEL")
        print("=" * 60)

        # ONNX
        if "onnx" in args.formats:
            print("\n--- ONNX Export ---")
            try:
                from ml.export.to_onnx import export_dr_model_to_onnx

                start = time.time()
                onnx_path = export_dr_model_to_onnx(
                    args.dr_checkpoint,
                    str(output_dir / "dr_grader.onnx"),
                    num_classes=args.num_classes,
                    img_size=args.img_size,
                    device=args.device,
                )
                elapsed = time.time() - start
                results["dr_onnx"] = {
                    "status": "success",
                    "path": onnx_path,
                    "time_sec": round(elapsed, 1),
                    "size_mb": round(Path(onnx_path).stat().st_size / 1024 / 1024, 1),
                }
                print(f"Completed in {elapsed:.1f}s")
            except Exception as e:
                results["dr_onnx"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

        # CoreML
        if "coreml" in args.formats:
            print("\n--- CoreML Export ---")
            try:
                from ml.export.to_coreml import export_dr_to_coreml

                start = time.time()
                coreml_path = export_dr_to_coreml(
                    args.dr_checkpoint,
                    str(output_dir / "dr_grader.mlpackage"),
                    num_classes=args.num_classes,
                    img_size=args.img_size,
                    device=args.device,
                )
                elapsed = time.time() - start
                results["dr_coreml"] = {
                    "status": "success",
                    "path": coreml_path,
                    "time_sec": round(elapsed, 1),
                }
                print(f"Completed in {elapsed:.1f}s")
            except Exception as e:
                results["dr_coreml"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

        # TFLite
        if "tflite" in args.formats:
            print("\n--- TFLite Export ---")
            try:
                from ml.export.to_tflite import export_dr_to_tflite

                start = time.time()
                tflite_path = export_dr_to_tflite(
                    args.dr_checkpoint,
                    str(output_dir / "dr_grader.tflite"),
                    num_classes=args.num_classes,
                    img_size=args.img_size,
                    quantize=args.tflite_quantize,
                    device=args.device,
                )
                elapsed = time.time() - start
                results["dr_tflite"] = {
                    "status": "success",
                    "path": tflite_path,
                    "time_sec": round(elapsed, 1),
                    "size_mb": round(Path(tflite_path).stat().st_size / 1024 / 1024, 1),
                }
                print(f"Completed in {elapsed:.1f}s")
            except Exception as e:
                results["dr_tflite"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

    # -----------------------------------------------------------------------
    # IQA Model exports
    # -----------------------------------------------------------------------
    if args.iqa_checkpoint:
        print("\n" + "=" * 60)
        print("EXPORTING IQA MODEL")
        print("=" * 60)

        # ONNX
        if "onnx" in args.formats:
            print("\n--- ONNX Export ---")
            try:
                from ml.export.to_onnx import export_to_onnx
                from ml.models.iqa_model import FundusIQA

                model = FundusIQA.from_checkpoint(args.iqa_checkpoint, device=args.device)

                class IQAWrapper(nn.Module):
                    def __init__(self, m):
                        super().__init__()
                        self.m = m
                    def forward(self, x):
                        out = self.m(x)
                        import torch
                        return torch.cat([out["quality"], torch.sigmoid(out["gradeable"]), torch.sigmoid(out["guidance"])], dim=1)

                import torch.nn as nn

                wrapper = IQAWrapper(model)
                wrapper.eval()

                start = time.time()
                onnx_path = export_to_onnx(
                    wrapper,
                    str(output_dir / "iqa.onnx"),
                    input_shape=(1, 3, args.img_size, args.img_size),
                    device=args.device,
                )
                elapsed = time.time() - start
                results["iqa_onnx"] = {
                    "status": "success",
                    "path": onnx_path,
                    "time_sec": round(elapsed, 1),
                    "size_mb": round(Path(onnx_path).stat().st_size / 1024 / 1024, 1),
                }
            except Exception as e:
                results["iqa_onnx"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

        # CoreML
        if "coreml" in args.formats:
            print("\n--- CoreML Export ---")
            try:
                from ml.export.to_coreml import export_iqa_to_coreml

                start = time.time()
                coreml_path = export_iqa_to_coreml(
                    args.iqa_checkpoint,
                    str(output_dir / "iqa.mlpackage"),
                    img_size=args.img_size,
                    device=args.device,
                )
                elapsed = time.time() - start
                results["iqa_coreml"] = {
                    "status": "success",
                    "path": coreml_path,
                    "time_sec": round(elapsed, 1),
                }
            except Exception as e:
                results["iqa_coreml"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

        # TFLite (INT8 for IQA — fast on-device inference)
        if "tflite" in args.formats:
            print("\n--- TFLite Export (INT8) ---")
            try:
                from ml.export.to_tflite import export_iqa_to_tflite

                start = time.time()
                tflite_path = export_iqa_to_tflite(
                    args.iqa_checkpoint,
                    str(output_dir / "iqa_int8.tflite"),
                    img_size=args.img_size,
                    quantize="int8",
                    device=args.device,
                )
                elapsed = time.time() - start
                results["iqa_tflite"] = {
                    "status": "success",
                    "path": tflite_path,
                    "time_sec": round(elapsed, 1),
                    "size_mb": round(Path(tflite_path).stat().st_size / 1024 / 1024, 1),
                }
            except Exception as e:
                results["iqa_tflite"] = {"status": "failed", "error": str(e)}
                print(f"FAILED: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    successes = 0
    failures = 0
    for name, info in results.items():
        status = info["status"]
        if status == "success":
            print(f"  {name}: OK ({info.get('size_mb', '?')} MB, {info.get('time_sec', '?')}s)")
            successes += 1
        else:
            print(f"  {name}: FAILED - {info.get('error', 'unknown')}")
            failures += 1

    print(f"\nTotal: {successes} succeeded, {failures} failed ({total_elapsed:.1f}s)")

    # Save manifest
    manifest_path = output_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
