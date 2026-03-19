"""
Export PyTorch models to TensorFlow Lite format for Android deployment.

TFLite enables on-device IQA on Android tablets used in field screenings.
The export pipeline is: PyTorch -> ONNX -> TensorFlow -> TFLite.
Supports INT8 quantization for smallest model size and fastest inference.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def export_to_tflite(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    quantize: str = "float16",
    calibration_data: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> str:
    """
    Export a PyTorch model to TFLite via ONNX -> TF -> TFLite pipeline.

    Args:
        model: PyTorch model in eval mode.
        output_path: Destination .tflite file path.
        input_shape: (batch, channels, height, width).
        quantize: Quantization mode:
                  - "none": float32 (largest, most accurate)
                  - "float16": float16 weights (good balance)
                  - "int8": full INT8 quantization (smallest, needs calibration)
        calibration_data: For INT8 quantization, provide representative
                          calibration data as (N, C, H, W) numpy array.
        device: Device for tracing.

    Returns:
        Path to the saved TFLite file.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Export to ONNX
        onnx_path = os.path.join(tmpdir, "model.onnx")
        dummy_input = torch.randn(*input_shape, device=device)

        print("Step 1/3: Exporting to ONNX ...")
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )

        # Step 2: Convert ONNX to TensorFlow SavedModel
        print("Step 2/3: Converting ONNX to TensorFlow ...")
        try:
            import onnx
            from onnx_tf.backend import prepare

            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            tf_saved_model_path = os.path.join(tmpdir, "saved_model")
            tf_rep.export_graph(tf_saved_model_path)
        except ImportError:
            raise ImportError(
                "onnx-tf is required for TFLite export. "
                "Install with: pip install onnx-tf tensorflow"
            )

        # Step 3: Convert TF SavedModel to TFLite
        print("Step 3/3: Converting to TFLite ...")
        try:
            import tensorflow as tf

            converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)

            # Quantization settings
            if quantize == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif quantize == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

                if calibration_data is not None:
                    def representative_dataset():
                        for i in range(min(len(calibration_data), 200)):
                            # TFLite expects NHWC format
                            sample = calibration_data[i:i+1].transpose(0, 2, 3, 1)
                            yield [sample.astype(np.float32)]

                    converter.representative_dataset = representative_dataset
                else:
                    # Generate random calibration data
                    def representative_dataset():
                        for _ in range(100):
                            sample = np.random.randn(1, input_shape[2], input_shape[3], input_shape[1])
                            yield [sample.astype(np.float32)]

                    converter.representative_dataset = representative_dataset
            elif quantize == "none":
                pass  # No quantization
            else:
                raise ValueError(f"Unknown quantize mode: {quantize}")

            tflite_model = converter.convert()

        except ImportError:
            raise ImportError(
                "tensorflow is required for TFLite export. "
                "Install with: pip install tensorflow"
            )

    # Save the TFLite model
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"TFLite model saved to {output_path} ({file_size_mb:.1f} MB)")

    return output_path


def export_iqa_to_tflite(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 224,
    quantize: str = "int8",
    device: str = "cpu",
) -> str:
    """
    Export the IQA model to TFLite for on-device quality assessment on Android.

    INT8 quantization is recommended for IQA since:
    - MobileNetV3-Small is designed for quantization.
    - Speed matters more than precision for quality gating.
    - INT8 enables running on DSP/NPU accelerators.

    Args:
        checkpoint_path: Path to IQA model checkpoint.
        output_path: Destination .tflite file.
        img_size: Input image size.
        quantize: Quantization mode ("none", "float16", "int8").
        device: Device for export.

    Returns:
        Path to saved TFLite model.
    """
    from ml.models.iqa_model import FundusIQA

    model = FundusIQA.from_checkpoint(checkpoint_path, device=device)

    # Wrapper for single-tensor output
    class IQAExportWrapper(nn.Module):
        def __init__(self, iqa_model: FundusIQA):
            super().__init__()
            self.model = iqa_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.model(x)
            quality = out["quality"]
            gradeable = torch.sigmoid(out["gradeable"])
            guidance = torch.sigmoid(out["guidance"])
            return torch.cat([quality, gradeable, guidance], dim=1)

    wrapper = IQAExportWrapper(model)
    wrapper.eval()

    # Generate calibration data for INT8
    calibration_data = None
    if quantize == "int8":
        calibration_data = np.random.randn(100, 3, img_size, img_size).astype(np.float32)

    return export_to_tflite(
        wrapper,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        quantize=quantize,
        calibration_data=calibration_data,
        device=device,
    )


def export_dr_to_tflite(
    checkpoint_path: str,
    output_path: str,
    num_classes: int = 5,
    img_size: int = 224,
    quantize: str = "float16",
    device: str = "cpu",
) -> str:
    """
    Export the DR grading model to TFLite.

    Float16 quantization is recommended for DR grading since accuracy
    matters more than speed for the diagnostic model.
    """
    from ml.evaluation.evaluate import load_model

    model = load_model(checkpoint_path, device=device, num_classes=num_classes)

    return export_to_tflite(
        model,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        quantize=quantize,
        device=device,
    )


def benchmark_tflite(
    tflite_path: str,
    input_shape: tuple = (1, 224, 224, 3),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Benchmark TFLite inference latency.

    Args:
        tflite_path: Path to .tflite file.
        input_shape: Input shape in NHWC format.
        num_warmup: Warmup iterations.
        num_runs: Benchmark iterations.

    Returns:
        Dict with latency statistics in milliseconds.
    """
    import time

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    dummy = np.random.randn(*input_shape).astype(input_dtype)

    # Warmup
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(latencies.std()),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to TFLite")
    parser.add_argument("--model-type", choices=["dr", "iqa"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--quantize", choices=["none", "float16", "int8"], default="float16")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.model_type == "dr":
        export_dr_to_tflite(
            args.checkpoint, args.output,
            img_size=args.img_size, quantize=args.quantize, device=args.device,
        )
    else:
        export_iqa_to_tflite(
            args.checkpoint, args.output,
            img_size=args.img_size, quantize=args.quantize, device=args.device,
        )
