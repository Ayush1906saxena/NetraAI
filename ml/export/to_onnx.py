"""
Export PyTorch DR grading model to ONNX format.

ONNX enables deployment across multiple runtimes:
- ONNX Runtime (CPU/GPU, optimized for server inference)
- TensorRT (NVIDIA GPU, via ONNX → TRT conversion)
- OpenVINO (Intel hardware)
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    simplify: bool = True,
    device: str = "cpu",
    verify: bool = True,
) -> str:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Destination .onnx file path.
        input_shape: Example input shape (batch, channels, height, width).
        opset_version: ONNX opset version. 17 supports all ViT ops.
        dynamic_batch: If True, allow variable batch size at inference.
        simplify: If True, run onnx-simplifier to optimize the graph.
        device: Device for tracing.
        verify: If True, verify output matches between PyTorch and ONNX.

    Returns:
        Path to the saved ONNX file.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    print(f"Exporting to ONNX (opset={opset_version}) ...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    print(f"ONNX model saved to {output_path}")

    # Validate the exported model
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed.")

    # Simplify if requested
    if simplify:
        try:
            import onnxsim

            print("Running onnx-simplifier ...")
            simplified, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified, output_path)
                print("Simplified ONNX model saved.")
            else:
                print("Simplification check failed; keeping original model.")
        except ImportError:
            print("onnxsim not installed; skipping simplification. "
                  "Install with: pip install onnxsim")

    # Verify output correctness
    if verify:
        _verify_onnx(model, output_path, dummy_input)

    # Report model size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.1f} MB")

    return output_path


def _verify_onnx(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    atol: float = 1e-4,
) -> None:
    """Verify that ONNX model produces the same output as PyTorch."""
    try:
        import onnxruntime as ort

        # PyTorch output
        with torch.no_grad():
            pt_output = pytorch_model(dummy_input).cpu().numpy()

        # ONNX Runtime output
        session = ort.InferenceSession(onnx_path)
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_output = session.run(None, ort_inputs)[0]

        # Compare
        max_diff = np.max(np.abs(pt_output - ort_output))
        mean_diff = np.mean(np.abs(pt_output - ort_output))
        print(f"Verification: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if max_diff < atol:
            print("ONNX output matches PyTorch output within tolerance.")
        else:
            print(
                f"WARNING: ONNX output differs from PyTorch by {max_diff:.6f} "
                f"(tolerance: {atol})"
            )
    except ImportError:
        print("onnxruntime not installed; skipping verification. "
              "Install with: pip install onnxruntime")


def export_dr_model_to_onnx(
    checkpoint_path: str,
    output_path: str,
    num_classes: int = 5,
    img_size: int = 224,
    device: str = "cpu",
) -> str:
    """
    High-level function to export a DR grading checkpoint to ONNX.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        output_path: Destination .onnx file.
        num_classes: Number of DR grades.
        img_size: Input image size.
        device: Device for export.

    Returns:
        Path to saved ONNX file.
    """
    from ml.evaluation.evaluate import load_model

    model = load_model(checkpoint_path, device=device, num_classes=num_classes)

    return export_to_onnx(
        model,
        output_path,
        input_shape=(1, 3, img_size, img_size),
        device=device,
    )


def benchmark_onnx(
    onnx_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Benchmark ONNX Runtime inference latency.

    Returns dict with mean, median, p95, p99 latency in milliseconds.
    """
    import time

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(*input_shape).astype(np.float32)
    inputs = {"input": dummy}

    # Warmup
    for _ in range(num_warmup):
        session.run(None, inputs)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(latencies.std()),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DR model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    export_dr_model_to_onnx(
        args.checkpoint, args.output,
        num_classes=args.num_classes,
        img_size=args.img_size,
        device=args.device,
    )
