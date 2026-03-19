"""
Export the trained EfficientNet-B3 DR grading model to ONNX format.

Loads the checkpoint from checkpoints/dr_aptos/best.pth, exports to ONNX,
validates output correctness, and benchmarks PyTorch vs ONNX Runtime speed.

Usage:
    python -m ml.scripts.export_dr_onnx
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "dr_aptos" / "best.pth"
ONNX_OUTPUT_PATH = PROJECT_ROOT / "checkpoints" / "dr_aptos" / "dr_model.onnx"

INPUT_SHAPE = (1, 3, 224, 224)
NUM_CLASSES = 5
OPSET_VERSION = 17
NUM_WARMUP = 10
NUM_BENCHMARK_RUNS = 50


def load_dr_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Load the DRGrader model from the trained checkpoint."""
    from server.services.inference_v2 import DRGrader

    model = DRGrader(num_classes=NUM_CLASSES)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        best_qwk = checkpoint.get("best_qwk", "?")
        print(f"  Loaded best checkpoint (epoch={epoch}, QWK={best_qwk})")
    else:
        model.load_state_dict(checkpoint)
        print("  Loaded raw state_dict checkpoint")

    model.to(device).eval()
    return model


def export_to_onnx(model: torch.nn.Module, output_path: Path, device: str = "cpu") -> None:
    """Export the PyTorch model to ONNX format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*INPUT_SHAPE, device=device)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    print(f"\nExporting to ONNX (opset={OPSET_VERSION}) ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
    print(f"  Saved to {output_path}")


def validate_onnx(onnx_path: Path) -> None:
    """Check the ONNX model is structurally valid."""
    import onnx

    print("\nValidating ONNX model ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model validation passed.")


def verify_outputs(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    device: str = "cpu",
    atol: float = 1e-4,
) -> bool:
    """Verify that the ONNX model produces the same outputs as PyTorch."""
    import onnxruntime as ort

    print("\nVerifying output equivalence ...")
    dummy_input = torch.randn(*INPUT_SHAPE, device=device)

    # PyTorch inference
    with torch.no_grad():
        pt_output = pytorch_model(dummy_input).cpu().numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {"input": dummy_input.cpu().numpy()}
    ort_output = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_output - ort_output)))
    mean_diff = float(np.mean(np.abs(pt_output - ort_output)))

    print(f"  Max absolute difference:  {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    print(f"  Tolerance:                {atol}")

    # Also compare softmax probabilities
    pt_probs = np.exp(pt_output) / np.exp(pt_output).sum(axis=-1, keepdims=True)
    ort_probs = np.exp(ort_output) / np.exp(ort_output).sum(axis=-1, keepdims=True)
    prob_diff = float(np.max(np.abs(pt_probs - ort_probs)))
    print(f"  Max probability difference: {prob_diff:.8f}")

    match = max_diff < atol
    if match:
        print("  PASS: ONNX output matches PyTorch within tolerance.")
    else:
        print(f"  WARNING: Outputs differ by {max_diff:.6f} (> tolerance {atol})")
    return match


def benchmark_pytorch(model: torch.nn.Module, device: str = "cpu") -> dict:
    """Benchmark PyTorch inference latency."""
    dummy = torch.randn(*INPUT_SHAPE, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            model(dummy)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(NUM_BENCHMARK_RUNS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def benchmark_onnx(onnx_path: Path) -> dict:
    """Benchmark ONNX Runtime inference latency."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    inputs = {"input": dummy}

    # Warmup
    for _ in range(NUM_WARMUP):
        session.run(None, inputs)

    # Benchmark
    latencies = []
    for _ in range(NUM_BENCHMARK_RUNS):
        t0 = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def print_size_comparison(checkpoint_path: Path, onnx_path: Path) -> None:
    """Print model file size comparison."""
    pt_size = checkpoint_path.stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    ratio = onnx_size / pt_size if pt_size > 0 else 0

    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)
    print(f"  PyTorch checkpoint:  {pt_size:.1f} MB")
    print(f"  ONNX model:          {onnx_size:.1f} MB")
    print(f"  Size ratio (ONNX/PT): {ratio:.2f}x")
    print("=" * 60)


def print_benchmark_comparison(pt_stats: dict, ort_stats: dict) -> None:
    """Print latency benchmark comparison."""
    speedup = pt_stats["mean_ms"] / ort_stats["mean_ms"] if ort_stats["mean_ms"] > 0 else 0

    print("\n" + "=" * 60)
    print("INFERENCE LATENCY BENCHMARK")
    print(f"  ({NUM_BENCHMARK_RUNS} runs, {NUM_WARMUP} warmup)")
    print("=" * 60)
    print(f"  {'Metric':<12} {'PyTorch':>12} {'ONNX RT':>12} {'Speedup':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for key, label in [
        ("mean_ms", "Mean"),
        ("median_ms", "Median"),
        ("p95_ms", "P95"),
        ("min_ms", "Min"),
        ("max_ms", "Max"),
    ]:
        pt_val = pt_stats[key]
        ort_val = ort_stats[key]
        sp = pt_val / ort_val if ort_val > 0 else 0
        print(f"  {label:<12} {pt_val:>10.1f}ms {ort_val:>10.1f}ms {sp:>9.2f}x")
    print(f"\n  Overall speedup (mean): {speedup:.2f}x")
    print("=" * 60)


def main():
    print("=" * 60)
    print("NetraAI — DR Model ONNX Export")
    print("=" * 60)

    # ── Check checkpoint exists ────────────────────────────────────────
    if not CHECKPOINT_PATH.exists():
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    device = "cpu"
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {ONNX_OUTPUT_PATH}")

    # ── Step 1: Load PyTorch model ─────────────────────────────────────
    print("\n[1/5] Loading PyTorch model ...")
    model = load_dr_model(CHECKPOINT_PATH, device=device)
    print("  Model loaded successfully.")

    # ── Step 2: Export to ONNX ─────────────────────────────────────────
    print("\n[2/5] Exporting to ONNX ...")
    export_to_onnx(model, ONNX_OUTPUT_PATH, device=device)

    # ── Step 3: Validate ONNX ─────────────────────────────────────────
    print("\n[3/5] Validating ONNX model ...")
    validate_onnx(ONNX_OUTPUT_PATH)

    # ── Step 4: Verify output equivalence ─────────────────────────────
    print("\n[4/5] Verifying output equivalence ...")
    verify_outputs(model, ONNX_OUTPUT_PATH, device=device)

    # ── Step 5: Benchmark ─────────────────────────────────────────────
    print("\n[5/5] Benchmarking inference speed ...")
    print("  Running PyTorch benchmark ...")
    pt_stats = benchmark_pytorch(model, device=device)
    print("  Running ONNX Runtime benchmark ...")
    ort_stats = benchmark_onnx(ONNX_OUTPUT_PATH)

    # ── Results ────────────────────────────────────────────────────────
    print_size_comparison(CHECKPOINT_PATH, ONNX_OUTPUT_PATH)
    print_benchmark_comparison(pt_stats, ort_stats)

    print(f"\nONNX model exported successfully to:\n  {ONNX_OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
