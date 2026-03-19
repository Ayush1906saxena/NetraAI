#!/usr/bin/env python3
"""
Benchmark inference latency for DR grading and IQA models.

Measures end-to-end latency including preprocessing, model inference,
and postprocessing across different devices, batch sizes, and model formats.

Usage:
    python -m ml.scripts.benchmark_latency \
        --checkpoint weights/dr_grader_v1.pt \
        --device mps \
        --batch-sizes 1 4 8 16 32

    python -m ml.scripts.benchmark_latency \
        --onnx exports/dr_grader.onnx \
        --num-runs 200
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def benchmark_pytorch(
    model: torch.nn.Module,
    device: str,
    img_size: int = 224,
    batch_sizes: list[int] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    include_preprocessing: bool = True,
) -> dict:
    """
    Benchmark PyTorch model inference latency.

    Args:
        model: Model in eval mode.
        device: 'cpu', 'cuda', or 'mps'.
        img_size: Input image size.
        batch_sizes: List of batch sizes to test.
        num_warmup: Warmup iterations.
        num_runs: Benchmark iterations per batch size.
        include_preprocessing: If True, include FundusPreprocessor in the benchmark.

    Returns:
        Dict with latency stats for each batch size.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    model = model.to(device).eval()
    results = {}

    # Optional preprocessing benchmark
    preprocess_time_ms = 0.0
    if include_preprocessing:
        from ml.data.preprocess import FundusPreprocessor

        preprocessor = FundusPreprocessor()
        # Simulate a 2048x1536 fundus image (typical smartphone camera)
        fake_image = np.random.randint(0, 255, (1536, 2048, 3), dtype=np.uint8)

        pre_latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            preprocessor.process(fake_image, target_size=img_size)
            elapsed = (time.perf_counter() - start) * 1000
            pre_latencies.append(elapsed)

        preprocess_time_ms = statistics.median(pre_latencies)
        results["preprocessing"] = {
            "mean_ms": round(statistics.mean(pre_latencies), 2),
            "median_ms": round(preprocess_time_ms, 2),
            "std_ms": round(statistics.stdev(pre_latencies), 2),
            "p95_ms": round(np.percentile(pre_latencies, 95), 2),
        }
        print(f"Preprocessing: {preprocess_time_ms:.1f}ms median")

    for bs in batch_sizes:
        print(f"\nBatch size {bs}:")
        dummy = torch.randn(bs, 3, img_size, img_size, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy)

        # Synchronize device before benchmarking
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()

                start = time.perf_counter()
                _ = model(dummy)

                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()

                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)

        latencies_arr = np.array(latencies)
        per_image = latencies_arr / bs

        stats = {
            "batch_size": bs,
            "total_mean_ms": round(float(latencies_arr.mean()), 2),
            "total_median_ms": round(float(np.median(latencies_arr)), 2),
            "total_std_ms": round(float(latencies_arr.std()), 2),
            "total_p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
            "total_p99_ms": round(float(np.percentile(latencies_arr, 99)), 2),
            "per_image_mean_ms": round(float(per_image.mean()), 2),
            "per_image_median_ms": round(float(np.median(per_image)), 2),
            "throughput_img_per_sec": round(float(bs * 1000 / latencies_arr.mean()), 1),
        }

        # End-to-end (including preprocessing)
        if include_preprocessing:
            e2e = stats["per_image_median_ms"] + preprocess_time_ms
            stats["e2e_per_image_median_ms"] = round(e2e, 2)

        results[f"batch_{bs}"] = stats

        print(f"  Total: {stats['total_median_ms']:.1f}ms median")
        print(f"  Per image: {stats['per_image_median_ms']:.1f}ms")
        print(f"  Throughput: {stats['throughput_img_per_sec']:.1f} img/s")

    return results


def benchmark_onnx_runtime(
    onnx_path: str,
    img_size: int = 224,
    batch_sizes: list[int] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Benchmark ONNX Runtime inference."""
    import onnxruntime as ort

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    # Configure session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    providers = ort.get_available_providers()
    print(f"ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

    results = {}

    for bs in batch_sizes:
        print(f"\nONNX Batch size {bs}:")
        dummy = np.random.randn(bs, 3, img_size, img_size).astype(np.float32)

        # Warmup
        for _ in range(num_warmup):
            session.run(None, {"input": dummy})

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {"input": dummy})
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        arr = np.array(latencies)
        per_image = arr / bs

        stats = {
            "batch_size": bs,
            "total_median_ms": round(float(np.median(arr)), 2),
            "per_image_median_ms": round(float(np.median(per_image)), 2),
            "throughput_img_per_sec": round(float(bs * 1000 / arr.mean()), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
        }
        results[f"batch_{bs}"] = stats

        print(f"  Total: {stats['total_median_ms']:.1f}ms")
        print(f"  Per image: {stats['per_image_median_ms']:.1f}ms")
        print(f"  Throughput: {stats['throughput_img_per_sec']:.1f} img/s")

    return results


def benchmark_memory(
    model: torch.nn.Module,
    device: str,
    img_size: int = 224,
    batch_size: int = 1,
) -> dict:
    """Measure peak memory usage during inference."""
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        current_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        return {
            "peak_memory_mb": round(peak_mb, 1),
            "current_memory_mb": round(current_mb, 1),
            "device": device,
        }
    elif device == "mps":
        # MPS doesn't have detailed memory tracking; estimate from model params
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_mb = param_bytes / (1024 * 1024)
        return {
            "param_memory_mb": round(param_mb, 1),
            "estimated_inference_mb": round(param_mb * 2.5, 1),
            "device": device,
        }
    else:
        import os
        import psutil

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        with torch.no_grad():
            _ = model(dummy)
        mem_after = process.memory_info().rss / (1024 * 1024)
        return {
            "memory_before_mb": round(mem_before, 1),
            "memory_after_mb": round(mem_after, 1),
            "inference_delta_mb": round(mem_after - mem_before, 1),
            "device": device,
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference latency")
    parser.add_argument("--checkpoint", help="PyTorch checkpoint path")
    parser.add_argument("--onnx", help="ONNX model path")
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32],
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--output", help="Save results JSON to this path")
    parser.add_argument(
        "--no-preprocessing", action="store_true",
        help="Exclude preprocessing from the benchmark",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        parser.error("At least one of --checkpoint or --onnx is required")

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

    all_results = {"device": device, "img_size": args.img_size, "num_runs": args.num_runs}

    # PyTorch benchmark
    if args.checkpoint:
        print("=" * 60)
        print(f"PYTORCH BENCHMARK (device={device})")
        print("=" * 60)

        from ml.evaluation.evaluate import load_model

        model = load_model(args.checkpoint, device=device, num_classes=args.num_classes)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

        pytorch_results = benchmark_pytorch(
            model,
            device=device,
            img_size=args.img_size,
            batch_sizes=args.batch_sizes,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            include_preprocessing=not args.no_preprocessing,
        )
        all_results["pytorch"] = pytorch_results
        all_results["model_params"] = {
            "total": total_params,
            "trainable": trainable_params,
        }

        # Memory benchmark
        print("\n--- Memory Usage ---")
        mem_info = benchmark_memory(model, device, args.img_size)
        all_results["memory"] = mem_info
        for k, v in mem_info.items():
            print(f"  {k}: {v}")

    # ONNX benchmark
    if args.onnx:
        print("\n" + "=" * 60)
        print("ONNX RUNTIME BENCHMARK")
        print("=" * 60)

        try:
            onnx_results = benchmark_onnx_runtime(
                args.onnx,
                img_size=args.img_size,
                batch_sizes=args.batch_sizes,
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
            all_results["onnx"] = onnx_results
        except ImportError:
            print("onnxruntime not installed; skipping ONNX benchmark.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (batch_size=1, per-image median)")
    print("=" * 60)

    if "pytorch" in all_results and "batch_1" in all_results["pytorch"]:
        pt = all_results["pytorch"]["batch_1"]
        print(f"  PyTorch ({device}): {pt['per_image_median_ms']:.1f}ms "
              f"({pt['throughput_img_per_sec']:.0f} img/s)")

    if "onnx" in all_results and "batch_1" in all_results["onnx"]:
        ort_r = all_results["onnx"]["batch_1"]
        print(f"  ONNX Runtime: {ort_r['per_image_median_ms']:.1f}ms "
              f"({ort_r['throughput_img_per_sec']:.0f} img/s)")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
