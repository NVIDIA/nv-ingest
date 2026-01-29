"""
Nemotron OCR benchmark test case.

This test case benchmarks the Nemotron OCR model for optical character recognition
on document images.

Uses torch.utils.benchmark for accurate timing with automatic CUDA synchronization
and warmup handling.

Usage:
     uv run nv-ingest-harness-run --case=nemotron_ocr --dataset=/path/to/images

Requirements:
    - nemotron_ocr package installed
    - GPU with CUDA support
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.benchmark import Timer
from tqdm import tqdm

from nv_ingest_harness.utils.interact import kv_event_log


def get_image_paths(input_path: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".webp")) -> list[Path]:
    """Get all image paths from a directory or return single file path."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    elif path.is_dir():
        image_paths = []
        for ext in extensions:
            image_paths.extend(path.rglob(f"*{ext}"))
        return sorted(image_paths)
    else:
        raise ValueError(f"Invalid path: {input_path}")


def benchmark_inference(ocr, img_path: str, num_repeats: int = 1) -> tuple[list, float]:
    """
    Run OCR inference on a single image using torch.utils.benchmark.Timer.
    
    Args:
        ocr: The loaded NemotronOCR model
        img_path: Path to the image file
        num_repeats: Number of times to run inference for timing (default: 1)
    
    Returns:
        Tuple of (predictions list, mean inference time in seconds)
    """
    # Timer measures the full OCR pipeline
    timer = Timer(
        stmt="ocr(img_path)",
        globals={"ocr": ocr, "img_path": img_path},
        num_threads=1,
    )
    
    # Run timed inference
    measurement = timer.timeit(num_repeats)
    
    # Run once more to get the actual predictions
    predictions = ocr(img_path)
    
    return predictions, measurement.mean


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main test entry point for Nemotron OCR benchmarking.

    Args:
        config: TestConfig object with all settings
        log_path: Path for logging output

    Returns:
        Exit code (0 = success)
    """
    # Import here to avoid loading model at module import time
    from nemotron_ocr.inference.pipeline import NemotronOCR

    # Handle config
    if config is None:
        print("ERROR: No configuration provided")
        print("This test case requires a config object from the test runner")
        return 2

    # Get dataset directory from config
    data_dir = config.dataset_dir
    num_repeats = getattr(config, "num_repeats", 1)

    print("=== Nemotron OCR Benchmark ===")
    print(f"Dataset: {data_dir}")
    print(f"Timing repeats per image: {num_repeats}")
    print("Using torch.utils.benchmark.Timer")
    print("===============================")

    # Get image paths
    image_paths = get_image_paths(data_dir)
    num_images = len(image_paths)
    print(f"Found {num_images} image(s) to process")
    kv_event_log("num_images", num_images, log_path)

    if num_images == 0:
        print("ERROR: No images found in dataset directory")
        return 1

    # Load model
    print("Loading OCR model...")
    model_load_start = time.perf_counter()
    ocr = NemotronOCR()
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")
    kv_event_log("model_load_time_s", model_load_time, log_path)

    # Warmup using torch.utils.benchmark (handles CUDA init automatically)
    print("Running warmup with torch.utils.benchmark...")
    warmup_path = str(image_paths[0])
    warmup_timer = Timer(
        stmt="ocr(img_path)",
        globals={"ocr": ocr, "img_path": warmup_path},
        num_threads=1,
    )
    # blocked_autorange runs enough iterations to get stable timing
    warmup_measurement = warmup_timer.blocked_autorange(min_run_time=0.5)
    print(f"Warmup complete. Estimated per-image time: {warmup_measurement.mean * 1000:.2f} ms")

    # Process each image
    results = []
    inference_times = []
    total_text_regions = 0

    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        # Run benchmarked inference
        predictions, inference_time = benchmark_inference(ocr, str(img_path), num_repeats=num_repeats)

        num_regions = len(predictions)
        total_text_regions += num_regions
        inference_times.append(inference_time * 1000)  # Convert to ms

        # Store result (without full predictions to keep results.json manageable)
        result = {
            "image_path": str(img_path),
            "image_name": img_path.name,
            "inference_time_ms": inference_time * 1000,
            "num_text_regions": num_regions,
        }
        results.append(result)

    # Calculate statistics
    times = np.array(inference_times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    total_time = np.sum(times)
    throughput = num_images / (total_time / 1000) if total_time > 0 else 0

    # Calculate IQR (interquartile range) - useful stat from torch.utils.benchmark
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    # Log metrics
    kv_event_log("total_text_regions", total_text_regions, log_path)
    kv_event_log("mean_inference_time_ms", mean_time, log_path)
    kv_event_log("std_inference_time_ms", std_time, log_path)
    kv_event_log("min_inference_time_ms", min_time, log_path)
    kv_event_log("max_inference_time_ms", max_time, log_path)
    kv_event_log("median_inference_time_ms", median_time, log_path)
    kv_event_log("iqr_inference_time_ms", iqr_time, log_path)
    kv_event_log("total_inference_time_ms", total_time, log_path)
    kv_event_log("throughput_images_per_sec", throughput, log_path)

    # Build test results for consolidation
    test_name = config.test_name or os.path.basename(data_dir.rstrip("/"))

    test_results = {
        "test_config": {
            "test_name": test_name,
            "dataset_dir": data_dir,
            "num_repeats": num_repeats,
            "model": "nemotron_ocr",
            "benchmark_method": "torch.utils.benchmark.Timer",
        },
        "results": {
            "num_images": num_images,
            "total_text_regions": total_text_regions,
            "mean_inference_time_ms": mean_time,
            "std_inference_time_ms": std_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "median_inference_time_ms": median_time,
            "iqr_inference_time_ms": iqr_time,
            "total_inference_time_ms": total_time,
            "throughput_images_per_sec": throughput,
            "model_load_time_s": model_load_time,
        },
        "per_image_results": results,
    }

    # Write test results
    os.makedirs(log_path, exist_ok=True)
    results_file = os.path.join(log_path, "_test_results.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
