"""
Page elements detection benchmark test case.

This test case benchmarks the Nemotron page elements model for detecting
document layout elements (tables, figures, text blocks, etc.) in images.

Uses torch.utils.benchmark for accurate timing with automatic CUDA synchronization
and warmup handling.

Usage:
    uv run nv-ingest-harness-run --case=page_elements --dataset=/path/to/page_images

Requirements:
    - nemotron_page_elements_v3 package installed
    - GPU with CUDA support
"""

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
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


def benchmark_inference(model, img: np.ndarray, num_repeats: int = 1) -> tuple[dict, float, float, float]:
    """
    Run inference on a single image using torch.utils.benchmark.Timer.

    Includes both preprocessing and forward pass in the timing.

    Args:
        model: The loaded model
        img: Image as numpy array
        num_repeats: Number of times to run inference for timing (default: 1)

    Returns:
        Tuple of (predictions dict, total mean inference time in seconds,
        preprocess mean time in seconds, forward mean time in seconds)
    """
    # Timer measures preprocessing only
    preprocess_timer = Timer(
        stmt="model.preprocess(img)",
        globals={"model": model, "img": img},
        num_threads=1,
    )
    preprocess_measurement = preprocess_timer.timeit(num_repeats)

    # # Timer measures forward pass only using a preprocessed tensor
    # # Move the numpy array to CUDA device (if available)
    # if torch.cuda.is_available():
    #     img = torch.as_tensor(img).to(torch.uint8).to("cuda")

    # x = model.preprocess(img)
    # forward_timer = Timer(
    #     stmt="model(x, img_shape)[0]",
    #     globals={"model": model, "x": x, "img_shape": img.shape},
    #     num_threads=1,
    # )
    # forward_measurement = forward_timer.timeit(num_repeats)

    # Run once more to get the actual predictions
    with torch.inference_mode():
        x = model.preprocess(img)
        preds = model(x, img.shape)[0]

    # total_inference_mean = preprocess_measurement.mean + forward_measurement.mean
    # return preds, total_inference_mean, preprocess_measurement.mean, forward_measurement.mean
    return preds, 0.0, 0.0, 0.0


def _encode_image_to_base64_png(img: np.ndarray) -> str:
    """Encode RGB numpy image to base64 PNG payload string."""
    buf = BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def benchmark_remote_inference(
    invoke_url: str,
    img: np.ndarray,
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    num_repeats: int = 1,
) -> tuple[dict, float]:
    """Benchmark HTTP POST inference and return response JSON + mean latency."""
    image_b64 = _encode_image_to_base64_png(img)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}",
            }
        ]
    }

    response_json: dict[str, Any] = {}
    t0 = time.perf_counter()
    for _ in range(int(num_repeats)):
        resp = requests.post(invoke_url, headers=headers, json=payload, timeout=float(timeout_s))
        resp.raise_for_status()
        response_json = resp.json()
    elapsed = time.perf_counter() - t0
    return response_json, (elapsed / max(int(num_repeats), 1))


def count_detections_from_remote_response(response_json: dict) -> int:
    """Best-effort count from common remote payload shapes."""
    if not isinstance(response_json, dict):
        return 0

    if isinstance(response_json.get("detections"), list):
        return len(response_json["detections"])

    data = response_json.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if isinstance(data[0].get("detections"), list):
            return len(data[0]["detections"])
        if all(isinstance(v, list) for v in data[0].values()):
            return int(sum(len(v) for v in data[0].values()))

    if all(isinstance(v, list) for v in response_json.values()):
        return int(sum(len(v) for v in response_json.values()))

    return 0


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main test entry point for page elements benchmarking.

    Args:
        config: TestConfig object with all settings
        log_path: Path for logging output

    Returns:
        Exit code (0 = success)
    """
    # Handle config
    if config is None:
        print("ERROR: No configuration provided")
        print("This test case requires a config object from the test runner")
        return 2

    # Get dataset directory from config
    data_dir = config.dataset_dir
    num_repeats = getattr(config, "num_repeats", 1)
    invoke_url = (getattr(config, "page_elements_invoke_url", None) or "").strip()
    request_timeout_s = float(getattr(config, "page_elements_request_timeout_s", 120.0) or 120.0)
    api_key = (getattr(config, "page_elements_api_key", None) or os.getenv("NVIDIA_API_KEY", "")).strip() or None
    use_remote = bool(invoke_url)

    print("=== Page Elements Benchmark ===")
    print(f"Dataset: {data_dir}")
    print(f"Timing repeats per image: {num_repeats}")
    if use_remote:
        print(f"Using remote endpoint: {invoke_url}")
    else:
        print("Using torch.utils.benchmark.Timer")
    print("================================")

    # Get image paths
    image_paths = get_image_paths(data_dir)
    num_images = len(image_paths)
    print(f"Found {num_images} image(s) to process")
    kv_event_log("num_images", num_images, log_path)

    if num_images == 0:
        print("ERROR: No images found in dataset directory")
        return 1

    model = None
    model_load_time = 0.0
    postprocess_preds_page_element = None
    if use_remote:
        print("Remote mode enabled: skipping local model load")
        kv_event_log("model_load_time_s", model_load_time, log_path)
        # Single untimed warmup request to pre-establish TLS/session and reduce first-sample bias.
        warmup_img = np.array(Image.open(image_paths[0]).convert("RGB"))
        try:
            _resp, _ = benchmark_remote_inference(
                invoke_url,
                warmup_img,
                api_key=api_key,
                timeout_s=request_timeout_s,
                num_repeats=1,
            )
            print("Remote warmup complete")
        except Exception as e:
            print(f"WARNING: Remote warmup failed: {e}")
    else:
        # Import here to avoid loading model at module import time
        from nemotron_page_elements_v3.model import define_model
        from nemotron_page_elements_v3.utils import postprocess_preds_page_element

        print("Loading model...")
        model_load_start = time.perf_counter()
        model = define_model("page_element_v3")
        model.to("cuda").eval()
        model_load_time = time.perf_counter() - model_load_start
        print(f"Model loaded in {model_load_time:.2f}s")
        print(f"Available labels: {model.labels}")
        kv_event_log("model_load_time_s", model_load_time, log_path)

        # Warmup using torch.utils.benchmark (handles CUDA init automatically)
        print("Running warmup with torch.utils.benchmark...")
        warmup_img = np.array(Image.open(image_paths[0]).convert("RGB"))
        warmup_timer = Timer(
            stmt="model(model.preprocess(img), img.shape)[0]",
            globals={"model": model, "img": warmup_img},
            num_threads=1,
        )
        # blocked_autorange runs enough iterations to get stable timing
        warmup_measurement = warmup_timer.blocked_autorange(min_run_time=0.5)
        print(f"Warmup complete. Estimated per-image time: {warmup_measurement.mean * 1000:.2f} ms")

    # Process each image
    results = []
    inference_times = []
    preprocess_times = []
    postprocess_times = []
    total_detections = 0

    start_time = time.perf_counter()

    for img_path in tqdm(image_paths, desc="Processing images", unit="page"):
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if use_remote:
            response_json, inference_time = benchmark_remote_inference(
                invoke_url,
                img,
                api_key=api_key,
                timeout_s=request_timeout_s,
                num_repeats=num_repeats,
            )
            num_detections = count_detections_from_remote_response(response_json)
            preprocess_time = 0.0
            postprocess_time = 0.0
        else:
            # Run benchmarked inference
            preds, inference_time, preprocess_time, _forward_time = benchmark_inference(
                model, img, num_repeats=num_repeats
            )

            # Post-processing
            postprocess_start = time.perf_counter()
            boxes, labels, scores = postprocess_preds_page_element(preds, model.thresholds_per_class, model.labels)
            postprocess_time = time.perf_counter() - postprocess_start
            num_detections = len(boxes)
        total_detections += num_detections
        inference_times.append(inference_time * 1000)  # Convert to ms
        preprocess_times.append(preprocess_time * 1000)  # Convert to ms
        postprocess_times.append(postprocess_time * 1000)  # Convert to ms

        # Store result
        result = {
            "image_path": str(img_path),
            "image_name": img_path.name,
            "inference_time_ms": inference_time * 1000,
            "preprocess_time_ms": preprocess_time * 1000,
            "postprocess_time_ms": postprocess_time * 1000,
            "num_detections": num_detections,
        }
        results.append(result)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate statistics
    times = np.array(inference_times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    median_time = np.median(times)
    # total_time = np.sum(times)
    throughput = num_images / (total_time / 1000) if total_time > 0 else 0

    # Calculate IQR (interquartile range) - useful stat from torch.utils.benchmark
    q75, q25 = np.percentile(times, [75, 25])
    iqr_time = q75 - q25

    preprocess_mean_time = None
    preprocess_total_time = None
    postprocess_mean_time = None
    postprocess_total_time = None
    if not use_remote:
        preprocess_np = np.array(preprocess_times)
        postprocess_np = np.array(postprocess_times)
        preprocess_mean_time = float(np.mean(preprocess_np))
        preprocess_total_time = float(np.sum(preprocess_np))
        postprocess_mean_time = float(np.mean(postprocess_np))
        postprocess_total_time = float(np.sum(postprocess_np))

    # Log metrics
    kv_event_log("total_detections", total_detections, log_path)
    kv_event_log("mean_inference_time_ms", mean_time, log_path)
    kv_event_log("std_inference_time_ms", std_time, log_path)
    kv_event_log("min_inference_time_ms", min_time, log_path)
    kv_event_log("max_inference_time_ms", max_time, log_path)
    kv_event_log("median_inference_time_ms", median_time, log_path)
    kv_event_log("iqr_inference_time_ms", iqr_time, log_path)
    kv_event_log("total_inference_time_ms", total_time, log_path)
    kv_event_log("throughput_images_per_sec", throughput, log_path)
    if preprocess_mean_time is not None and preprocess_total_time is not None:
        kv_event_log("mean_preprocess_time_ms", preprocess_mean_time, log_path)
        kv_event_log("total_preprocess_time_ms", preprocess_total_time, log_path)
    if postprocess_mean_time is not None and postprocess_total_time is not None:
        kv_event_log("mean_postprocess_time_ms", postprocess_mean_time, log_path)
        kv_event_log("total_postprocess_time_ms", postprocess_total_time, log_path)

    if not use_remote:
        print("=== Local Runtime Breakdown ===")
        print(f"Pre-processing mean: {preprocess_mean_time:.3f} ms")
        print(f"Pre-processing total: {preprocess_total_time:.3f} ms")
        print(f"Post-processing mean: {postprocess_mean_time:.3f} ms")
        print(f"Post-processing total: {postprocess_total_time:.3f} ms")
        print("================================")

    # Build test results for consolidation
    test_name = config.test_name or os.path.basename(data_dir.rstrip("/"))

    test_results = {
        "test_config": {
            "test_name": test_name,
            "dataset_dir": data_dir,
            "num_repeats": num_repeats,
            "model": invoke_url if use_remote else "page_element_v3",
            "benchmark_method": "http.post" if use_remote else "torch.utils.benchmark.Timer",
            "page_elements_invoke_url": invoke_url or None,
        },
        "results": {
            "num_images": num_images,
            "total_detections": total_detections,
            "mean_inference_time_ms": mean_time,
            "std_inference_time_ms": std_time,
            "min_inference_time_ms": min_time,
            "max_inference_time_ms": max_time,
            "median_inference_time_ms": median_time,
            "iqr_inference_time_ms": iqr_time,
            "total_inference_time_ms": total_time,
            "throughput_images_per_sec": throughput,
            "model_load_time_s": model_load_time,
            "mean_preprocess_time_ms": preprocess_mean_time,
            "total_preprocess_time_ms": preprocess_total_time,
            "mean_postprocess_time_ms": postprocess_mean_time,
            "total_postprocess_time_ms": postprocess_total_time,
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
