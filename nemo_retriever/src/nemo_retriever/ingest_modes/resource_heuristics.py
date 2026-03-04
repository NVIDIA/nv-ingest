# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ray

"""Helpers for inferring batch-stage worker and GPU allocations.

This module centralizes small, environment-overridable heuristics used by the
batch ingestor to derive:

- worker pool sizes from available CPU/GPU resources, and
- per-stage logical GPU requests for Ray Data actor stages.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml


def _read_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read an integer environment variable with fallback and lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a float environment variable with fallback and lower bound."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


CPU_THRESHOLD_WORKERS = 64

HIGH_CPU_PAGE_ELEMENTS_PER_GPU = 4
HIGH_CPU_OCR_PER_GPU = 4
HIGH_CPU_EMBED_PER_GPU = 2

LOW_CPU_PAGE_ELEMENTS_PER_GPU = 3
LOW_CPU_OCR_PER_GPU = 3
LOW_CPU_EMBED_PER_GPU = 3

CPU_ONLY_STAGE_NUM_GPUS = 0.0
HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS = 0.5
HIGH_OVERLAP_OCR_NUM_GPUS = 1.0
HIGH_OVERLAP_EMBED_NUM_GPUS = 0.5
MAX_GPU_PER_STAGE = 1.0

ENV_BATCH_NUM_CPUS = "NEMO_RETRIEVER_BATCH_NUM_CPUS"
ENV_BATCH_NUM_GPUS = "NEMO_RETRIEVER_BATCH_NUM_GPUS"

# Default per-model VRAM ceiling (bytes) used for future scheduling heuristics.
DEFAULT_MODEL_MAX_VRAM_BYTES = 3 * 1024 * 1024 * 1024  # 3GB
MODEL_MAX_VRAM_BYTES: dict[tuple[str, int], int] = {
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 1): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 4): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 8): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 16): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 32): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 62): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nvidia/llama-3.2-nv-embedqa-1b-v2", 128): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 1): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 4): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 8): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 16): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 32): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 62): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-page-elements-v3", 128): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 1): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 4): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 8): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 16): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 32): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 62): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-ocr-v1", 128): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 1): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 4): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 8): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 16): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 32): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 62): DEFAULT_MODEL_MAX_VRAM_BYTES,
    ("nemotron-parse-v1.2", 128): DEFAULT_MODEL_MAX_VRAM_BYTES,
}


@ray.remote
def _get_gpu_memory_info() -> dict[int, int]:
    """Get the memory information for each GPU."""
    from pynvml import (
        nvmlInit,
        nvmlSystemGetDriverVersion,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlShutdown,
        nvmlDeviceGetName,
        nvmlDeviceGetUUID,
        nvmlDeviceGetBrand,
        nvmlDeviceGetProductBrand,
        nvmlDeviceGetProductSeries,
        nvmlDeviceGetProductArchitecture,
        nvmlDeviceGetProductGeneration,
        nvmlDeviceGetProductRevision,
        nvmlDeviceGetProductVersion,
        nvmlDeviceGetProductVersionMajor,
        nvmlDeviceGetProductVersionMinor,
    )

    # Initialize the NVML library
    nvmlInit()

    driver_version = nvmlSystemGetDriverVersion()
    print(f"Driver Version: {driver_version}")

    # Get the number of available GPUs
    device_count = nvmlDeviceGetCount()

    print(f"Found {device_count} GPU(s).")

    gpus_info = {}

    # Iterate over each GPU to get memory information
    for i in range(device_count):
        # Get a handle to the device
        handle = nvmlDeviceGetHandleByIndex(i)

        # Get memory information (total, free, used) in bytes
        info = nvmlDeviceGetMemoryInfo(handle)

        gpu_name = nvmlDeviceGetName(handle)
        gpu_uuid = nvmlDeviceGetUUID(handle)
        gpu_brand = nvmlDeviceGetBrand(handle)
        # gpu_product_name = nvmlDeviceGetProductName(handle)
        gpu_product_brand = nvmlDeviceGetProductBrand(handle)
        gpu_product_series = nvmlDeviceGetProductSeries(handle)
        gpu_product_architecture = nvmlDeviceGetProductArchitecture(handle)
        gpu_product_generation = nvmlDeviceGetProductGeneration(handle)
        gpu_product_revision = nvmlDeviceGetProductRevision(handle)
        gpu_product_version = nvmlDeviceGetProductVersion(handle)
        gpu_product_version_major = nvmlDeviceGetProductVersionMajor(handle)
        gpu_product_version_minor = nvmlDeviceGetProductVersionMinor(handle)

        print(f"\n--- GPU {i} ---")
        # Convert bytes to MiB for better readability (1 MiB = 1024*1024 bytes)
        print(f"Total memory: {info.total // (1024**2)} MiB")
        print(f"Used memory: {info.used // (1024**2)} MiB")
        print(f"Free memory: {info.free // (1024**2)} MiB")

        gpus_info[i] = {
            "drvier_version": driver_version,
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "gpu_brand": gpu_brand,
            "gpu_product_brand": gpu_product_brand,
            "gpu_product_series": gpu_product_series,
            "gpu_product_architecture": gpu_product_architecture,
            "gpu_product_generation": gpu_product_generation,
            "gpu_product_revision": gpu_product_revision,
            "gpu_product_version": gpu_product_version,
            "gpu_product_version_major": gpu_product_version_major,
            "gpu_product_version_minor": gpu_product_version_minor,
            "total": info.total // (1024**2),
            "used": info.used // (1024**2),
            "free": info.free // (1024**2),
        }

    # Shutdown the NVML library when finished
    nvmlShutdown()

    return gpus_info


def _debug_print(message: str) -> None:
    """Emit a lightweight resource-resolution debug message."""
    print(f"[resource_heuristics] {message}")


def _default_config_path() -> Path:
    """Return the default resource heuristic config path."""
    return Path.home() / ".nemo-retriever" / "config.yaml"


def _coerce_path(filepath: str | os.PathLike[str] | None) -> Path:
    return Path(filepath).expanduser() if filepath is not None else _default_config_path()


def _load_config(filepath: str | os.PathLike[str] | None = None) -> dict:
    """Load YAML config, returning an empty mapping when absent/invalid."""
    path = _coerce_path(filepath)
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _cfg_get(cfg: dict, *keys: str) -> object:
    cur: object = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _as_int(value: object, *, minimum: int) -> Optional[int]:
    try:
        return max(minimum, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value: object, *, minimum: float) -> Optional[float]:
    try:
        return max(minimum, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _ResolvedHeuristicConfig:
    cpu_threshold_workers: int
    high_cpu_page_elements_per_gpu: int
    high_cpu_ocr_per_gpu: int
    high_cpu_embed_per_gpu: int
    low_cpu_page_elements_per_gpu: int
    low_cpu_ocr_per_gpu: int
    low_cpu_embed_per_gpu: int
    cpu_only_stage_num_gpus: float
    high_overlap_page_elements_num_gpus: float
    high_overlap_ocr_num_gpus: float
    high_overlap_embed_num_gpus: float
    max_gpu_per_stage: float


@dataclass(frozen=True)
class ResourceResolutionDetails:
    """Resolved CPU/GPU values and where each value came from."""

    cpu_count: int
    gpu_count: int
    cpu_source: str
    gpu_source: str
    auto_source: str
    config_path: Optional[str]


def _resolve_heuristic_config(filepath: str | os.PathLike[str] | None = None) -> _ResolvedHeuristicConfig:
    """Resolve heuristic constants with precedence defaults -> config -> env."""
    cfg = _load_config(filepath)

    cpu_threshold_workers = _as_int(_cfg_get(cfg, "heuristics", "cpu_threshold_workers"), minimum=1)
    high_cpu_page_elements_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "high_cpu_page_elements_per_gpu"), minimum=1)
    high_cpu_ocr_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "high_cpu_ocr_per_gpu"), minimum=1)
    high_cpu_embed_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "high_cpu_embed_per_gpu"), minimum=1)
    low_cpu_page_elements_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "low_cpu_page_elements_per_gpu"), minimum=1)
    low_cpu_ocr_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "low_cpu_ocr_per_gpu"), minimum=1)
    low_cpu_embed_per_gpu = _as_int(_cfg_get(cfg, "heuristics", "low_cpu_embed_per_gpu"), minimum=1)

    cpu_only_stage_num_gpus = _as_float(_cfg_get(cfg, "heuristics", "cpu_only_stage_num_gpus"), minimum=0.0)
    high_overlap_page_elements_num_gpus = _as_float(
        _cfg_get(cfg, "heuristics", "high_overlap_page_elements_num_gpus"), minimum=0.0
    )
    high_overlap_ocr_num_gpus = _as_float(_cfg_get(cfg, "heuristics", "high_overlap_ocr_num_gpus"), minimum=0.0)
    high_overlap_embed_num_gpus = _as_float(_cfg_get(cfg, "heuristics", "high_overlap_embed_num_gpus"), minimum=0.0)
    max_gpu_per_stage = _as_float(_cfg_get(cfg, "heuristics", "max_gpu_per_stage"), minimum=0.0)

    return _ResolvedHeuristicConfig(
        cpu_threshold_workers=_read_env_int(
            "NEMO_RETRIEVER_BATCH_CPU_THRESHOLD_WORKERS",
            cpu_threshold_workers if cpu_threshold_workers is not None else CPU_THRESHOLD_WORKERS,
            minimum=1,
        ),
        high_cpu_page_elements_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_HIGH_CPU_PAGE_ELEMENTS_PER_GPU",
            (
                high_cpu_page_elements_per_gpu
                if high_cpu_page_elements_per_gpu is not None
                else HIGH_CPU_PAGE_ELEMENTS_PER_GPU
            ),
            minimum=1,
        ),
        high_cpu_ocr_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_HIGH_CPU_OCR_PER_GPU",
            high_cpu_ocr_per_gpu if high_cpu_ocr_per_gpu is not None else HIGH_CPU_OCR_PER_GPU,
            minimum=1,
        ),
        high_cpu_embed_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_HIGH_CPU_EMBED_PER_GPU",
            high_cpu_embed_per_gpu if high_cpu_embed_per_gpu is not None else HIGH_CPU_EMBED_PER_GPU,
            minimum=1,
        ),
        low_cpu_page_elements_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_LOW_CPU_PAGE_ELEMENTS_PER_GPU",
            (
                low_cpu_page_elements_per_gpu
                if low_cpu_page_elements_per_gpu is not None
                else LOW_CPU_PAGE_ELEMENTS_PER_GPU
            ),
            minimum=1,
        ),
        low_cpu_ocr_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_LOW_CPU_OCR_PER_GPU",
            low_cpu_ocr_per_gpu if low_cpu_ocr_per_gpu is not None else LOW_CPU_OCR_PER_GPU,
            minimum=1,
        ),
        low_cpu_embed_per_gpu=_read_env_int(
            "NEMO_RETRIEVER_BATCH_LOW_CPU_EMBED_PER_GPU",
            low_cpu_embed_per_gpu if low_cpu_embed_per_gpu is not None else LOW_CPU_EMBED_PER_GPU,
            minimum=1,
        ),
        cpu_only_stage_num_gpus=_read_env_float(
            "NEMO_RETRIEVER_BATCH_CPU_ONLY_STAGE_NUM_GPUS",
            cpu_only_stage_num_gpus if cpu_only_stage_num_gpus is not None else CPU_ONLY_STAGE_NUM_GPUS,
            minimum=0.0,
        ),
        high_overlap_page_elements_num_gpus=_read_env_float(
            "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS",
            (
                high_overlap_page_elements_num_gpus
                if high_overlap_page_elements_num_gpus is not None
                else HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS
            ),
            minimum=0.0,
        ),
        high_overlap_ocr_num_gpus=_read_env_float(
            "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_OCR_NUM_GPUS",
            high_overlap_ocr_num_gpus if high_overlap_ocr_num_gpus is not None else HIGH_OVERLAP_OCR_NUM_GPUS,
            minimum=0.0,
        ),
        high_overlap_embed_num_gpus=_read_env_float(
            "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_EMBED_NUM_GPUS",
            high_overlap_embed_num_gpus if high_overlap_embed_num_gpus is not None else HIGH_OVERLAP_EMBED_NUM_GPUS,
            minimum=0.0,
        ),
        max_gpu_per_stage=_read_env_float(
            "NEMO_RETRIEVER_BATCH_MAX_GPU_PER_STAGE",
            max_gpu_per_stage if max_gpu_per_stage is not None else MAX_GPU_PER_STAGE,
            minimum=0.0,
        ),
    )


def _detect_auto_resources(ray_cluster_address: Optional[str] = None) -> SystemResources:
    """Detect base CPU/GPU resources from Ray (if available) or local machine."""
    local_cpu_count = int(os.cpu_count() or 1)
    local_gpu_count = int(_detect_local_gpu_count())
    source = "local"

    try:
        import ray
    except Exception:
        _debug_print(
            "Auto-detect: Ray unavailable; using local resources " f"(cpu={local_cpu_count}, gpu={local_gpu_count})."
        )
        return SystemResources(cpu_count=local_cpu_count, gpu_count=local_gpu_count, source=source)

    detected_cpu_count = local_cpu_count
    detected_gpu_count = local_gpu_count
    try:
        if ray.is_initialized():
            resources = ray.cluster_resources() or ray.available_resources()
            detected_cpu_count = int(resources.get("CPU", local_cpu_count))
            detected_gpu_count = int(resources.get("GPU", local_gpu_count))
            source = "ray"
        elif ray_cluster_address:
            ray.init(address=ray_cluster_address, ignore_reinit_error=True, log_to_driver=False)
            try:
                resources = ray.cluster_resources() or ray.available_resources()
                detected_cpu_count = int(resources.get("CPU", local_cpu_count))
                detected_gpu_count = int(resources.get("GPU", local_gpu_count))
                source = "ray"
            finally:
                ray.shutdown()
    except Exception:
        pass

    detected = SystemResources(
        cpu_count=max(1, detected_cpu_count),
        gpu_count=max(0, detected_gpu_count),
        source=source,
    )
    _debug_print(
        "Auto-detect: "
        f"source={detected.source}, cpu={detected.cpu_count}, gpu={detected.gpu_count}, "
        f"ray_cluster_address={ray_cluster_address!r}"
    )
    return detected


def resolve_resource_details(
    *,
    ray_cluster_address: Optional[str] = None,
    resource_config_path: str | os.PathLike[str] | None = None,
    override_cpu_count: Optional[int] = None,
    override_gpu_count: Optional[int] = None,
) -> ResourceResolutionDetails:
    """Resolve CPU/GPU counts and annotate precedence source for each value."""
    auto = _detect_auto_resources(ray_cluster_address=ray_cluster_address)
    cfg = _load_config(resource_config_path)
    cfg_path = _coerce_path(resource_config_path)
    cfg_exists = cfg_path.exists()
    cfg_cpu_count = _as_int(_cfg_get(cfg, "resources", "cpu_count"), minimum=1)
    cfg_gpu_count = _as_int(_cfg_get(cfg, "resources", "gpu_count"), minimum=0)

    resolved_cpu_count = int(auto.cpu_count)
    resolved_gpu_count = int(auto.gpu_count)
    cpu_source = f"auto:{auto.source}"
    gpu_source = f"auto:{auto.source}"

    if cfg_cpu_count is not None:
        resolved_cpu_count = int(cfg_cpu_count)
        cpu_source = "config"
    if cfg_gpu_count is not None:
        resolved_gpu_count = int(cfg_gpu_count)
        gpu_source = "config"

    raw_env_cpu = os.getenv(ENV_BATCH_NUM_CPUS)
    if raw_env_cpu is not None:
        env_cpu = _as_int(raw_env_cpu, minimum=1)
        if env_cpu is not None:
            resolved_cpu_count = int(env_cpu)
            cpu_source = "env"

    raw_env_gpu = os.getenv(ENV_BATCH_NUM_GPUS)
    if raw_env_gpu is not None:
        env_gpu = _as_int(raw_env_gpu, minimum=0)
        if env_gpu is not None:
            resolved_gpu_count = int(env_gpu)
            gpu_source = "env"

    if override_cpu_count is not None:
        resolved_cpu_count = max(1, int(override_cpu_count))
        cpu_source = "arg"
    if override_gpu_count is not None:
        resolved_gpu_count = max(0, int(override_gpu_count))
        gpu_source = "arg"

    resolved = ResourceResolutionDetails(
        cpu_count=resolved_cpu_count,
        gpu_count=resolved_gpu_count,
        cpu_source=cpu_source,
        gpu_source=gpu_source,
        auto_source=auto.source,
        config_path=str(cfg_path) if cfg_exists else None,
    )
    _debug_print(
        "Resolved resources: "
        f"cpu={resolved.cpu_count} (source={resolved.cpu_source}), "
        f"gpu={resolved.gpu_count} (source={resolved.gpu_source}), "
        f"config_path={resolved.config_path or '(not found)'}"
    )
    if raw_env_cpu is not None or raw_env_gpu is not None:
        _debug_print(
            f"Environment overrides: {ENV_BATCH_NUM_CPUS}={raw_env_cpu!r}, " f"{ENV_BATCH_NUM_GPUS}={raw_env_gpu!r}"
        )
    if override_cpu_count is not None or override_gpu_count is not None:
        _debug_print(
            "Argument overrides: "
            f"override_cpu_count={override_cpu_count!r}, override_gpu_count={override_gpu_count!r}"
        )
    return resolved


@dataclass(frozen=True)
class SystemResources:
    """Detected compute resources and where they came from."""

    cpu_count: int
    gpu_count: int
    source: str


@dataclass(frozen=True)
class WorkerHeuristicResult:
    """Resolved worker counts and per-stage GPU resource allocations."""

    cpu_count: int
    gpu_count: int
    profile_name: str
    page_elements_per_gpu: int
    ocr_per_gpu: int
    embed_per_gpu: int
    heuristic_page_elements_workers: int
    heuristic_detect_workers: int
    heuristic_embed_workers: int
    page_elements_workers: int
    detect_workers: int
    embed_workers: int
    page_elements_override: Optional[int]
    detect_override: Optional[int]
    embed_override: Optional[int]
    cpu_only_stage_num_gpus: float
    page_elements_num_gpus: float
    detect_num_gpus: float
    embed_num_gpus: float
    cpu_threshold_workers: int


def _detect_local_gpu_count() -> int:
    """Detect local GPU count from torch first, then CUDA_VISIBLE_DEVICES."""
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        pass

    cuda_visible_devices = (os.getenv("CUDA_VISIBLE_DEVICES") or "").strip()
    if not cuda_visible_devices or cuda_visible_devices in {"-1", "none", "None"}:
        return 0

    return len([device for device in cuda_visible_devices.split(",") if device.strip()])


def resolve_effective_resources(
    ray_cluster_address: Optional[str] = None,
    *,
    resource_config_path: str | os.PathLike[str] | None = None,
    override_cpu_count: Optional[int] = None,
    override_gpu_count: Optional[int] = None,
) -> SystemResources:
    """Return resources with precedence autodetect -> config file -> env/args."""
    details = resolve_resource_details(
        ray_cluster_address=ray_cluster_address,
        resource_config_path=resource_config_path,
        override_cpu_count=override_cpu_count,
        override_gpu_count=override_gpu_count,
    )
    return SystemResources(cpu_count=details.cpu_count, gpu_count=details.gpu_count, source=details.auto_source)


def resolve_batch_worker_plan(
    *,
    override_cpu_count: Optional[int] = None,
    override_gpu_count: Optional[int] = None,
    override_page_elements_actors: Optional[int] = None,
    override_ocr_actors: Optional[int] = None,
    override_embed_actors: Optional[int] = None,
    concurrent_gpu_stage_count: Optional[int] = None,
    resource_config_path: str | os.PathLike[str] | None = None,
    ray_cluster_address: Optional[str] = None,
) -> WorkerHeuristicResult:
    """Resolve worker counts and stage GPU defaults for batch ingest stages.

    Explicit worker overrides take precedence over heuristic values.
    """
    cfg = _resolve_heuristic_config(resource_config_path)
    if override_cpu_count is None or override_gpu_count is None:
        detected = resolve_effective_resources(
            ray_cluster_address=ray_cluster_address,
            resource_config_path=resource_config_path,
            override_cpu_count=override_cpu_count,
            override_gpu_count=override_gpu_count,
        )
        cpu_count = int(override_cpu_count if override_cpu_count is not None else detected.cpu_count)
        gpu_count = int(override_gpu_count if override_gpu_count is not None else detected.gpu_count)
    else:
        cpu_count = int(override_cpu_count)
        gpu_count = int(override_gpu_count)

    cpu_count = max(1, cpu_count)
    gpu_count = max(0, gpu_count)

    if cpu_count >= cfg.cpu_threshold_workers:
        profile_name = "high_cpu"
        page_elements_per_gpu = cfg.high_cpu_page_elements_per_gpu
        ocr_per_gpu = cfg.high_cpu_ocr_per_gpu
        embed_per_gpu = cfg.high_cpu_embed_per_gpu
    else:
        profile_name = "low_cpu"
        page_elements_per_gpu = cfg.low_cpu_page_elements_per_gpu
        ocr_per_gpu = cfg.low_cpu_ocr_per_gpu
        embed_per_gpu = cfg.low_cpu_embed_per_gpu

    if gpu_count > 0:
        heuristic_page_elements_workers = max(1, gpu_count * page_elements_per_gpu)
        heuristic_detect_workers = max(1, gpu_count * ocr_per_gpu)
        heuristic_embed_workers = max(1, gpu_count * embed_per_gpu)
    else:
        heuristic_page_elements_workers = 1
        heuristic_detect_workers = 1
        heuristic_embed_workers = 1

    final_page_elements_workers = (
        int(override_page_elements_actors)
        if override_page_elements_actors is not None
        else heuristic_page_elements_workers
    )
    final_detect_workers = int(override_ocr_actors) if override_ocr_actors is not None else heuristic_detect_workers
    final_embed_workers = int(override_embed_actors) if override_embed_actors is not None else heuristic_embed_workers

    effective_gpu_stage_count = max(1, int(concurrent_gpu_stage_count if concurrent_gpu_stage_count is not None else 1))
    if gpu_count >= 2 and effective_gpu_stage_count == 3:
        page_elements_num_gpus = float(cfg.high_overlap_page_elements_num_gpus)
        detect_num_gpus = float(cfg.high_overlap_ocr_num_gpus)
        embed_num_gpus = float(cfg.high_overlap_embed_num_gpus)
    else:
        gpu_per_stage = min(float(cfg.max_gpu_per_stage), float(gpu_count) / float(effective_gpu_stage_count))
        page_elements_num_gpus = float(max(0.0, gpu_per_stage))
        detect_num_gpus = float(max(0.0, gpu_per_stage))
        embed_num_gpus = float(max(0.0, gpu_per_stage))

    result = WorkerHeuristicResult(
        cpu_count=cpu_count,
        gpu_count=gpu_count,
        profile_name=profile_name,
        page_elements_per_gpu=page_elements_per_gpu,
        ocr_per_gpu=ocr_per_gpu,
        embed_per_gpu=embed_per_gpu,
        heuristic_page_elements_workers=heuristic_page_elements_workers,
        heuristic_detect_workers=heuristic_detect_workers,
        heuristic_embed_workers=heuristic_embed_workers,
        page_elements_workers=max(1, final_page_elements_workers),
        detect_workers=max(1, final_detect_workers),
        embed_workers=max(1, final_embed_workers),
        page_elements_override=override_page_elements_actors,
        detect_override=override_ocr_actors,
        embed_override=override_embed_actors,
        cpu_only_stage_num_gpus=float(cfg.cpu_only_stage_num_gpus),
        page_elements_num_gpus=page_elements_num_gpus,
        detect_num_gpus=detect_num_gpus,
        embed_num_gpus=embed_num_gpus,
        cpu_threshold_workers=cfg.cpu_threshold_workers,
    )
    _debug_print(
        "Worker plan: "
        f"profile={result.profile_name}, cpu={result.cpu_count}, gpu={result.gpu_count}, "
        f"workers(page_elements={result.page_elements_workers}, "
        "detect={result.detect_workers}, embed={result.embed_workers}), "
        f"gpu_per_stage(page_elements={result.page_elements_num_gpus:.3f}, "
        f"detect={result.detect_num_gpus:.3f}, embed={result.embed_num_gpus:.3f})"
    )
    return result


def format_worker_heuristic_summary(
    result: WorkerHeuristicResult,
    *,
    final_page_elements_workers: Optional[int] = None,
    final_detect_workers: Optional[int] = None,
    final_embed_workers: Optional[int] = None,
) -> str:
    """Build a human-readable summary of resolved worker/GPU allocations."""
    effective_page_elements = (
        int(final_page_elements_workers) if final_page_elements_workers is not None else result.page_elements_workers
    )
    effective_detect = int(final_detect_workers) if final_detect_workers is not None else result.detect_workers
    effective_embed = int(final_embed_workers) if final_embed_workers is not None else result.embed_workers

    return "\n".join(
        [
            "Batch worker heuristic configuration:",
            f"  resources: cpu={result.cpu_count}, gpu={result.gpu_count}",
            f"  profile: {result.profile_name} (threshold={result.cpu_threshold_workers})",
            (
                "  per_gpu_ratio: "
                f"PageElementDetectionActor={result.page_elements_per_gpu}, "
                f"OCRActor={result.ocr_per_gpu}, "
                f"_BatchEmbedActor={result.embed_per_gpu}"
            ),
            (
                "  heuristic_workers: "
                f"PageElementDetectionActor={result.heuristic_page_elements_workers}, "
                f"OCRActor={result.heuristic_detect_workers}, "
                f"_BatchEmbedActor={result.heuristic_embed_workers}"
            ),
            (
                "  explicit_overrides: "
                f"PageElementDetectionActor={result.page_elements_override}, "
                f"OCRActor={result.detect_override}, "
                f"_BatchEmbedActor={result.embed_override}"
            ),
            (
                "  final_workers: "
                f"PageElementDetectionActor={effective_page_elements}, "
                f"OCRActor={effective_detect}, "
                f"_BatchEmbedActor={effective_embed}"
            ),
            (
                "  gpu_stage_allocation: "
                f"cpu_only={result.cpu_only_stage_num_gpus}, "
                f"PageElementDetectionActor={result.page_elements_num_gpus}, "
                f"OCRActor={result.detect_num_gpus}, "
                f"_BatchEmbedActor={result.embed_num_gpus}"
            ),
        ]
    )


def pretty_print_worker_heuristic_summary(
    result: WorkerHeuristicResult,
    *,
    final_page_elements_workers: Optional[int] = None,
    final_detect_workers: Optional[int] = None,
    final_embed_workers: Optional[int] = None,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Print the worker heuristic summary using the provided print function."""
    print_fn(
        format_worker_heuristic_summary(
            result,
            final_page_elements_workers=final_page_elements_workers,
            final_detect_workers=final_detect_workers,
            final_embed_workers=final_embed_workers,
        )
    )


def freeze_resource_config(
    output_path: str | os.PathLike[str] | None = None,
    *,
    ray_cluster_address: Optional[str] = None,
) -> Path:
    """Persist the current resolved resource/heuristic configuration to YAML.

    By default this writes to ``$HOME/.nemo-retriever/config.yaml``.
    """
    target = _coerce_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    settings = _resolve_heuristic_config(target)
    resources = resolve_effective_resources(ray_cluster_address=ray_cluster_address, resource_config_path=target)

    payload = {
        "resources": {
            "cpu_count": int(resources.cpu_count),
            "gpu_count": int(resources.gpu_count),
        },
        "heuristics": {
            "cpu_threshold_workers": int(settings.cpu_threshold_workers),
            "high_cpu_page_elements_per_gpu": int(settings.high_cpu_page_elements_per_gpu),
            "high_cpu_ocr_per_gpu": int(settings.high_cpu_ocr_per_gpu),
            "high_cpu_embed_per_gpu": int(settings.high_cpu_embed_per_gpu),
            "low_cpu_page_elements_per_gpu": int(settings.low_cpu_page_elements_per_gpu),
            "low_cpu_ocr_per_gpu": int(settings.low_cpu_ocr_per_gpu),
            "low_cpu_embed_per_gpu": int(settings.low_cpu_embed_per_gpu),
            "cpu_only_stage_num_gpus": float(settings.cpu_only_stage_num_gpus),
            "high_overlap_page_elements_num_gpus": float(settings.high_overlap_page_elements_num_gpus),
            "high_overlap_ocr_num_gpus": float(settings.high_overlap_ocr_num_gpus),
            "high_overlap_embed_num_gpus": float(settings.high_overlap_embed_num_gpus),
            "max_gpu_per_stage": float(settings.max_gpu_per_stage),
        },
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target
