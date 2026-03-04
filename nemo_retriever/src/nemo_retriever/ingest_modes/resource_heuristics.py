# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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


def _detect_auto_resources(ray_address: Optional[str] = None) -> SystemResources:
    """Detect base CPU/GPU resources from Ray (if available) or local machine."""
    local_cpu_count = int(os.cpu_count() or 1)
    local_gpu_count = int(_detect_local_gpu_count())
    source = "local"

    try:
        import ray
    except Exception:
        return SystemResources(cpu_count=local_cpu_count, gpu_count=local_gpu_count, source=source)

    detected_cpu_count = local_cpu_count
    detected_gpu_count = local_gpu_count
    try:
        if ray.is_initialized():
            resources = ray.cluster_resources() or ray.available_resources()
            detected_cpu_count = int(resources.get("CPU", local_cpu_count))
            detected_gpu_count = int(resources.get("GPU", local_gpu_count))
            source = "ray"
        elif ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
            try:
                resources = ray.cluster_resources() or ray.available_resources()
                detected_cpu_count = int(resources.get("CPU", local_cpu_count))
                detected_gpu_count = int(resources.get("GPU", local_gpu_count))
                source = "ray"
            finally:
                ray.shutdown()
    except Exception:
        pass

    return SystemResources(
        cpu_count=max(1, detected_cpu_count),
        gpu_count=max(0, detected_gpu_count),
        source=source,
    )


def resolve_resource_details(
    *,
    ray_address: Optional[str] = None,
    config_path: str | os.PathLike[str] | None = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
) -> ResourceResolutionDetails:
    """Resolve CPU/GPU counts and annotate precedence source for each value."""
    auto = _detect_auto_resources(ray_address=ray_address)
    cfg = _load_config(config_path)
    cfg_path = _coerce_path(config_path)
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

    if num_cpus is not None:
        resolved_cpu_count = max(1, int(num_cpus))
        cpu_source = "arg"
    if num_gpus is not None:
        resolved_gpu_count = max(0, int(num_gpus))
        gpu_source = "arg"

    return ResourceResolutionDetails(
        cpu_count=resolved_cpu_count,
        gpu_count=resolved_gpu_count,
        cpu_source=cpu_source,
        gpu_source=gpu_source,
        auto_source=auto.source,
        config_path=str(cfg_path) if cfg_exists else None,
    )


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


def get_cluster_or_local_resources(
    ray_address: Optional[str] = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
) -> SystemResources:
    """Return resources with precedence autodetect -> config file -> env/args."""
    details = resolve_resource_details(
        ray_address=ray_address,
        config_path=config_path,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    return SystemResources(cpu_count=details.cpu_count, gpu_count=details.gpu_count, source=details.auto_source)


def resolve_worker_heuristic(
    *,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    page_elements_workers: Optional[int] = None,
    detect_workers: Optional[int] = None,
    embed_workers: Optional[int] = None,
    gpu_stage_count: Optional[int] = None,
    config_path: str | os.PathLike[str] | None = None,
    ray_address: Optional[str] = None,
) -> WorkerHeuristicResult:
    """Resolve worker counts and stage GPU defaults for batch ingest stages.

    Explicit worker overrides take precedence over heuristic values.
    """
    cfg = _resolve_heuristic_config(config_path)
    if num_cpus is None or num_gpus is None:
        detected = get_cluster_or_local_resources(
            ray_address=ray_address,
            config_path=config_path,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )
        cpu_count = int(num_cpus if num_cpus is not None else detected.cpu_count)
        gpu_count = int(num_gpus if num_gpus is not None else detected.gpu_count)
    else:
        cpu_count = int(num_cpus)
        gpu_count = int(num_gpus)

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
        int(page_elements_workers) if page_elements_workers is not None else heuristic_page_elements_workers
    )
    final_detect_workers = int(detect_workers) if detect_workers is not None else heuristic_detect_workers
    final_embed_workers = int(embed_workers) if embed_workers is not None else heuristic_embed_workers

    effective_gpu_stage_count = max(1, int(gpu_stage_count if gpu_stage_count is not None else 1))
    if gpu_count >= 2 and effective_gpu_stage_count == 3:
        page_elements_num_gpus = float(cfg.high_overlap_page_elements_num_gpus)
        detect_num_gpus = float(cfg.high_overlap_ocr_num_gpus)
        embed_num_gpus = float(cfg.high_overlap_embed_num_gpus)
    else:
        gpu_per_stage = min(float(cfg.max_gpu_per_stage), float(gpu_count) / float(effective_gpu_stage_count))
        page_elements_num_gpus = float(max(0.0, gpu_per_stage))
        detect_num_gpus = float(max(0.0, gpu_per_stage))
        embed_num_gpus = float(max(0.0, gpu_per_stage))

    return WorkerHeuristicResult(
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
        page_elements_override=page_elements_workers,
        detect_override=detect_workers,
        embed_override=embed_workers,
        cpu_only_stage_num_gpus=float(cfg.cpu_only_stage_num_gpus),
        page_elements_num_gpus=page_elements_num_gpus,
        detect_num_gpus=detect_num_gpus,
        embed_num_gpus=embed_num_gpus,
        cpu_threshold_workers=cfg.cpu_threshold_workers,
    )


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


def freeze(filepath: str | os.PathLike[str] | None = None, *, ray_address: Optional[str] = None) -> Path:
    """Persist the current resolved resource/heuristic configuration to YAML.

    By default this writes to ``$HOME/.nemo-retriever/config.yaml``.
    """
    target = _coerce_path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)

    settings = _resolve_heuristic_config(target)
    resources = get_cluster_or_local_resources(ray_address=ray_address, config_path=target)

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
