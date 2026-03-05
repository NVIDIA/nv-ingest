# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import ray
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Global constants for controlling Ray resource hueristic calculations when 
# the user does not specify a requested override.

# EMBEDDING Actor constants (PER-GPU)
EMBED_INITIAL_ACTORS = 1 # Hueristic initial num actors per GPU (initial_size of ActorPoolStrategy). Ray starts up this many actors on start-up.
EMBED_MIN_ACTORS = 1 # Hueristic minimum num actors per GPU (min_size of ActorPoolStrategy). Ray tries to never let running actors fall below this number.
EMBED_MAX_ACTORS = 8 # Hueristic baseline num actors per GPU (max_size of ActorPoolStrategy). Ray will grow to this size when resources are available.
EMBED_GPUS_PER_ACTOR = 0.5 # Hueristic baseline num GPUs per actor. Used to determine which GPU to schedule the actor on.
EMBED_BATCH_SIZE = 256 # Ray batch size AND EMBEDDING inference batch size

#Nemotron Parse Actor constants (PER-GPU)
NEMOTRON_PARSE_INITIAL_ACTORS = 1 # Hueristic initial num actors per GPU (initial_size of ActorPoolStrategy). Ray starts up this many actors on start-up.
NEMOTRON_PARSE_MIN_ACTORS = 1 # Hueristic minimum num actors per GPU (min_size of ActorPoolStrategy). Ray tries to never let running actors fall below this number.
NEMOTRON_PARSE_MAX_ACTORS = 4 # Hueristic baseline num actors per GPU (max_size of ActorPoolStrategy). Ray will grow to this size when resources are available.
NEMOTRON_PARSE_GPUS_PER_ACTOR = 0.1 # Hueristic baseline num GPUs per actor. Used to determine which GPU to schedule the actor on.
NEMOTRON_PARSE_BATCH_SIZE = 64 # Ray batch size AND Nemotron Parse inference batch size

# OCR Actor constants (PER-GPU)
OCR_INITIAL_ACTORS = 1 # Hueristic initial num actors per GPU (initial_size of ActorPoolStrategy). Ray starts up this many actors on start-up.
OCR_MIN_ACTORS = 1 # Hueristic minimum num actors per GPU (min_size of ActorPoolStrategy). Ray tries to never let running actors fall below this number.
OCR_MAX_ACTORS = 6 # Hueristic baseline num actors per GPU (max_size of ActorPoolStrategy). Ray will grow to this size when resources are available.
OCR_GPUS_PER_ACTOR = 0.1 # Hueristic baseline num GPUs per actor. Used to determine which GPU to schedule the actor on.
OCR_BATCH_SIZE = 64 # Ray batch size AND OCR inference batch size

# PAGE-ELEMENTS Actor constants (PER-GPU)
PAGE_ELEMENTS_INITIAL_ACTORS = 1 # Hueristic initial num actors per GPU (initial_size of ActorPoolStrategy). Ray starts up this many actors on start-up.
PAGE_ELEMENTS_MIN_ACTORS = 1 # Hueristic minimum num actors per GPU (min_size of ActorPoolStrategy). Ray tries to never let running actors fall below this number.
PAGE_ELEMENTS_MAX_ACTORS = 6 # Hueristic baseline num actors per GPU (max_size of ActorPoolStrategy). Ray will grow to this size when resources are available.
PAGE_ELEMENTS_GPUS_PER_ACTOR = 0.1 # Hueristic baseline num GPUs per actor. Used to determine which GPU to schedule the actor on.
PAGE_ELEMENTS_BATCH_SIZE = 64 # Ray batch size AND PAGE-ELEMENTS inference batch size

# PDF EXTRACTOR constants (PER-GPU) - reason being more GPU means more CPU needed to feed the models and keep up
PDF_EXTRACT_BATCH_SIZE = 8 # Ray batch size AND PDF extraction batch size
PDF_EXTRACT_CPUS_PER_TASK = 2.0 # Hueristic baseline num CPUs per task. Used to determine which CPU to schedule the task on.
PDF_EXTRACT_TASKS = 12 # Hueristic baseline num tasks. Used to determine how many CPU tasks to run in parallel.


# def _read_env_int(name: str, default: int, *, minimum: int = 0) -> int:
#     """Read an integer environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = int(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)


# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)
# def _read_env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
#     """Read a float environment variable with fallback and lower bound."""
#     raw = os.getenv(name)
#     if raw is None:
#         return default
#     try:
#         value = float(raw.strip())
#     except (TypeError, ValueError):
#         return default
#     return max(minimum, value)

# def _debug_print(message: str) -> None:
#     print(f"[resource_heuristics] {message}")


# def _as_int(value: object, *, minimum: int) -> Optional[int]:
#     try:
#         return max(minimum, int(value)) if value is not None else None
#     except (TypeError, ValueError):
#         return None


class GpuInfo(BaseModel):
    driver_version: str
    gpu_name: str
    gpu_uuid: str
    gpu_brand: str
    total_mib: int
    used_mib: int
    free_mib: int


class NodeGpuInfo(BaseModel):
    gpus: dict[int, GpuInfo]


@ray.remote
def _get_gpu_memory_info() -> NodeGpuInfo:
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
    )

    nvmlInit()
    driver_version = nvmlSystemGetDriverVersion()
    device_count = nvmlDeviceGetCount()

    gpu_info: dict[int, GpuInfo] = {}
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        gpu_uuid = nvmlDeviceGetUUID(handle)
        gpu_info[i] = GpuInfo(
            driver_version=(
                driver_version.decode("utf-8", errors="replace")
                if isinstance(driver_version, (bytes, bytearray))
                else str(driver_version)
            ),
            gpu_name=(
                gpu_name.decode("utf-8", errors="replace")
                if isinstance(gpu_name, (bytes, bytearray))
                else str(gpu_name)
            ),
            gpu_uuid=(
                gpu_uuid.decode("utf-8", errors="replace")
                if isinstance(gpu_uuid, (bytes, bytearray))
                else str(gpu_uuid)
            ),
            gpu_brand=str(nvmlDeviceGetBrand(handle)),
            total_mib=int(info.total // (1024**2)),
            used_mib=int(info.used // (1024**2)),
            free_mib=int(info.free // (1024**2)),
        )

    nvmlShutdown()
    return NodeGpuInfo(gpus=gpu_info)


# class _ResolvedHeuristicConfig(BaseModel):
#     model_config = ConfigDict(frozen=True)

#     page_elements_per_gpu: int
#     ocr_per_gpu: int
#     embed_per_gpu: int
#     cpu_only_stage_num_gpus: float
#     high_overlap_page_elements_num_gpus: float
#     high_overlap_ocr_num_gpus: float
#     high_overlap_embed_num_gpus: float
#     max_gpu_per_stage: float


# class ResourceResolutionDetails(BaseModel):
#     """Resolved CPU/GPU values and where each value came from."""

#     model_config = ConfigDict(frozen=True)

#     cpu_count: int
#     gpu_count: int
#     cpu_source: str
#     gpu_source: str
#     auto_source: str


class Resources(BaseModel):
    """Resources and where they came from."""

    model_config = ConfigDict(frozen=True)

    cpu_count: int
    gpu_count: int

    def __str__(self) -> str:
        return f"Resources(cpu_count={self.cpu_count}, gpu_count={self.gpu_count}, source={self.source})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.cpu_count, self.gpu_count, self.source))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resources):
            return False
        return self.cpu_count == other.cpu_count and self.gpu_count == other.gpu_count and self.source == other.source

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class ClusterResources(BaseModel):
    """Detected compute resources and where they came from."""

    model_config = ConfigDict(frozen=True)

    total_resources: Resources  # Total resources available to the cluster
    available_resources: Resources  # Available resources to the cluster (not in use currently)
    source: str = "Ray"

    def __str__(self) -> str:
        return f"ClusterResources(total_resources={self.total_resources}, available_resources={self.available_resources}, source={self.source})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.total_resources, self.available_resources, self.source))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterResources):
            return False
        return self.total_resources == other.total_resources and self.available_resources == other.available_resources and self.source == other.source

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


# class WorkerHeuristicResult(BaseModel):
#     """Resolved worker counts and per-stage GPU resource allocations."""

#     model_config = ConfigDict(frozen=True)

#     cpu_count: int
#     gpu_count: int
#     page_elements_per_gpu: int
#     ocr_per_gpu: int
#     embed_per_gpu: int
#     heuristic_page_elements_workers: int
#     heuristic_detect_workers: int
#     heuristic_embed_workers: int
#     page_elements_workers: int
#     detect_workers: int
#     embed_workers: int
#     page_elements_override: Optional[int]
#     detect_override: Optional[int]
#     embed_override: Optional[int]
#     cpu_only_stage_num_gpus: float
#     page_elements_num_gpus: float
#     detect_num_gpus: float
#     embed_num_gpus: float


# def _resolve_heuristic_config() -> _ResolvedHeuristicConfig:
#     """Resolve heuristic constants with precedence defaults -> env."""
#     return _ResolvedHeuristicConfig(
#         page_elements_per_gpu=_read_env_int(
#             "NEMO_RETRIEVER_BATCH_PAGE_ELEMENTS_PER_GPU",
#             PAGE_ELEMENTS_PER_GPU,
#             minimum=1,
#         ),
#         ocr_per_gpu=_read_env_int(
#             "NEMO_RETRIEVER_BATCH_OCR_PER_GPU",
#             OCR_PER_GPU,
#             minimum=1,
#         ),
#         embed_per_gpu=_read_env_int(
#             "NEMO_RETRIEVER_BATCH_EMBED_PER_GPU",
#             EMBED_PER_GPU,
#             minimum=1,
#         ),
#         cpu_only_stage_num_gpus=_read_env_float(
#             "NEMO_RETRIEVER_BATCH_CPU_ONLY_STAGE_NUM_GPUS",
#             CPU_ONLY_STAGE_NUM_GPUS,
#             minimum=0.0,
#         ),
#         high_overlap_page_elements_num_gpus=_read_env_float(
#             "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS",
#             HIGH_OVERLAP_PAGE_ELEMENTS_NUM_GPUS,
#             minimum=0.0,
#         ),
#         high_overlap_ocr_num_gpus=_read_env_float(
#             "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_OCR_NUM_GPUS",
#             HIGH_OVERLAP_OCR_NUM_GPUS,
#             minimum=0.0,
#         ),
#         high_overlap_embed_num_gpus=_read_env_float(
#             "NEMO_RETRIEVER_BATCH_HIGH_OVERLAP_EMBED_NUM_GPUS",
#             HIGH_OVERLAP_EMBED_NUM_GPUS,
#             minimum=0.0,
#         ),
#         max_gpu_per_stage=_read_env_float(
#             "NEMO_RETRIEVER_BATCH_MAX_GPU_PER_STAGE",
#             MAX_GPU_PER_STAGE,
#             minimum=0.0,
#         ),
#     )



def resolve_available_resources(ray: object) -> ClusterResources:
    """Resolve available CPU/GPU resources from a Ray cluster."""

    if not ray.is_initialized():
        raise ValueError("Ray is not initialized")

    total_resources: dict[str, object] = ray.cluster_resources()
    available_resources: dict[str, object] = ray.available_resources()

    return ClusterResources(
        total_resources=Resources(cpu_count=total_resources.get("CPU", 0), gpu_count=total_resources.get("GPU", 0)),
        available_resources=Resources(cpu_count=available_resources.get("CPU", 0), gpu_count=available_resources.get("GPU", 0)),
        source="ray"
    )


# def resolve_resource_details(
#     *,
#     ray_cluster_address: Optional[str] = None,
#     override_cpu_count: Optional[int] = None,
#     override_gpu_count: Optional[int] = None,
# ) -> ResourceResolutionDetails:
#     """Resolve CPU/GPU counts and annotate precedence source for each value."""
#     auto = _detect_auto_resources(ray_cluster_address=ray_cluster_address)

#     resolved_cpu_count = int(auto.cpu_count)
#     resolved_gpu_count = int(auto.gpu_count)
#     cpu_source = f"auto:{auto.source}"
#     gpu_source = f"auto:{auto.source}"

#     raw_env_cpu = os.getenv(ENV_BATCH_NUM_CPUS)
#     if raw_env_cpu is not None:
#         env_cpu = _as_int(raw_env_cpu, minimum=1)
#         if env_cpu is not None:
#             resolved_cpu_count = int(env_cpu)
#             cpu_source = "env"

#     raw_env_gpu = os.getenv(ENV_BATCH_NUM_GPUS)
#     if raw_env_gpu is not None:
#         env_gpu = _as_int(raw_env_gpu, minimum=0)
#         if env_gpu is not None:
#             resolved_gpu_count = int(env_gpu)
#             gpu_source = "env"

#     if override_cpu_count is not None:
#         resolved_cpu_count = max(1, int(override_cpu_count))
#         cpu_source = "arg"
#     if override_gpu_count is not None:
#         resolved_gpu_count = max(0, int(override_gpu_count))
#         gpu_source = "arg"

#     resolved = ResourceResolutionDetails(
#         cpu_count=resolved_cpu_count,
#         gpu_count=resolved_gpu_count,
#         cpu_source=cpu_source,
#         gpu_source=gpu_source,
#         auto_source=auto.source,
#     )
#     _debug_print(
#         "Resolved resources: "
#         f"cpu={resolved.cpu_count} (source={resolved.cpu_source}), "
#         f"gpu={resolved.gpu_count} (source={resolved.gpu_source})"
#     )
#     return resolved


# def resolve_effective_resources(
#     ray_cluster_address: Optional[str] = None,
#     *,
#     override_cpu_count: Optional[int] = None,
#     override_gpu_count: Optional[int] = None,
# ) -> ClusterResources:
#     """Return resources with precedence autodetect -> env -> args."""
#     details = resolve_resource_details(
#         ray_cluster_address=ray_cluster_address,
#         override_cpu_count=override_cpu_count,
#         override_gpu_count=override_gpu_count,
#     )
#     return ClusterResources(cpu_count=details.cpu_count, gpu_count=details.gpu_count, source=details.auto_source)


# def resolve_batch_worker_plan(
#     *,
#     override_cpu_count: Optional[int] = None,
#     override_gpu_count: Optional[int] = None,
#     override_page_elements_actors: Optional[int] = None,
#     override_ocr_actors: Optional[int] = None,
#     override_embed_actors: Optional[int] = None,
#     concurrent_gpu_stage_count: Optional[int] = None,
#     ray_cluster_address: Optional[str] = None,
# ) -> WorkerHeuristicResult:
#     """Resolve worker counts and stage GPU defaults for batch ingest stages."""
#     cfg = _resolve_heuristic_config()
#     if override_cpu_count is None or override_gpu_count is None:
#         detected = resolve_effective_resources(
#             ray_cluster_address=ray_cluster_address,
#             override_cpu_count=override_cpu_count,
#             override_gpu_count=override_gpu_count,
#         )
#         cpu_count = int(override_cpu_count if override_cpu_count is not None else detected.cpu_count)
#         gpu_count = int(override_gpu_count if override_gpu_count is not None else detected.gpu_count)
#     else:
#         cpu_count = int(override_cpu_count)
#         gpu_count = int(override_gpu_count)

#     cpu_count = max(1, cpu_count)
#     gpu_count = max(0, gpu_count)

#     if gpu_count > 0:
#         heuristic_page_elements_workers = max(1, gpu_count * cfg.page_elements_per_gpu)
#         heuristic_detect_workers = max(1, gpu_count * cfg.ocr_per_gpu)
#         heuristic_embed_workers = max(1, gpu_count * cfg.embed_per_gpu)
#     else:
#         heuristic_page_elements_workers = 1
#         heuristic_detect_workers = 1
#         heuristic_embed_workers = 1

#     final_page_elements_workers = (
#         int(override_page_elements_actors)
#         if override_page_elements_actors is not None
#         else heuristic_page_elements_workers
#     )
#     final_detect_workers = int(override_ocr_actors) if override_ocr_actors is not None else heuristic_detect_workers
#     final_embed_workers = int(override_embed_actors) if override_embed_actors is not None else heuristic_embed_workers

#     effective_gpu_stage_count = max(1, int(concurrent_gpu_stage_count if concurrent_gpu_stage_count is not None else 1))
#     if gpu_count >= 2 and effective_gpu_stage_count == 3:
#         page_elements_num_gpus = float(cfg.high_overlap_page_elements_num_gpus)
#         detect_num_gpus = float(cfg.high_overlap_ocr_num_gpus)
#         embed_num_gpus = float(cfg.high_overlap_embed_num_gpus)
#     else:
#         gpu_per_stage = min(float(cfg.max_gpu_per_stage), float(gpu_count) / float(effective_gpu_stage_count))
#         page_elements_num_gpus = float(max(0.0, gpu_per_stage))
#         detect_num_gpus = float(max(0.0, gpu_per_stage))
#         embed_num_gpus = float(max(0.0, gpu_per_stage))

#     result = WorkerHeuristicResult(
#         cpu_count=cpu_count,
#         gpu_count=gpu_count,
#         page_elements_per_gpu=cfg.page_elements_per_gpu,
#         ocr_per_gpu=cfg.ocr_per_gpu,
#         embed_per_gpu=cfg.embed_per_gpu,
#         heuristic_page_elements_workers=heuristic_page_elements_workers,
#         heuristic_detect_workers=heuristic_detect_workers,
#         heuristic_embed_workers=heuristic_embed_workers,
#         page_elements_workers=max(1, final_page_elements_workers),
#         detect_workers=max(1, final_detect_workers),
#         embed_workers=max(1, final_embed_workers),
#         page_elements_override=override_page_elements_actors,
#         detect_override=override_ocr_actors,
#         embed_override=override_embed_actors,
#         cpu_only_stage_num_gpus=float(cfg.cpu_only_stage_num_gpus),
#         page_elements_num_gpus=page_elements_num_gpus,
#         detect_num_gpus=detect_num_gpus,
#         embed_num_gpus=embed_num_gpus,
#     )

#     return result


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
