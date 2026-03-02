from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional


def _read_env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


CPU_THRESHOLD_WORKERS = _read_env_int("NEMO_RETRIEVER_BATCH_CPU_THRESHOLD_WORKERS", 64, minimum=1)

HIGH_CPU_PAGE_ELEMENTS_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_HIGH_CPU_PAGE_ELEMENTS_PER_GPU", 4, minimum=1)
HIGH_CPU_OCR_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_HIGH_CPU_OCR_PER_GPU", 4, minimum=1)
HIGH_CPU_EMBED_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_HIGH_CPU_EMBED_PER_GPU", 2, minimum=1)

LOW_CPU_PAGE_ELEMENTS_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_LOW_CPU_PAGE_ELEMENTS_PER_GPU", 3, minimum=1)
LOW_CPU_OCR_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_LOW_CPU_OCR_PER_GPU", 3, minimum=1)
LOW_CPU_EMBED_PER_GPU = _read_env_int("NEMO_RETRIEVER_BATCH_LOW_CPU_EMBED_PER_GPU", 3, minimum=1)


@dataclass(frozen=True)
class SystemResources:
    cpu_count: int
    gpu_count: int
    source: str


@dataclass(frozen=True)
class WorkerHeuristicResult:
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


def _detect_local_gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        pass

    cuda_visible_devices = (os.getenv("CUDA_VISIBLE_DEVICES") or "").strip()
    if not cuda_visible_devices or cuda_visible_devices in {"-1", "none", "None"}:
        return 0

    return len([device for device in cuda_visible_devices.split(",") if device.strip()])


def get_cluster_or_local_resources(ray_address: Optional[str] = None) -> SystemResources:
    local_cpu_count = int(os.cpu_count() or 1)
    local_gpu_count = int(_detect_local_gpu_count())

    try:
        import ray
    except Exception:
        return SystemResources(cpu_count=local_cpu_count, gpu_count=local_gpu_count, source="local")

    try:
        if ray.is_initialized():
            resources = ray.cluster_resources() or ray.available_resources()
            cpu_count = int(resources.get("CPU", local_cpu_count))
            gpu_count = int(resources.get("GPU", local_gpu_count))
            return SystemResources(cpu_count=max(1, cpu_count), gpu_count=max(0, gpu_count), source="ray")

        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
            try:
                resources = ray.cluster_resources() or ray.available_resources()
                cpu_count = int(resources.get("CPU", local_cpu_count))
                gpu_count = int(resources.get("GPU", local_gpu_count))
                return SystemResources(cpu_count=max(1, cpu_count), gpu_count=max(0, gpu_count), source="ray")
            finally:
                ray.shutdown()
    except Exception:
        pass

    return SystemResources(cpu_count=local_cpu_count, gpu_count=local_gpu_count, source="local")


def resolve_worker_heuristic(
    *,
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    page_elements_workers: Optional[int] = None,
    detect_workers: Optional[int] = None,
    embed_workers: Optional[int] = None,
) -> WorkerHeuristicResult:
    if num_cpus is None or num_gpus is None:
        detected = get_cluster_or_local_resources()
        cpu_count = int(num_cpus if num_cpus is not None else detected.cpu_count)
        gpu_count = int(num_gpus if num_gpus is not None else detected.gpu_count)
    else:
        cpu_count = int(num_cpus)
        gpu_count = int(num_gpus)

    cpu_count = max(1, cpu_count)
    gpu_count = max(0, gpu_count)

    if cpu_count >= CPU_THRESHOLD_WORKERS:
        profile_name = "high_cpu"
        page_elements_per_gpu = HIGH_CPU_PAGE_ELEMENTS_PER_GPU
        ocr_per_gpu = HIGH_CPU_OCR_PER_GPU
        embed_per_gpu = HIGH_CPU_EMBED_PER_GPU
    else:
        profile_name = "low_cpu"
        page_elements_per_gpu = LOW_CPU_PAGE_ELEMENTS_PER_GPU
        ocr_per_gpu = LOW_CPU_OCR_PER_GPU
        embed_per_gpu = LOW_CPU_EMBED_PER_GPU

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
    )


def format_worker_heuristic_summary(
    result: WorkerHeuristicResult,
    *,
    final_page_elements_workers: Optional[int] = None,
    final_detect_workers: Optional[int] = None,
    final_embed_workers: Optional[int] = None,
) -> str:
    effective_page_elements = (
        int(final_page_elements_workers) if final_page_elements_workers is not None else result.page_elements_workers
    )
    effective_detect = int(final_detect_workers) if final_detect_workers is not None else result.detect_workers
    effective_embed = int(final_embed_workers) if final_embed_workers is not None else result.embed_workers

    return "\n".join(
        [
            "Batch worker heuristic configuration:",
            f"  resources: cpu={result.cpu_count}, gpu={result.gpu_count}",
            f"  profile: {result.profile_name} (threshold={CPU_THRESHOLD_WORKERS})",
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
    print_fn(
        format_worker_heuristic_summary(
            result,
            final_page_elements_workers=final_page_elements_workers,
            final_detect_workers=final_detect_workers,
            final_embed_workers=final_embed_workers,
        )
    )
