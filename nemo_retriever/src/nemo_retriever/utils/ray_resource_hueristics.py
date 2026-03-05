# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional

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


class Resources(BaseModel):
    """Resources and where they came from."""

    model_config = ConfigDict(frozen=True)

    cpu_count: int
    gpu_count: int

    def __str__(self) -> str:
        return f"Resources(cpu_count={self.cpu_count}, gpu_count={self.gpu_count})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.cpu_count, self.gpu_count))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resources):
            return False
        return self.cpu_count == other.cpu_count and self.gpu_count == other.gpu_count

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class ClusterResources(BaseModel):
    """Detected compute resources and where they came from."""

    model_config = ConfigDict(frozen=True)

    total_resources: Resources  # Total resources available to the cluster
    available_resources: Resources  # Available resources to the cluster (not in use currently)

    def total_cpu_count(self) -> int:
        return self.total_resources.cpu_count
    
    def total_gpu_count(self) -> int:
        return self.total_resources.gpu_count
    
    def available_cpu_count(self) -> int:
        return self.available_resources.cpu_count
    
    def available_gpu_count(self) -> int:
        return self.available_resources.gpu_count

    def __str__(self) -> str:
        return f"ClusterResources(total_resources={self.total_resources}, available_resources={self.available_resources})"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.total_resources, self.available_resources))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterResources):
            return False
        return self.total_resources == other.total_resources and self.available_resources == other.available_resources

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


def gather_cluster_resources(ray: object) -> ClusterResources:
    """Gather total and available CPU/GPU resources from a Ray cluster."""

    if not ray.is_initialized():
        raise ValueError("Ray is not initialized")

    total_resources: dict[str, object] = ray.cluster_resources()
    available_resources: dict[str, object] = ray.available_resources()

    return ClusterResources(
        total_resources=Resources(cpu_count=total_resources.get("CPU", 0), gpu_count=total_resources.get("GPU", 0)),
        available_resources=Resources(cpu_count=available_resources.get("CPU", 0), gpu_count=available_resources.get("GPU", 0)),
    )


class RequestedPlan(BaseModel):
    """Contains the requested Ray DAG plan for the batch ingest."""

    model_config = ConfigDict(frozen=True)

    # Embedder resources requested to satisfy DAG plan
    embed_initial_actors: int
    embed_min_actors: int
    embed_max_actors: int
    embed_gpus_per_actor: float
    embed_batch_size: int

    # Nemotron Parse resources requested to satisfy DAG plan
    nemotron_parse_initial_actors: int
    nemotron_parse_min_actors: int
    nemotron_parse_max_actors: int
    nemotron_parse_gpus_per_actor: float
    nemotron_parse_batch_size: int

    # OCR resources requested to satisfy DAG plan
    ocr_initial_actors: int
    ocr_min_actors: int
    ocr_max_actors: int
    ocr_gpus_per_actor: float
    ocr_batch_size: int

    # Page Elements resources requested to satisfy DAG plan
    page_elements_initial_actors: int
    page_elements_min_actors: int
    page_elements_max_actors: int
    page_elements_gpus_per_actor: float
    page_elements_batch_size: int

    # PDF Extraction resources requested to satisfy DAG plan
    pdf_extract_batch_size: int
    pdf_extract_cpus_per_task: float
    pdf_extract_tasks: int

    def get_embed_initial_actors(self) -> int:
        return self.embed_initial_actors

    def get_embed_min_actors(self) -> int:
        return self.embed_min_actors

    def get_embed_max_actors(self) -> int:
        return self.embed_max_actors

    def get_embed_gpus_per_actor(self) -> float:
        return self.embed_gpus_per_actor

    def get_embed_batch_size(self) -> int:
        return self.embed_batch_size

    def get_nemotron_parse_initial_actors(self) -> int:
        return self.nemotron_parse_initial_actors

    def get_nemotron_parse_min_actors(self) -> int:
        return self.nemotron_parse_min_actors

    def get_nemotron_parse_max_actors(self) -> int:
        return self.nemotron_parse_max_actors

    def get_nemotron_parse_gpus_per_actor(self) -> float:
        return self.nemotron_parse_gpus_per_actor

    def get_nemotron_parse_batch_size(self) -> int:
        return self.nemotron_parse_batch_size

    def get_ocr_initial_actors(self) -> int:
        return self.ocr_initial_actors

    def get_ocr_min_actors(self) -> int:
        return self.ocr_min_actors

    def get_ocr_max_actors(self) -> int:
        return self.ocr_max_actors

    def get_ocr_gpus_per_actor(self) -> float:
        return self.ocr_gpus_per_actor

    def get_ocr_batch_size(self) -> int:
        return self.ocr_batch_size

    def get_page_elements_initial_actors(self) -> int:
        return self.page_elements_initial_actors

    def get_page_elements_min_actors(self) -> int:
        return self.page_elements_min_actors

    def get_page_elements_max_actors(self) -> int:
        return self.page_elements_max_actors

    def get_page_elements_gpus_per_actor(self) -> float:
        return self.page_elements_gpus_per_actor

    def get_page_elements_batch_size(self) -> int:
        return self.page_elements_batch_size

    def get_pdf_extract_batch_size(self) -> int:
        return self.pdf_extract_batch_size

    def get_pdf_extract_cpus_per_task(self) -> float:
        return self.pdf_extract_cpus_per_task

    def get_pdf_extract_tasks(self) -> int:
        return self.pdf_extract_tasks


    def __str__(self) -> str:
        return f"RequestedPlan(embed_initial_actors={self.embed_initial_actors}, embed_min_actors={self.embed_min_actors}, embed_max_actors={self.embed_max_actors}, embed_gpus_per_actor={self.embed_gpus_per_actor}, embed_batch_size={self.embed_batch_size}, nemotron_parse_initial_actors={self.nemotron_parse_initial_actors}, nemotron_parse_min_actors={self.nemotron_parse_min_actors}, nemotron_parse_max_actors={self.nemotron_parse_max_actors}, nemotron_parse_gpus_per_actor={self.nemotron_parse_gpus_per_actor}, nemotron_parse_batch_size={self.nemotron_parse_batch_size}, ocr_initial_actors={self.ocr_initial_actors}, ocr_min_actors={self.ocr_min_actors}, ocr_max_actors={self.ocr_max_actors}, ocr_gpus_per_actor={self.ocr_gpus_per_actor}, ocr_batch_size={self.ocr_batch_size}, page_elements_initial_actors={self.page_elements_initial_actors}, page_elements_min_actors={self.page_elements_min_actors}, page_elements_max_actors={self.page_elements_max_actors}, page_elements_gpus_per_actor={self.page_elements_gpus_per_actor}, page_elements_batch_size={self.page_elements_batch_size}, pdf_extract_batch_size={self.pdf_extract_batch_size}, pdf_extract_cpus_per_task={self.pdf_extract_cpus_per_task}, pdf_extract_tasks={self.pdf_extract_tasks})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.embed_initial_actors, self.embed_min_actors, self.embed_max_actors, self.embed_gpus_per_actor, self.embed_batch_size, self.nemotron_parse_initial_actors, self.nemotron_parse_min_actors, self.nemotron_parse_max_actors, self.nemotron_parse_gpus_per_actor, self.nemotron_parse_batch_size, self.ocr_initial_actors, self.ocr_min_actors, self.ocr_max_actors, self.ocr_gpus_per_actor, self.ocr_batch_size, self.page_elements_initial_actors, self.page_elements_min_actors, self.page_elements_max_actors, self.page_elements_gpus_per_actor, self.page_elements_batch_size, self.pdf_extract_batch_size, self.pdf_extract_cpus_per_task, self.pdf_extract_tasks))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RequestedPlan):
            return False
        return self.embed_initial_actors == other.embed_initial_actors and self.embed_min_actors == other.embed_min_actors and self.embed_max_actors == other.embed_max_actors and self.embed_gpus_per_actor == other.embed_gpus_per_actor and self.embed_batch_size == other.embed_batch_size and self.nemotron_parse_initial_actors == other.nemotron_parse_initial_actors and self.nemotron_parse_min_actors == other.nemotron_parse_min_actors and self.nemotron_parse_max_actors == other.nemotron_parse_max_actors and self.nemotron_parse_gpus_per_actor == other.nemotron_parse_gpus_per_actor and self.nemotron_parse_batch_size == other.nemotron_parse_batch_size and self.ocr_initial_actors == other.ocr_initial_actors and self.ocr_min_actors == other.ocr_min_actors and self.ocr_max_actors == other.ocr_max_actors and self.ocr_gpus_per_actor == other.ocr_gpus_per_actor and self.ocr_batch_size == other.ocr_batch_size and self.page_elements_initial_actors == other.page_elements_initial_actors and self.page_elements_min_actors == other.page_elements_min_actors and self.page_elements_max_actors == other.page_elements_max_actors and self.page_elements_gpus_per_actor == other.page_elements_gpus_per_actor and self.page_elements_batch_size == other.page_elements_batch_size and self.pdf_extract_batch_size == other.pdf_extract_batch_size and self.pdf_extract_cpus_per_task == other.pdf_extract_cpus_per_task and self.pdf_extract_tasks == other.pdf_extract_tasks

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


def resolve_requested_plan(
    *,
    cluster_resources: ClusterResources,
    override_embed_initial_actors: Optional[int] = None,
    override_embed_min_actors: Optional[int] = None,
    override_embed_max_actors: Optional[int] = None,
    override_embed_gpus_per_actor: Optional[float] = None,
    override_embed_batch_size: Optional[int] = None,
    override_nemotron_parse_initial_actors: Optional[int] = None,
    override_nemotron_parse_min_actors: Optional[int] = None,
    override_nemotron_parse_max_actors: Optional[int] = None,
    override_nemotron_parse_gpus_per_actor: Optional[float] = None,
    override_nemotron_parse_batch_size: Optional[int] = None,
    override_ocr_initial_actors: Optional[int] = None,
    override_ocr_min_actors: Optional[int] = None,
    override_ocr_max_actors: Optional[int] = None,
    override_ocr_gpus_per_actor: Optional[float] = None,
    override_ocr_batch_size: Optional[int] = None,
    override_page_elements_initial_actors: Optional[int] = None,
    override_page_elements_min_actors: Optional[int] = None,
    override_page_elements_max_actors: Optional[int] = None,
    override_page_elements_gpus_per_actor: Optional[float] = None,
    override_page_elements_batch_size: Optional[int] = None,
    override_pdf_extract_batch_size: Optional[int] = None,
    override_pdf_extract_cpus_per_task: Optional[float] = None,
    override_pdf_extract_tasks: Optional[int] = None,
) -> RequestedPlan:
    available_gpu_count = max(0, int(cluster_resources.available_gpu_count()))

    if available_gpu_count == 0:
        raise ValueError("No GPUs available")

    def _resolve_int(override: Optional[int], default: int, multiply_by_available_num_gpu: bool) -> int:
        if override is not None:
            return int(override)
        if multiply_by_available_num_gpu:
            return int(default * available_gpu_count)
        return int(default)

    def _resolve_float(override: Optional[float], default: float, multiply_by_available_num_gpu: bool) -> float:
        if override is not None:
            return float(override)
        if multiply_by_available_num_gpu:
            return float(default * available_gpu_count)
        return float(default)

    embed_initial_actors = _resolve_int(override_embed_initial_actors, EMBED_INITIAL_ACTORS, True)
    embed_min_actors = _resolve_int(override_embed_min_actors, EMBED_MIN_ACTORS, True)
    embed_max_actors = _resolve_int(override_embed_max_actors, EMBED_MAX_ACTORS, True)
    embed_gpus_per_actor = _resolve_float(override_embed_gpus_per_actor, EMBED_GPUS_PER_ACTOR, False)
    embed_batch_size = _resolve_int(override_embed_batch_size, EMBED_BATCH_SIZE, False)

    nemotron_parse_initial_actors = _resolve_int(override_nemotron_parse_initial_actors, NEMOTRON_PARSE_INITIAL_ACTORS, True)
    nemotron_parse_min_actors = _resolve_int(override_nemotron_parse_min_actors, NEMOTRON_PARSE_MIN_ACTORS, True)
    nemotron_parse_max_actors = _resolve_int(override_nemotron_parse_max_actors, NEMOTRON_PARSE_MAX_ACTORS, True)
    nemotron_parse_gpus_per_actor = _resolve_float(
        override_nemotron_parse_gpus_per_actor, NEMOTRON_PARSE_GPUS_PER_ACTOR, False
    )
    nemotron_parse_batch_size = _resolve_int(override_nemotron_parse_batch_size, NEMOTRON_PARSE_BATCH_SIZE, False)

    ocr_initial_actors = _resolve_int(override_ocr_initial_actors, OCR_INITIAL_ACTORS, True)
    ocr_min_actors = _resolve_int(override_ocr_min_actors, OCR_MIN_ACTORS, True)
    ocr_max_actors = _resolve_int(override_ocr_max_actors, OCR_MAX_ACTORS, True)
    ocr_gpus_per_actor = _resolve_float(override_ocr_gpus_per_actor, OCR_GPUS_PER_ACTOR, False)
    ocr_batch_size = _resolve_int(override_ocr_batch_size, OCR_BATCH_SIZE, False)

    page_elements_initial_actors = _resolve_int(override_page_elements_initial_actors, PAGE_ELEMENTS_INITIAL_ACTORS, True)
    page_elements_min_actors = _resolve_int(override_page_elements_min_actors, PAGE_ELEMENTS_MIN_ACTORS, True)
    page_elements_max_actors = _resolve_int(override_page_elements_max_actors, PAGE_ELEMENTS_MAX_ACTORS, True)
    page_elements_gpus_per_actor = _resolve_float(override_page_elements_gpus_per_actor, PAGE_ELEMENTS_GPUS_PER_ACTOR, False)
    page_elements_batch_size = _resolve_int(override_page_elements_batch_size, PAGE_ELEMENTS_BATCH_SIZE, False)

    pdf_extract_batch_size = _resolve_int(override_pdf_extract_batch_size, PDF_EXTRACT_BATCH_SIZE, False)
    pdf_extract_cpus_per_task = _resolve_float(override_pdf_extract_cpus_per_task, PDF_EXTRACT_CPUS_PER_TASK, False)
    pdf_extract_tasks = _resolve_int(override_pdf_extract_tasks, PDF_EXTRACT_TASKS, True)

    return RequestedPlan(
        embed_initial_actors=embed_initial_actors,
        embed_min_actors=embed_min_actors,
        embed_max_actors=embed_max_actors,
        embed_gpus_per_actor=embed_gpus_per_actor,
        embed_batch_size=embed_batch_size,
        nemotron_parse_initial_actors=nemotron_parse_initial_actors,
        nemotron_parse_min_actors=nemotron_parse_min_actors,
        nemotron_parse_max_actors=nemotron_parse_max_actors,
        nemotron_parse_gpus_per_actor=nemotron_parse_gpus_per_actor,
        nemotron_parse_batch_size=nemotron_parse_batch_size,
        ocr_initial_actors=ocr_initial_actors,
        ocr_min_actors=ocr_min_actors,
        ocr_max_actors=ocr_max_actors,
        ocr_gpus_per_actor=ocr_gpus_per_actor,
        ocr_batch_size=ocr_batch_size,
        page_elements_initial_actors=page_elements_initial_actors,
        page_elements_min_actors=page_elements_min_actors,
        page_elements_max_actors=page_elements_max_actors,
        page_elements_gpus_per_actor=page_elements_gpus_per_actor,
        page_elements_batch_size=page_elements_batch_size,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
        pdf_extract_tasks=pdf_extract_tasks,
    )
