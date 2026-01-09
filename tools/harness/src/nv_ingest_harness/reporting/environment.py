# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Environment metadata collection for benchmark reports."""

import platform
import socket
import subprocess
from typing import Any

import docker


def get_environment_data() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": _get_gpu_info(),
        "git_commit": _get_git_commit(),
        "docker_images": _get_docker_images(),
        "python_version": platform.python_version(),
    }


def _get_gpu_info() -> str:
    import pynvml

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count == 0:
        return "No GPU detected"
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    return f"{device_count}x {name}" if device_count > 1 else name


def _get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"


def _get_docker_images() -> str:
    client = docker.from_env()
    containers = client.containers.list()
    relevant_prefixes = ["nv-ingest", "nvcr.io", "llama", "nemoretriever", "milvus"]
    images = []
    for container in containers:
        if container.image.tags:
            tag = container.image.tags[0]
            short_name = tag.split("/")[-1]
            if any(prefix in tag.lower() for prefix in relevant_prefixes):
                images.append(short_name)
    return ", ".join(images[:5]) if images else "none detected"


def get_nvingest_image() -> str | None:
    client = docker.from_env()
    for container in client.containers.list():
        if "nv-ingest-ms-runtime" in container.name or "nv-ingest" in container.name:
            if container.image.tags:
                return container.image.tags[0].split("/")[-1]
    return None
