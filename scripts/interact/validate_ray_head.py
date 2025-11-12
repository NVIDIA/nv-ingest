#!/usr/bin/env python3
"""
Validate connectivity and scheduling against a running Ray head node.

This script performs:
- Simple CPU task and actor executions
- Optional GPU-tagged task and actor executions (skipped if no GPUs in cluster)

Examples:
  python scripts/interact/validate_ray_head.py --address ray://localhost:10001
  RAY_ADDRESS=ray://localhost:10001 python scripts/interact/validate_ray_head.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

import click
import ray


def _resolve_address(address_opt: str | None) -> str:
    if address_opt and address_opt.strip():
        return address_opt
    return os.environ.get("RAY_ADDRESS", "ray://localhost:10001")


# ---------------------------
# Simple CPU tasks and actors
# ---------------------------
@ray.remote
def add(a: int, b: int) -> int:
    return a + b


@ray.remote
def ping() -> str:
    import socket

    return f"pong from {socket.gethostname()}"


@ray.remote
class Counter:
    def __init__(self) -> None:
        self._value = 0

    def inc(self, n: int = 1) -> int:
        self._value += n
        return self._value

    def get(self) -> int:
        return self._value


# ---------------------------
# GPU-tagged tasks and actors
# ---------------------------
@ray.remote(num_gpus=1)
def gpu_echo(x: Any) -> Dict[str, Any]:
    # This function does not require CUDA libraries; it validates scheduling
    # to a GPU node by requiring 1 GPU resource.
    import socket

    return {
        "input": x,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


@ray.remote(num_gpus=1)
class GpuInfo:
    def info(self) -> Dict[str, Any]:
        import socket

        return {
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }


def has_gpus() -> bool:
    try:
        cluster = ray.cluster_resources() or {}
        return float(cluster.get("GPU", 0)) > 0.0
    except Exception:
        return False


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--address",
    type=str,
    default=None,
    show_default=False,
    help="Ray cluster address. Defaults to env RAY_ADDRESS or ray://localhost:10001",
)
def main(address: str | None) -> None:
    """Entry point for validation script using Click for configuration."""
    resolved = _resolve_address(address)
    print(f"Connecting to Ray at: {resolved}")
    ray.init(address=resolved, ignore_reinit_error=True)

    # Display basic cluster info
    try:
        cluster = ray.cluster_resources() or {}
        available = ray.available_resources() or {}
        print("Cluster resources:", cluster)
        print("Available resources:", available)
    except Exception as e:
        print(f"Warning: unable to fetch cluster resources: {e}")

    # 1) Simple CPU tasks
    print("\n[CPU] Running simple tasks...")
    res_add = ray.get(add.remote(3, 4))
    res_ping = ray.get(ping.remote())
    print(f"add(3, 4) -> {res_add}")
    print(f"ping() -> {res_ping}")

    # 2) Simple CPU actor
    print("\n[CPU] Running simple actor...")
    ctr = Counter.remote()
    v1 = ray.get(ctr.inc.remote())
    v2 = ray.get(ctr.inc.remote(5))
    v3 = ray.get(ctr.get.remote())
    print(f"Counter after inc(): {v1}, after inc(5): {v2}, get(): {v3}")

    # 3) GPU-tagged tasks/actors (optional)
    print("\n[GPU] Checking for GPU resources in the cluster...")
    if has_gpus():
        print("GPU resources detected. Running GPU-tagged task...")
        gpu_task_out = ray.get(gpu_echo.remote({"msg": "hello-gpu"}))
        print("gpu_echo output:", gpu_task_out)

        print("Starting GPU-tagged actor and fetching info...")
        gpu_actor = GpuInfo.remote()
        info = ray.get(gpu_actor.info.remote())
        print("GpuInfo.info():", info)
    else:
        print("No GPU resources available in the Ray cluster. Skipping GPU tests.")

    print("\nValidation complete.")
    # Click commands return None; use sys.exit for explicit status if desired
    sys.exit(0)


if __name__ == "__main__":
    main()
