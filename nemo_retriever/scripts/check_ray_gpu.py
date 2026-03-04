#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Print Ray cluster GPU (and CPU) usage and list actors that use or are waiting for GPU.

Run in a separate terminal to see what is holding the GPU while the comparison (or
another Ray job) is running.

To see the comparison's actors, the comparison must use a persistent Ray cluster
(started with `ray start --head` and optional `--num-gpus=1`), and you must set
RAY_ADDRESS when running both the comparison and this script (e.g. RAY_ADDRESS=127.0.0.1:6379).
If the comparison uses ray.init("local") in-process, no separate Ray server exists
and this script cannot attach to it.

Usage (from repo root):
  RAY_ADDRESS=127.0.0.1:6379 uv run python nemo_retriever/scripts/check_ray_gpu.py
  # Or, if a cluster is already running and RAY_ADDRESS is set in the env:
  uv run python nemo_retriever/scripts/check_ray_gpu.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    import ray

    address = os.environ.get("RAY_ADDRESS", "auto")
    ray_addr_env = os.environ.get("RAY_ADDRESS")
    print(
        f"RAY_ADDRESS env: {repr(ray_addr_env) if ray_addr_env is not None else 'unset'} -> connecting with "
        "address={address!r}"
    )
    if not ray.is_initialized():
        try:
            ray.init(address=address, ignore_reinit_error=True)
        except Exception as e:
            print(f"Failed to connect to Ray (address={address!r}): {e}", file=sys.stderr)
            print("Set RAY_ADDRESS to the cluster (e.g. 127.0.0.1:6379) to see that cluster's actors.", file=sys.stderr)
            print(
                "If the comparison uses address='local' (no --ray-address), it starts an in-process cluster and this "
                "script cannot attach.",
                file=sys.stderr,
            )
            sys.exit(1)

    cluster = ray.cluster_resources()
    available = ray.available_resources()
    gpu_total = cluster.get("GPU", 0)
    gpu_avail = available.get("GPU", 0)
    print("--- Ray resources ---")
    print(f"  cluster_resources:    {cluster}")
    print(f"  available_resources:  {available}")
    print(f"  GPU: total={gpu_total}  available={gpu_avail}  used={gpu_total - gpu_avail}")
    print()

    try:
        from ray.util.state import list_actors
    except ImportError:
        print("ray.util.state.list_actors not available (older Ray?). Run: ray status")
        return

    actors = list_actors(detail=True, limit=500)
    gpu_actors = [
        a for a in actors if getattr(a, "required_resources", None) and (a.required_resources or {}).get("GPU", 0) > 0
    ]
    pending = [a for a in actors if getattr(a, "state", None) == "PENDING"]
    pending_gpu = [
        a for a in pending if getattr(a, "required_resources", None) and (a.required_resources or {}).get("GPU", 0) > 0
    ]

    print("--- Actors requiring GPU ---")
    if not gpu_actors:
        print("  (none)")
    else:
        for a in gpu_actors:
            name = getattr(a, "name", None) or "(no name)"
            state = getattr(a, "state", "?")
            res = getattr(a, "required_resources", None) or {}
            pid = getattr(a, "pid", None)
            print(f"  name={name!r}  state={state}  required_resources={res}  pid={pid}")
    print()

    if pending_gpu:
        print("--- PENDING actors waiting for GPU (likely cause of 'stuck') ---")
        for a in pending_gpu:
            name = getattr(a, "name", None) or "(no name)"
            res = getattr(a, "required_resources", None) or {}
            print(f"  name={name!r}  required_resources={res}")
        print()

    print("For more detail: ray status   and   ray list actors")
    return


if __name__ == "__main__":
    main()
