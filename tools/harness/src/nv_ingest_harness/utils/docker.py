# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Shared Docker service management utilities."""

import subprocess
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yaml"
HEALTH_URL = "http://localhost:7670/v1/health/ready"

# Default profiles for nv-ingest services
DEFAULT_PROFILES = ["retrieval", "table-structure"]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, cwd=cwd)


def stop_services(profiles: list[str] | None = None) -> int:
    print("Stopping services...")
    profile_args = _build_profile_args(profiles or ["*"])

    down_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)] + profile_args + ["down"]
    rc = run_cmd(down_cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"Warning: docker compose down returned {rc}")

    rm_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)] + profile_args + ["rm", "--force"]
    rc = run_cmd(rm_cmd, cwd=REPO_ROOT)
    if rc != 0:
        print(f"Warning: docker compose rm returned {rc}")

    return 0


def clean_services() -> int:
    print("Cleaning up containers, networks, and volumes...")
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "*", "down", "-v", "--remove-orphans"]
    run_cmd(cmd, cwd=REPO_ROOT)

    rm_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "*", "rm", "--force"]
    run_cmd(rm_cmd, cwd=REPO_ROOT)

    print("Cleanup completed!")
    return 0


def start_services(profiles: list[str] | None = None, build: bool = False) -> int:
    profiles = profiles or DEFAULT_PROFILES
    profile_args = _build_profile_args(profiles)

    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)] + profile_args
    if build:
        cmd += ["up", "--force-recreate", "--build", "-d"]
    else:
        cmd += ["up", "--force-recreate", "-d"]

    print(f"Starting services with profiles: {profiles}")
    return run_cmd(cmd, cwd=REPO_ROOT)


def readiness_wait(timeout_s: int = 600, check_milvus: bool = True) -> bool:
    print(f"Waiting for nv-ingest to be ready (timeout: {timeout_s}s)...")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    print("nv-ingest is ready!")
                    break
        except Exception:
            pass
        time.sleep(3)
    else:
        print("Warning: nv-ingest did not become ready within timeout")
        return False

    if check_milvus:
        if not milvus_readiness_wait(timeout_s=min(60, int(deadline - time.time()))):
            return False

    return True


def milvus_readiness_wait(timeout_s: int = 60, uri: str = "http://localhost:19530") -> bool:
    print(f"Waiting for Milvus to be ready (timeout: {timeout_s}s)...")
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=uri)
            client.list_collections()
            client.close()
            print("Milvus is ready!")
            return True
        except Exception:
            pass
        time.sleep(2)

    print("Warning: Milvus did not become ready within timeout")
    return False


def restart_services(
    profiles: list[str] | None = None,
    build: bool = False,
    timeout: int = 600,
    clean: bool = True,
) -> int:
    print("Restarting services" + (" (with build)" if build else ""))

    if clean:
        clean_services()
    else:
        stop_services(profiles)

    rc = start_services(profiles, build=build)
    if rc != 0:
        print(f"Failed to start services (exit code: {rc})")
        return rc

    if not readiness_wait(timeout):
        return 1

    return 0


def _build_profile_args(profiles: list[str]) -> list[str]:
    args = []
    for p in profiles:
        args += ["--profile", p]
    return args
