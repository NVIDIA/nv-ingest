# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging regressions for the validated local CUDA inference stack."""

import tomllib
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_runtime_requirements_and_uv_overrides_agree() -> None:
    """Keep pip and uv installations on the same vLLM and Transformers stack."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    local = {
        requirement.name: requirement
        for requirement in map(Requirement, project["project"]["optional-dependencies"]["local"])
    }
    overrides = {
        requirement.name: requirement
        for requirement in map(Requirement, project["tool"]["uv"]["override-dependencies"])
    }
    assert local["vllm"].specifier == overrides["vllm"].specifier
    assert "nvidia-cutlass-dsl" not in overrides
    assert local["vllm"].specifier.contains("0.26.0")
    assert local["transformers"].specifier.contains("5.20.2")
    assert overrides["transformers"].specifier.contains("5.20.2")
    assert local["transformers"].specifier.contains("5.17.0")
    assert overrides["transformers"].specifier.contains("5.17.0")
    assert not local["transformers"].specifier.contains("6.0.0")


def test_flashinfer_does_not_install_mismatched_cubin_wheel() -> None:
    """An old cubin wheel makes FlashInfer fail during import, before inference."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    local = {
        requirement.name: requirement
        for requirement in map(Requirement, project["project"]["optional-dependencies"]["local"])
    }
    assert local["flashinfer-python"].specifier.contains("0.6.14")
    assert "flashinfer-cubin" not in local
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    assert "flashinfer-cubin" not in {package["name"] for package in lock["package"]}


def test_lock_uses_quack_compatible_with_vllm_required_cutlass() -> None:
    """Upgrade Quack instead of overriding vLLM's exact CUTLASS requirement."""
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    cutlass = [package for package in lock["package"] if package["name"] == "nvidia-cutlass-dsl"]
    quack = [package for package in lock["package"] if package["name"] == "quack-kernels"]
    assert cutlass and all(package["version"] == "4.6.0" for package in cutlass)
    assert quack and all(package["version"] == "0.6.1" for package in quack)


def test_locked_cuda_stack_satisfies_vendor_requirements() -> None:
    """Detect overrides that make a project lock installable but break plain pip resolution."""
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    runtime_names = {"torch", "torchvision", "nvidia-cutlass-dsl", "quack-kernels", "flashinfer-python"}
    for vendor in ("vllm", "quack-kernels"):
        try:
            vendor_requirements = requires(vendor) or []
        except PackageNotFoundError:
            pytest.skip(f"{vendor} metadata is only available in the local inference environment")
        for requirement_text in vendor_requirements:
            requirement = Requirement(requirement_text)
            if requirement.name not in runtime_names or (requirement.marker and not requirement.marker.evaluate()):
                continue
            locked_versions = [package["version"] for package in lock["package"] if package["name"] == requirement.name]
            assert locked_versions, requirement.name
            assert all(
                requirement.specifier.contains(version) for version in locked_versions
            ), f"Locked {requirement.name}={locked_versions} conflicts with {vendor}'s {requirement}"
