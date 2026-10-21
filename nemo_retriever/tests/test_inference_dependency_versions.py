# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging regressions for the validated local CUDA inference stack."""

import tomllib
from pathlib import Path

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
    assert local["nvidia-cutlass-dsl"].specifier == overrides["nvidia-cutlass-dsl"].specifier
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


def test_lock_preserves_quack_compatible_cutlass_api() -> None:
    """CUTLASS 4.6 removes cute.core.ThrMma used by the pinned Quack kernels."""
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    cutlass = [package for package in lock["package"] if package["name"] == "nvidia-cutlass-dsl"]
    assert cutlass and all(package["version"] == "4.5.2" for package in cutlass)
