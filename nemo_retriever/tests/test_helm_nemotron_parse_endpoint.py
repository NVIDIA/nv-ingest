# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Nemotron Parse service endpoint Helm wiring."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest

import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "nemo_retriever" / "helm"
VALUES_PATH = CHART_DIR / "values.yaml"
CONFIGMAP_PATH = CHART_DIR / "templates" / "configmap.yaml"
PARSE_SERVICE = "nemotron-parse"
PARSE_PATH = "/v1/chat/completions"


def _helm_template(
    extra_args: Sequence[str] = (),
    *,
    api_versions: Sequence[str] = (),
) -> str:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available")
    command = [
        helm,
        "template",
        "parse-endpoint",
        str(CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
    ]
    for version in api_versions:
        command += ["--api-versions", version]
    command += list(extra_args)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_values_expose_parse_endpoint_and_model_overrides() -> None:
    values = yaml.safe_load(VALUES_PATH.read_text())
    endpoints = values["serviceConfig"]["nimEndpoints"]
    assert endpoints["nemotronParseInvokeUrl"] == ""
    assert endpoints["nemotronParseModel"] == ""


def test_configmap_resolves_and_renders_parse_fields() -> None:
    template = CONFIGMAP_PATH.read_text()
    assert '"key" "nemotron_parse"' in template
    assert f'"serviceName" "{PARSE_SERVICE}"' in template
    assert '"configKey" "nemotronParseInvokeUrl"' in template
    assert f'"invokePath" "{PARSE_PATH}"' in template
    assert "nemotronParseModel" in template
    assert "nemotron_parse_invoke_url:" in template
    assert "nemotron_parse_model:" in template


def test_helm_template_parse_null_when_disabled() -> None:
    rendered = _helm_template(
        ("--set", "nimOperator.nemotron_parse.enabled=false"),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert "nemotron_parse_invoke_url: null" in rendered
    assert "nemotron_parse_model: null" in rendered


def test_helm_template_autowires_operator_parse_endpoint() -> None:
    rendered = _helm_template(
        ("--set", "nimOperator.nemotron_parse.enabled=true"),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert f'nemotron_parse_invoke_url: "http://{PARSE_SERVICE}:8000{PARSE_PATH}"' in rendered
    assert 'nemotron_parse_model: "nvidia/nemotron-parse-v1.2"' in rendered


@pytest.mark.parametrize("topology", ["standalone", "split"])
def test_helm_template_autowires_v2_model_from_parse_image_repository(topology: str) -> None:
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "nimOperator.nemotron_parse.image.repository=registry.example/nemotron-parse-v2.0",
            "--set",
            "nimOperator.nemotron_parse.image.tag=2.0.8-variant",
            "--set",
            f"topology.mode={topology}",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert f'nemotron_parse_invoke_url: "http://{PARSE_SERVICE}:8000{PARSE_PATH}"' in rendered
    assert 'nemotron_parse_model: "nvidia/nemotron-parse-v2.0"' in rendered


@pytest.mark.parametrize("hosted_model", ["nvidia/nemotron-parse", "nvidia/nemotron-parse-2.0"])
def test_helm_template_explicit_hosted_endpoint_and_model_win(hosted_model: str) -> None:
    hosted_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "nimOperator.nemotron_parse.image.repository=nvcr.io/nim/nvidia/nemotron-parse-v2.0",
            "--set",
            f"serviceConfig.nimEndpoints.nemotronParseInvokeUrl={hosted_url}",
            "--set",
            f"serviceConfig.nimEndpoints.nemotronParseModel={hosted_model}",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert f'nemotron_parse_invoke_url: "{hosted_url}"' in rendered
    assert f'nemotron_parse_model: "{hosted_model}"' in rendered


def test_helm_template_explicit_model_wins_over_operator_image() -> None:
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "serviceConfig.nimEndpoints.nemotronParseModel=custom/parse",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert 'nemotron_parse_model: "custom/parse"' in rendered


def test_helm_template_external_endpoint_without_model_keeps_url_inference() -> None:
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "nimOperator.nemotron_parse.image.repository=nvcr.io/nim/nvidia/nemotron-parse-v2.0",
            "--set",
            "serviceConfig.nimEndpoints.nemotronParseInvokeUrl=https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert "nemotron_parse_model: null" in rendered


def test_helm_template_unknown_parse_repository_does_not_infer_model() -> None:
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "nimOperator.nemotron_parse.image.repository=registry.example/custom-parse",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    assert f'nemotron_parse_invoke_url: "http://{PARSE_SERVICE}:8000{PARSE_PATH}"' in rendered
    assert "nemotron_parse_model: null" in rendered


def test_helm_template_parse_not_autowired_without_operator_crds() -> None:
    rendered = _helm_template(
        ("--set", "nimOperator.nemotron_parse.enabled=true", "--set", "serviceConfig.vectordb.enabled=false")
    )
    assert "nemotron_parse_invoke_url: null" in rendered
    assert "nemotron_parse_model: null" in rendered


def test_helm_template_parse_endpoint_renders_in_split_mode() -> None:
    rendered = _helm_template(
        (
            "--set",
            "nimOperator.nemotron_parse.enabled=true",
            "--set",
            "topology.mode=split",
        ),
        api_versions=("apps.nvidia.com/v1alpha1",),
    )
    expected = f"http://{PARSE_SERVICE}:8000{PARSE_PATH}"
    assert rendered.count(expected) >= 3
