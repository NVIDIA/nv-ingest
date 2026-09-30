# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm wiring for operator-managed answer-generation LLM NIMs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main


_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALUES_YAML = _REPO_ROOT / "nemo_retriever/helm/values.yaml"
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"

_ANSWER_LLM_KEY = "  answer_llm:"
_ANSWER_LLM_SERVICE = "answer-llm"
_LIGHTNING_REPOSITORY = "nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b"
_LIGHTNING_TAG = "2.0.9-variant"
_LIGHTNING_MODEL = "openai/nvidia/nemotron-3.5-lightning-30b-a3b"
_LIGHTNING_SERVED_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
_NANO_SERVICE = "nemotron-3-nano"
_NANO_REPOSITORY = "nvcr.io/nim/nvidia/nemotron-3-nano"
_NANO_TAG = "1.7.0-variant"
_NANO_MODEL = "openai/nvidia/nemotron-3-nano-30b-a3b"
_NANO_SERVED_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
_NANO_A100_PROFILE = "5f89f01a0af587fd8bae50c611b1f358f92effdb9fb29362e1af0a986e5561c3"


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    cmd = [
        helm,
        "template",
        "retriever",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        "--api-versions",
        "apps.nvidia.com/v1alpha1",
    ]
    cmd += list(extra_args)
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _assert_helm_ok(self: TestCase, proc: subprocess.CompletedProcess[str]) -> None:
    self.assertEqual(
        proc.returncode,
        0,
        f"`helm template` failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
    )


class HelmAnswerLLMGenerationTests(TestCase):
    def test_values_define_generic_answer_llm_with_lightning_defaults(self) -> None:
        values = _read_required_file(_VALUES_YAML)

        self.assertIn(_ANSWER_LLM_KEY, values)
        start = values.index(_ANSWER_LLM_KEY)
        block = values[start : values.index("\n  rerankqa:", start)]
        self.assertIn("enabled: false", block)
        self.assertIn(f"nimServiceName: {_ANSWER_LLM_SERVICE}", block)
        self.assertIn(f"repository: {_LIGHTNING_REPOSITORY}", block)
        self.assertIn(f'tag: "{_LIGHTNING_TAG}"', block)
        self.assertIn(f'model: "{_LIGHTNING_MODEL}"', block)
        self.assertIn("resources: {}", block)
        self.assertIn("modelProfile: {}", block)
        self.assertIn(_LIGHTNING_SERVED_MODEL, block)
        self.assertIn("NIM_MODEL_NAME", block)
        self.assertIn("--reasoning-parser nemotron_v3", block)
        self.assertIn('size: "250Gi"', block)
        self.assertIn("reasoningEnabled: true", values)
        self.assertIn('ragSystemPromptPrefix: ""', block)

    def test_default_render_omits_answer_llm_and_disables_llm_answering(self) -> None:
        proc = _helm_template()
        _assert_helm_ok(self, proc)

        self.assertNotIn(f"name: {_ANSWER_LLM_SERVICE}", proc.stdout)
        self.assertIn("llm:", proc.stdout)
        self.assertIn("enabled: false", proc.stdout)
        self.assertIn("api_base: null", proc.stdout)
        self.assertNotIn("NEMO_RETRIEVER_LLM_API_KEY", proc.stdout)

    def test_answer_llm_opt_in_renders_default_lightning_nim_and_autowires_llm_config(self) -> None:
        proc = _helm_template(extra_args=("--set", "nimOperator.answer_llm.enabled=true"))
        _assert_helm_ok(self, proc)

        self.assertIn(f"name: {_ANSWER_LLM_SERVICE}", proc.stdout)
        self.assertIn(f"repository: {_LIGHTNING_REPOSITORY}", proc.stdout)
        self.assertIn(f"tag: {_LIGHTNING_TAG}", proc.stdout)
        self.assertIn("nvidia.com/gpu: 1", proc.stdout)
        self.assertNotIn("NIM_MODEL_PROFILE", proc.stdout)
        self.assertIn("NIM_MODEL_NAME", proc.stdout)
        self.assertIn("NIM_SERVED_MODEL_NAME", proc.stdout)
        self.assertIn(_LIGHTNING_SERVED_MODEL, proc.stdout)
        self.assertIn("NIM_PASSTHROUGH_ARGS", proc.stdout)
        self.assertIn("--reasoning-parser nemotron_v3", proc.stdout)
        self.assertNotIn("NCCL_IB_DISABLE", proc.stdout)
        self.assertNotIn("NCCL_P2P_DISABLE", proc.stdout)
        self.assertIn(f'api_base: "http://{_ANSWER_LLM_SERVICE}:8000/v1"', proc.stdout)
        self.assertIn(f'model: "{_LIGHTNING_MODEL}"', proc.stdout)
        self.assertIn("reasoning_enabled: true", proc.stdout)
        self.assertIn("max_tokens: 4096", proc.stdout)
        self.assertIn("rag_system_prompt_prefix: null", proc.stdout)
        self.assertIn("enabled: true", proc.stdout)
        self.assertIn("NEMO_RETRIEVER_LLM_API_KEY", proc.stdout)
        self.assertIn('name: "ngc-api"', proc.stdout)
        self.assertIn('key: "NGC_API_KEY"', proc.stdout)

    def test_answer_llm_rejects_empty_pull_secrets(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.page_elements.enabled=false",
                "--set",
                "nimOperator.table_structure.enabled=false",
                "--set",
                "nimOperator.ocr.enabled=false",
                "--set",
                "nimOperator.vlm_embed.enabled=false",
                "--set",
                "nimOperator.answer_llm.enabled=true",
                "--set-json",
                "nimOperator.answer_llm.image.pullSecrets=[]",
                "--set",
                "ngcImagePullSecret.name=",
            )
        )

        self.assertNotEqual(proc.returncode, 0)
        combined = proc.stdout + proc.stderr
        self.assertIn("nimOperator.answer_llm.image.pullSecrets is empty", combined)
        self.assertIn("ngcImagePullSecret.name is unset", combined)

    def test_answer_llm_can_swap_to_nano_image_model_and_profile(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.answer_llm.enabled=true",
                "--set",
                f"nimOperator.answer_llm.nimServiceName={_NANO_SERVICE}",
                "--set",
                f"nimOperator.answer_llm.image.repository={_NANO_REPOSITORY}",
                "--set",
                f"nimOperator.answer_llm.image.tag={_NANO_TAG}",
                "--set",
                f"nimOperator.answer_llm.model={_NANO_MODEL}",
                "--set-json",
                f'nimOperator.answer_llm.modelProfile={{"profiles":["{_NANO_A100_PROFILE}"]}}',
                "--set-json",
                'nimOperator.answer_llm.resources={"limits":{"nvidia.com/gpu":1},"requests":{"nvidia.com/gpu":1}}',
                "--set",
                "nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT",
                "--set-string",
                "nimOperator.answer_llm.env[0].value=8000",
                "--set",
                "nimOperator.answer_llm.env[1].name=NIM_SERVED_MODEL_NAME",
                "--set-string",
                f"nimOperator.answer_llm.env[1].value={_NANO_SERVED_MODEL}",
                "--set",
                "nimOperator.answer_llm.env[2].name=NIM_TENSOR_PARALLEL_SIZE",
                "--set-string",
                "nimOperator.answer_llm.env[2].value=1",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn(f"name: {_NANO_SERVICE}", proc.stdout)
        self.assertIn(f"repository: {_NANO_REPOSITORY}", proc.stdout)
        self.assertIn(f"tag: {_NANO_TAG}", proc.stdout)
        self.assertIn("NIM_SERVED_MODEL_NAME", proc.stdout)
        self.assertIn(_NANO_SERVED_MODEL, proc.stdout)
        self.assertIn("NIM_TENSOR_PARALLEL_SIZE", proc.stdout)
        self.assertIn(_NANO_A100_PROFILE, proc.stdout)
        self.assertIn("nvidia.com/gpu: 1", proc.stdout)
        self.assertIn(f'api_base: "http://{_NANO_SERVICE}:8000/v1"', proc.stdout)
        self.assertIn(f'model: "{_NANO_MODEL}"', proc.stdout)
        self.assertIn("rag_system_prompt_prefix: null", proc.stdout)
        self.assertIn("enabled: true", proc.stdout)

    def test_explicit_llm_api_base_wins_without_operator_nim(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.llm.enabled=true",
                "--set",
                "serviceConfig.llm.apiBase=http://external-llm:8000/v1",
                "--set",
                "serviceConfig.llm.model=openai/custom-answerer",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertNotIn(f"name: {_ANSWER_LLM_SERVICE}", proc.stdout)
        self.assertIn('api_base: "http://external-llm:8000/v1"', proc.stdout)
        self.assertIn('model: "openai/custom-answerer"', proc.stdout)
        self.assertIn("rag_system_prompt_prefix: null", proc.stdout)
        self.assertIn("reasoning_enabled: true", proc.stdout)

    def test_service_llm_reasoning_enabled_can_be_disabled(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.llm.enabled=true",
                "--set",
                "serviceConfig.llm.apiBase=http://external-llm:8000/v1",
                "--set",
                "serviceConfig.llm.reasoningEnabled=false",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn("reasoning_enabled: false", proc.stdout)
        self.assertIn("rag_system_prompt_prefix: null", proc.stdout)

    def test_answer_llm_filters_reserved_ngc_api_key_from_custom_env(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.answer_llm.enabled=true",
                "--set",
                "nimOperator.answer_llm.env[0].name=NGC_API_KEY",
                "--set-string",
                "nimOperator.answer_llm.env[0].value=ignored-user-secret",
                "--set",
                "nimOperator.answer_llm.env[1].name=OTHER_ENV",
                "--set-string",
                "nimOperator.answer_llm.env[1].value=kept",
            )
        )
        _assert_helm_ok(self, proc)

        answer_llm_service_start = proc.stdout.index(f"kind: NIMService\nmetadata:\n  name: {_ANSWER_LLM_SERVICE}")
        next_manifest_start = proc.stdout.find("\n---", answer_llm_service_start + 1)
        answer_llm_service = proc.stdout[
            answer_llm_service_start : next_manifest_start if next_manifest_start != -1 else len(proc.stdout)
        ]

        self.assertNotIn("ignored-user-secret", answer_llm_service)
        self.assertIn("OTHER_ENV", answer_llm_service)
        self.assertIn("value: kept", answer_llm_service)
        self.assertEqual(answer_llm_service.count("- name: NGC_API_KEY"), 1)

    def test_llm_api_key_secret_renders_env_not_configmap_value(self) -> None:
        proc = _helm_template(
            extra_args=(
                "--set",
                "serviceConfig.llm.enabled=true",
                "--set",
                "serviceConfig.llm.apiBase=http://external-llm:8000/v1",
                "--set",
                "serviceConfig.llm.apiKeySecret.name=llm-secret",
                "--set",
                "serviceConfig.llm.apiKeySecret.key=OPENAI_API_KEY",
            )
        )
        _assert_helm_ok(self, proc)

        self.assertIn("api_key: null", proc.stdout)
        self.assertIn("NEMO_RETRIEVER_LLM_API_KEY", proc.stdout)
        self.assertIn('name: "llm-secret"', proc.stdout)
        self.assertIn('key: "OPENAI_API_KEY"', proc.stdout)
        self.assertNotIn('api_key: "', proc.stdout)


if __name__ == "__main__":
    main()
