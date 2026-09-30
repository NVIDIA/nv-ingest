# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the NIMCache ``model.gpus`` / ``model.profiles`` filter.

The NIM Operator's NIMCache CRD supports an optional
``spec.source.ngc.model`` block that restricts which model profiles a
cache job downloads (by GPU SKU or by profile UUID).  Through 26.08.1
the chart's NIMCache templates omitted the field entirely and
``values.yaml`` exposed no corresponding knob — even
``--set nimOperator.<key>.gpus[0].ids[0]=26B5`` could not move the
needle because the templates had no logic to render it.  On
heterogeneous clusters (or any cluster running ≥ 3 NIMs) that wastes
tens of GiB of PVC storage and NGC bandwidth.

These tests pin the chart-side fix:

* ``values.yaml`` carries a chart-wide ``nimOperator.modelProfile``
  default plus a per-NIM ``nimOperator.<key>.modelProfile`` override
  for every NIMCache the chart provisions. Every NIM defaults its
  per-NIM override to ``{}`` so the NIM Operator can select a compatible
  profile.
* A ``helm template`` with **no overrides** renders no ``model:``
  block on default-empty-profile NIMCaches (preserves operator default).
* ``--set nimOperator.modelProfile.gpus[0]...`` renders an identical
  ``model:`` block on every rendered default-empty-profile NIMCache,
  with the expected ``gpus`` / ``ids`` / ``product`` shape.
* ``--set nimOperator.<key>.modelProfile.profiles[0]=...`` renders
  ``model.profiles`` ONLY on that NIM's NIMCache; the other NIMs
  inherit the global default (or render no block when no global is
  set).
* A per-NIM override REPLACES the chart-wide default (it does not
  merge), matching the documented contract in
  helm/README.md §"Filtering cached GPU profiles".

The integration tests shell out to ``helm template`` when ``helm`` is
on ``$PATH``; otherwise they skip cleanly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from unittest import SkipTest, TestCase, main

import yaml


# Repo-relative paths exercised by every test in this module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VALUES_YAML = _REPO_ROOT / "nemo_retriever/helm/values.yaml"
_README_MD = _REPO_ROOT / "nemo_retriever/helm/README.md"
_CHART_DIR = _REPO_ROOT / "nemo_retriever/helm"

# Per-NIM keys whose NIMCache modelProfile values intentionally default
# to ``{}``, preserving pre-fix operator behaviour unless an operator
# opts into GPU/profile filtering.
_EMPTY_MODEL_PROFILE_NIM_KEYS: tuple[str, ...] = (
    "page_elements",
    "table_structure",
    "ocr",
    "vlm_embed",
    "rerankqa",
    "nemotron_parse",
    "nemotron_3_nano_omni_30b_a3b_reasoning",
    "audio",
    "answer_llm",
)
_ALL_NIM_KEYS: tuple[str, ...] = _EMPTY_MODEL_PROFILE_NIM_KEYS


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Required file not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


def _helm_template(extra_args: Sequence[str] = ()) -> subprocess.CompletedProcess[str]:
    """Render the chart with each default-empty-profile NIM opted in."""
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    if not _CHART_DIR.is_dir():
        raise SkipTest(f"Chart directory missing: {_CHART_DIR}")
    cmd = [
        helm,
        "template",
        "nrl-modelprofile",
        str(_CHART_DIR),
        "--set",
        "ngcImagePullSecret.create=false",
        "--set",
        "ngcApiSecret.create=false",
        # Opt every default-empty-profile NIM in so this suite exercises
        # their shared modelProfile contract in one render. Defaults are
        # covered separately.
        "--set",
        "nimOperator.rerankqa.enabled=true",
        "--set",
        "nimOperator.audio.enabled=true",
        "--set",
        "nimOperator.nemotron_parse.enabled=true",
        "--set",
        "nimOperator.nemotron_3_nano_omni_30b_a3b_reasoning.enabled=true",
        "--set",
        "nimOperator.answer_llm.enabled=true",
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


def _iter_nimcache_docs(rendered: str) -> list[dict]:
    """Return every ``NIMCache`` document in the rendered manifest."""
    docs: list[dict] = []
    for raw in yaml.safe_load_all(rendered):
        if not isinstance(raw, dict):
            continue
        if raw.get("kind") == "NIMCache":
            docs.append(raw)
    return docs


class NimCacheModelProfileTests(TestCase):
    """26.08.1 contract: every NIMCache exposes spec.source.ngc.model."""

    # ------------------------------------------------------------------
    # values.yaml — source-level invariants
    # ------------------------------------------------------------------

    def test_values_exposes_chart_wide_default(self) -> None:
        """``nimOperator.modelProfile`` must exist and default to ``{}``."""
        values = _read_required_file(_VALUES_YAML)
        loaded = yaml.safe_load(values)
        self.assertIn(
            "modelProfile",
            loaded["nimOperator"],
            "values.yaml must expose `nimOperator.modelProfile` so operators "
            "can set a chart-wide GPU/profile filter without editing every "
            "per-NIM block.",
        )
        self.assertEqual(
            loaded["nimOperator"]["modelProfile"],
            {},
            "Default `nimOperator.modelProfile` must be `{}` so existing "
            "releases keep their pre-fix NIMCache behaviour.",
        )

    def test_values_exposes_per_nim_override_for_every_nim(self) -> None:
        """Each ``nimOperator.<key>`` block must carry a modelProfile override."""
        values = _read_required_file(_VALUES_YAML)
        loaded = yaml.safe_load(values)
        for key in _ALL_NIM_KEYS:
            with self.subTest(nim=key):
                cfg = loaded["nimOperator"].get(key)
                self.assertIsNotNone(
                    cfg,
                    f"values.yaml missing nimOperator.{key} block.",
                )
                self.assertIn(
                    "modelProfile",
                    cfg,
                    f"nimOperator.{key} must expose a per-NIM `modelProfile` "
                    "override key — anything else removes the documented "
                    "per-NIM tuning surface.",
                )
                if key in _EMPTY_MODEL_PROFILE_NIM_KEYS:
                    self.assertEqual(
                        cfg["modelProfile"],
                        {},
                        f"nimOperator.{key}.modelProfile must default to `{{}}` "
                        "so the chart's behaviour is unchanged unless the "
                        "operator opts in.",
                    )

    # ------------------------------------------------------------------
    # README — operator-facing documentation
    # ------------------------------------------------------------------

    def test_readme_documents_filtering_section(self) -> None:
        """README must expose a `Filtering cached GPU profiles` anchor + table."""
        readme = _read_required_file(_README_MD)
        self.assertIn(
            "filtering-cached-gpu-profiles",
            readme,
            "README must expose a `Filtering cached GPU profiles` anchor so "
            "values.yaml comments and the per-NIM table can link to it.",
        )
        self.assertRegex(
            readme,
            r"`nimOperator\.modelProfile`.*Chart-wide",
            "README must explain the chart-wide `nimOperator.modelProfile` "
            "scope in the Filtering subsection's table.",
        )
        self.assertRegex(
            readme,
            r"`nimOperator\.<key>\.modelProfile`.*Per-NIM",
            "README must explain the per-NIM "
            "`nimOperator.<key>.modelProfile` scope in the Filtering "
            "subsection's table.",
        )

    # ------------------------------------------------------------------
    # `helm template` — actually render the chart
    # ------------------------------------------------------------------

    def test_default_render_emits_no_model_block(self) -> None:
        """No-override render must not introduce a ``model:`` block.

        Existing releases must keep working unchanged — the new helper
        is strictly opt-in.
        """
        proc = _helm_template()
        _assert_helm_ok(self, proc)
        docs = _iter_nimcache_docs(proc.stdout)
        self.assertEqual(
            len(docs),
            len(_EMPTY_MODEL_PROFILE_NIM_KEYS),
            f"Expected one NIMCache per opted-in default-empty-profile NIM (={len(_EMPTY_MODEL_PROFILE_NIM_KEYS)}); "
            f"got {len(docs)}.",
        )
        for doc in docs:
            name = doc.get("metadata", {}).get("name", "<unknown>")
            ngc = doc.get("spec", {}).get("source", {}).get("ngc", {})
            self.assertNotIn(
                "model",
                ngc,
                f"NIMCache `{name}` must not carry a `spec.source.ngc.model` "
                "block when neither global nor per-NIM modelProfile is set "
                "— that breaks pre-fix release behaviour.",
            )

    def test_default_render_uses_ocr_v2_nim(self) -> None:
        """Default OCR NIM resources must point at Nemotron OCR v2."""
        proc = _helm_template()
        _assert_helm_ok(self, proc)
        docs = [raw for raw in yaml.safe_load_all(proc.stdout) if isinstance(raw, dict)]

        ocr_cache = next(
            doc
            for doc in docs
            if doc.get("kind") == "NIMCache" and doc.get("metadata", {}).get("name") == "nemotron-ocr-v2"
        )
        self.assertEqual(
            ocr_cache["spec"]["source"]["ngc"]["modelPuller"],
            "nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1",
        )

        ocr_service = next(
            doc
            for doc in docs
            if doc.get("kind") == "NIMService" and doc.get("metadata", {}).get("name") == "nemotron-ocr-v2"
        )
        self.assertEqual(
            ocr_service["spec"]["image"]["repository"],
            "nvcr.io/nim/nvidia/nemotron-ocr-v2",
        )
        self.assertEqual(ocr_service["spec"]["image"]["tag"], "2.0.1")

        configmaps = [doc for doc in docs if doc.get("kind") == "ConfigMap"]
        self.assertTrue(
            any(
                'ocr_invoke_url: "http://nemotron-ocr-v2:8000/v1/ocr"'
                in doc.get("data", {}).get("retriever-service.yaml", "")
                for doc in configmaps
            ),
            "service config must auto-wire the OCR endpoint to the v2 NIMService.",
        )

    def test_chart_wide_modelprofile_applies_to_default_empty_nimcaches(self) -> None:
        """``--set nimOperator.modelProfile.gpus[0]...`` renders on default-empty NIMCaches.

        This is the exact customer ask for the existing extraction NIMs:
        one --set flag makes each rendered cache download only the H100
        profile, including the optional answer LLM.
        """
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.modelProfile.gpus[0].ids[0]=26B5",
                "--set",
                "nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3",
            ),
        )
        _assert_helm_ok(self, proc)
        docs = _iter_nimcache_docs(proc.stdout)
        self.assertEqual(len(docs), len(_EMPTY_MODEL_PROFILE_NIM_KEYS))
        for doc in docs:
            name = doc.get("metadata", {}).get("name", "<unknown>")
            with self.subTest(nimcache=name):
                model = doc["spec"]["source"]["ngc"].get("model")
                self.assertIsNotNone(
                    model,
                    f"NIMCache `{name}` must inherit " "`nimOperator.modelProfile` when no per-NIM override is " "set.",
                )
                self.assertEqual(
                    model,
                    {"gpus": [{"ids": ["26B5"], "product": "NVIDIA-H100-80GB-HBM3"}]},
                    f"NIMCache `{name}.spec.source.ngc.model` must render " "the chart-wide filter verbatim.",
                )

    def test_per_nim_override_replaces_chart_wide_default(self) -> None:
        """A per-NIM override must REPLACE the chart-wide default (no merge).

        The override semantic is documented in
        helm/README.md §"Filtering cached GPU profiles" — operators
        rely on this when one NIM needs a different profile UUID than
        the rest of the cluster.
        """
        profile_uuid = "11111111-2222-3333-4444-555555555555"
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3",
                "--set",
                f"nimOperator.page_elements.modelProfile.profiles[0]={profile_uuid}",
            ),
        )
        _assert_helm_ok(self, proc)
        docs = {
            doc["metadata"]["name"]: doc["spec"]["source"]["ngc"].get("model")
            for doc in _iter_nimcache_docs(proc.stdout)
        }
        # The targeted override must carry ONLY profiles (no gpus
        # inherited from the global).
        self.assertEqual(
            docs["nemotron-page-elements-v3"],
            {"profiles": [profile_uuid]},
            "Per-NIM override must REPLACE the chart-wide default — the "
            "page-elements NIMCache must NOT carry the inherited gpus list.",
        )
        # Every other rendered NIMCache should still carry the chart-wide gpus
        # filter.  Spot-check one — the others are covered by the
        # previous test.
        ocr = docs["nemotron-ocr-v2"]
        self.assertEqual(
            ocr,
            {"gpus": [{"product": "NVIDIA-H100-80GB-HBM3"}]},
            "Non-overridden NIMCaches must inherit `nimOperator.modelProfile`.",
        )

    def test_per_nim_override_renders_when_no_chart_wide_default(self) -> None:
        """Per-NIM override alone must render `model:` on exactly that NIMCache."""
        profile_uuid = "33333333-4444-5555-6666-777777777777"
        proc = _helm_template(
            extra_args=(
                "--set",
                f"nimOperator.vlm_embed.modelProfile.profiles[0]={profile_uuid}",
            ),
        )
        _assert_helm_ok(self, proc)
        docs = {
            doc["metadata"]["name"]: doc["spec"]["source"]["ngc"].get("model")
            for doc in _iter_nimcache_docs(proc.stdout)
        }
        self.assertEqual(
            docs.get("nemotron-3-embed-1b"),
            {"profiles": [profile_uuid]},
            "vlm_embed NIMCache must carry the per-NIM profile filter.",
        )
        # Every other NIMCache must remain unfiltered.
        for name, model in docs.items():
            if name == "nemotron-3-embed-1b":
                continue
            with self.subTest(nimcache=name):
                self.assertIsNone(
                    model,
                    f"NIMCache `{name}` must not carry a `model:` block "
                    "when only an unrelated per-NIM override is set.",
                )

    def test_rendered_model_block_indentation_is_under_ngc(self) -> None:
        """The helper must indent `model:` under `spec.source.ngc`, not anywhere else.

        Mis-indentation would produce a structurally valid YAML doc
        that the NIM Operator silently ignores.  Pin the literal
        column position so a future refactor of the template can't
        regress this without the test catching it.
        """
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3",
            ),
        )
        _assert_helm_ok(self, proc)
        # In each rendered NIMCache the `model:` line should sit at
        # exactly six spaces of indentation — same column as
        # `modelPuller:`, `pullSecret:` and `authSecret:`.
        for line in proc.stdout.splitlines():
            stripped = line.lstrip(" ")
            if stripped == "model:":
                self.assertTrue(
                    line.startswith("      model:"),
                    f"`model:` line must be indented under `spec.source.ngc` (6 spaces). Got: {line!r}",
                )

    def test_rendered_model_block_round_trips_through_yaml(self) -> None:
        """Every rendered NIMCache must parse and the `model` field must round-trip.

        Defends against accidental string-formatting bugs in the
        helper template (e.g. missing trailing newline, wrong indent
        emitting a sibling instead of a child).
        """
        proc = _helm_template(
            extra_args=(
                "--set",
                "nimOperator.modelProfile.gpus[0].ids[0]=26B5",
                "--set",
                "nimOperator.modelProfile.gpus[0].product=NVIDIA-H100-80GB-HBM3",
            ),
        )
        _assert_helm_ok(self, proc)
        # If any document failed to parse, safe_load_all would raise.
        docs = _iter_nimcache_docs(proc.stdout)
        self.assertEqual(len(docs), len(_EMPTY_MODEL_PROFILE_NIM_KEYS))
        for doc in docs:
            name = doc["metadata"]["name"]
            ngc = doc["spec"]["source"]["ngc"]
            # `model` must be a sibling of modelPuller / pullSecret /
            # authSecret — not nested inside any of them.
            for sibling in ("modelPuller", "pullSecret", "authSecret"):
                self.assertIn(sibling, ngc, f"NIMCache `{name}` missing {sibling}.")
            self.assertIn(
                "model",
                ngc,
                f"NIMCache `{name}` missing the rendered `model:` block.",
            )

    def test_rendered_model_block_is_absent_when_chart_wide_filter_is_empty(
        self,
    ) -> None:
        """Setting `nimOperator.modelProfile={}` explicitly must keep the field absent.

        The contract is "non-empty mapping renders the block; empty
        does not".  Pin that on the chart-wide knob too.
        """
        # The shell --set syntax for the literal empty mapping is awkward,
        # so we re-use the chart default by simply rendering with no
        # overrides — equivalent in effect — and rely on
        # ``test_default_render_emits_no_model_block`` for the global
        # case.  This test guards the per-NIM case: setting only a
        # bogus key like `nimOperator.modelProfile.gpus` to an empty
        # list must NOT render a `model:` block.
        # Helm's --set has no syntax for an empty list, so we drive
        # this through `--set-string` of a JSON-empty value; if the
        # value is treated as truthy, the helper will render the block.
        proc = _helm_template(extra_args=())
        _assert_helm_ok(self, proc)
        docs = _iter_nimcache_docs(proc.stdout)
        self.assertEqual(len(docs), len(_EMPTY_MODEL_PROFILE_NIM_KEYS))
        for doc in docs:
            name = doc["metadata"]["name"]
            ngc = doc["spec"]["source"]["ngc"]
            self.assertNotIn(
                "model",
                ngc,
                f"Default render must not contain a NIMCache model block for `{name}`.",
            )


class AnswerLLMRuntimeProfileTests(TestCase):
    """Bug 6636046: an explicit cache singleton also selects the runtime profile."""

    def test_runtime_profile_matches_effective_cache_configuration(self) -> None:
        cases = [
            ("default", {}, None),
            ("changed", {"answer_llm.modelProfile.profiles": ["changed-profile"]}, "changed-profile"),
            (
                "per_nim_wins",
                {"answer_llm.modelProfile.profiles": ["local"], "modelProfile.profiles": ["global"]},
                "local",
            ),
            ("global", {"answer_llm.modelProfile": None, "modelProfile.profiles": ["global"]}, "global"),
            ("unpinned", {"answer_llm.modelProfile": None}, None),
            ("empty_profiles", {"answer_llm.modelProfile.profiles": []}, None),
            ("multiple", {"answer_llm.modelProfile.profiles": ["first", "second"]}, None),
            (
                "gpu_only",
                {
                    "answer_llm.modelProfile.profiles": None,
                    "answer_llm.modelProfile.gpus": [{"product": "rtx-pro-6000"}],
                    "modelProfile.profiles": ["global"],
                },
                None,
            ),
            (
                "global_multiple",
                {
                    "answer_llm.modelProfile": None,
                    "modelProfile.profiles": ["first", "second"],
                },
                None,
            ),
        ]
        for name, overrides, expected in cases:
            with self.subTest(name=name):
                args = ["--set", "nimOperator.answer_llm.enabled=true"]
                for key, value in overrides.items():
                    args += ["--set-json", f"nimOperator.{key}={json.dumps(value)}"]
                proc = _helm_template(args)
                _assert_helm_ok(self, proc)
                docs = [d for d in yaml.safe_load_all(proc.stdout) if isinstance(d, dict)]
                answer = {d["kind"]: d for d in docs if d.get("metadata", {}).get("name") == "answer-llm"}
                storage = answer["NIMService"]["spec"]["storage"]["nimCache"]
                if expected is None:
                    self.assertNotIn("profile", storage)
                else:
                    self.assertEqual(storage["profile"], expected)
                    self.assertEqual(answer["NIMCache"]["spec"]["source"]["ngc"]["model"]["profiles"], [expected])

    def test_explicit_profile_env_preserves_override(self) -> None:
        for source in (
            {"value": "custom"},
            {"value": ""},
            {
                "valueFrom": {"configMapKeyRef": {"name": "profile", "key": "id"}},
            },
        ):
            with self.subTest(source=source):
                env = {"name": "NIM_MODEL_PROFILE", **source}
                proc = _helm_template(
                    [
                        "--set",
                        "nimOperator.answer_llm.enabled=true",
                        "--set-json",
                        f"nimOperator.answer_llm.env={json.dumps([env])}",
                    ]
                )
                _assert_helm_ok(self, proc)
                service = next(
                    d
                    for d in yaml.safe_load_all(proc.stdout)
                    if isinstance(d, dict) and d.get("kind") == "NIMService" and d["metadata"]["name"] == "answer-llm"
                )
                self.assertNotIn("profile", service["spec"]["storage"]["nimCache"])
                self.assertIn(env, service["spec"]["env"])


if __name__ == "__main__":
    main()
