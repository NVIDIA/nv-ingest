# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep CPU-only artifact selection explicit and separate from service GPU env."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import SkipTest, TestCase

import yaml

CHART = Path(__file__).resolve().parents[1] / "helm"


class TestEmbedCacheEnv(TestCase):
    def render(self, values=None, example=False):
        helm = shutil.which("helm")
        if helm is None:
            raise SkipTest("helm is not available")
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "values.yaml"
            override.write_text(yaml.safe_dump(values or {}))
            command = [
                helm,
                "template",
                "cache-env-test",
                str(CHART),
                "--api-versions",
                "apps.nvidia.com/v1alpha1",
            ]
            if example:
                command += ["-f", str(CHART / "examples/values-nemotron-3-embed-sm120.yaml")]
            command += ["-f", str(override)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]

    def test_default_cache_has_no_extra_environment(self):
        docs = self.render()
        self.assertTrue(any(d["kind"] == "NIMCache" and d["metadata"]["name"] == "nemotron-3-embed-1b" for d in docs))
        for doc in docs:
            if doc["kind"] == "NIMCache":
                self.assertNotIn("env", doc["spec"])

    def test_cache_only_gpu_visibility_and_secret_ref_do_not_leak_to_service(self):
        env = [
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"},
            {
                "name": "HF_TOKEN",
                "valueFrom": {"secretKeyRef": {"name": "download-auth", "key": "token"}},
            },
        ]
        docs = self.render({"nimOperator": {"vlm_embed": {"cacheEnv": env}}})
        self.assertTrue(any(d["kind"] == "NIMCache" and d["metadata"]["name"] == "nemotron-3-embed-1b" for d in docs))
        for doc in docs:
            if doc["kind"] == "NIMCache":
                if doc["metadata"]["name"] == "nemotron-3-embed-1b":
                    self.assertEqual(doc["spec"]["env"], env)
                else:
                    self.assertNotIn("env", doc["spec"])
            if doc["kind"] == "NIMService":
                names = {entry["name"] for entry in doc["spec"]["env"]}
                self.assertNotIn("NVIDIA_VISIBLE_DEVICES", names)
                self.assertNotIn("HF_TOKEN", names)

    def test_sm120_example_aligns_artifact_and_runtime_precision(self):
        docs = self.render(example=True)
        pair = [
            doc
            for doc in docs
            if doc["kind"] in ("NIMCache", "NIMService") and doc["metadata"]["name"] == "nemotron-3-embed-1b"
        ]
        self.assertEqual({doc["kind"] for doc in pair}, {"NIMCache", "NIMService"})
        self.assertEqual(len(pair), 2)
        for doc in pair:
            self.assertEqual(doc["spec"]["nodeSelector"], {"accelerator": "sm120"})
            env = {item["name"]: item.get("value") for item in doc["spec"]["env"]}
            self.assertEqual(env["NIM_ENGINE_PRECISION"], "nvfp4")
            if doc["kind"] == "NIMCache":
                ngc = doc["spec"]["source"]["ngc"]
                self.assertEqual(ngc["model"]["gpus"], [{"ids": ["2BB5"]}])
                self.assertEqual(ngc["modelPuller"], "nvcr.io/nim/nvidia/nemotron-3-embed-1b:2.2.2")
            else:
                self.assertEqual(doc["spec"]["resources"]["limits"]["nvidia.com/gpu"], 1)
                defaults = yaml.safe_load((CHART / "values.yaml").read_text())["nimOperator"]["vlm_embed"]["env"]
                for item in defaults:
                    self.assertIn(item, doc["spec"]["env"])
        config = next(
            doc for doc in docs if doc["kind"] == "ConfigMap" and "retriever-service.yaml" in doc.get("data", {})
        )
        self.assertIn('embed_model_name: "nvidia/nemotron-3-embed-1b"', config["data"]["retriever-service.yaml"])
