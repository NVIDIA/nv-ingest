# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution evidence must distinguish Parse inference from preserved PDF text."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from nemo_retriever.common.modality.parse.inference_audit import summarize_parse_inference
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.ingest.execution import IngestExecutionResult
from nemo_retriever.ingestor.graph_ingestor import GraphIngestionError, GraphIngestor
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseGPUActor, nemotron_parse_pages


def frame(statuses):
    return pd.DataFrame(
        [
            {
                "path": "paper.pdf",
                "page_number": i + 1,
                "text": "Native PDF text",
                "nemotron_parse_v1_2": {"inference_status": status, "error": None},
            }
            for i, status in enumerate(statuses)
        ]
    )


def test_coverage_counts_pages_not_split_rows_and_never_overrides_failures():
    data = frame(["succeeded", "skipped", "failed", "unknown"])
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    assert summarize_parse_inference(data) == {
        "pages_seen": 4,
        "pages_succeeded": 1,
        "pages_skipped": 1,
        "pages_failed": 1,
        "pages_missing_evidence": 1,
    }
    failed = data.iloc[[0]].copy()
    failed["nemotron_parse_v1_2"] = [{"inference_status": "succeeded", "error": {"message": "failed"}}]
    assert summarize_parse_inference(pd.concat([data, failed]))["pages_failed"] == 2


def test_coverage_rejects_missing_identity_and_old_text_only_results():
    data = frame(["succeeded"])
    data["page_number"] = None
    assert summarize_parse_inference(data)["pages_missing_evidence"] == 1
    data = frame(["succeeded"]).drop(columns="nemotron_parse_v1_2")
    assert summarize_parse_inference(data)["pages_succeeded"] == 0


def test_workflow_summary_adds_optional_coverage_without_changing_other_summaries():
    result = IngestExecutionResult(SimpleNamespace(documents=["paper.pdf"]), None, 1, 1, None, "db", "table", {})
    assert "parse_inference" not in result.to_summary_dict()
    coverage = summarize_parse_inference(frame(["succeeded"]))
    result = IngestExecutionResult(result.plan, None, 1, 1, None, "db", "table", {}, coverage)
    assert result.to_summary_dict()["parse_inference"] == coverage


@pytest.mark.parametrize(
    "outputs,status", [([""], "succeeded"), ([], "failed"), (["", ""], "failed"), ([None], "failed")]
)
def test_inference_requires_response_per_image_and_accepts_blank_pages(monkeypatch, outputs, status):
    monkeypatch.setattr("nemo_retriever.operators.extract.parse.nemotron_parse._decode_page_image", lambda _: object())
    batch = pd.DataFrame({"page_image": [{"image_b64": "image"}, {}], "text": ["", "Native"]})
    model = SimpleNamespace(invoke_batch=lambda *a, **k: outputs)
    out = nemotron_parse_pages(batch, model=model, extract_text=True)
    assert [p["inference_status"] for p in out["nemotron_parse_v1_2"]] == [status, "skipped"]


@pytest.mark.parametrize("policy", ["raise", "collect"])
def test_local_initialization_error_preserved_and_honors_error_policy(monkeypatch, policy):
    actor = NemotronParseGPUActor(extract_text=True)

    def fail():
        raise OSError("Missing offline model dependency")

    monkeypatch.setattr(actor, "_ensure_model", fail)
    out = actor(frame(["succeeded"]))
    assert out.iloc[0]["text"] == "Native PDF text"
    assert out.iloc[0]["nemotron_parse_v1_2"]["inference_status"] == "failed"
    ingest = GraphIngestor(run_mode="inprocess", error_policy=policy).extract(ExtractParams(method="nemotron_parse"))
    if policy == "raise":
        with pytest.raises(GraphIngestionError, match="Missing offline model dependency"):
            ingest._finalize_ingest_result(out, return_failures=False)
    else:
        assert ingest._finalize_ingest_result(out, return_failures=False) is out
        assert ingest._collect_failure_tuples(out)


def test_tied_patch_is_pinned_reaches_workers_and_restores_pythonpath(tmp_path, monkeypatch):
    from nemo_retriever.models.local.nemotron_parse_2_0_runtime import parse_vllm_runtime

    patch = tmp_path / "vllm_tied_patch/sitecustomize.py"
    patch.parent.mkdir()
    patch.write_text("import os\nos.environ['NRL_TEST_PATCH_APPLIED'] = 'yes'\n")
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.delenv("NRL_TEST_PATCH_APPLIED", raising=False)
    with parse_vllm_runtime("nvidia/NVIDIA-Nemotron-Parse-2.0", "immutable"):
        assert os.environ["NRL_TEST_PATCH_APPLIED"] == "yes"
        assert os.environ["PYTHONPATH"].endswith(os.pathsep + "existing-path")
        env = dict(os.environ)
        env.pop("NRL_TEST_PATCH_APPLIED")
        child = subprocess.check_output(
            [sys.executable, "-c", "import os;print(os.environ['NRL_TEST_PATCH_APPLIED'])"], env=env, text=True
        )
        assert child.strip() == "yes"
    assert os.environ["PYTHONPATH"] == "existing-path"
    assert calls[0]["revision"] == "immutable"


def test_missing_compatibility_patch_fails_instead_of_using_untied_head(tmp_path, monkeypatch):
    from nemo_retriever.models.local.nemotron_parse_2_0_runtime import parse_vllm_runtime

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="tied-head patch"):
        with parse_vllm_runtime("model", "pin"):
            pytest.fail("Missing patch must fail before inference")


@pytest.mark.parametrize("return_results", [False, True])
def test_local_parse_batch_errors_are_checked_before_vdb_upload(monkeypatch, return_results):
    ingest = GraphIngestor(run_mode="batch").files(["paper.pdf"]).extract(method="nemotron_parse").vdb_upload()
    cluster = SimpleNamespace(total_gpu_count=lambda: 0, available_gpu_count=lambda: 0, total_cpu_count=lambda: 4)

    class Executor:
        def __init__(self, *a, **k):
            pass

        def _ingest(self, data, *, return_results, validate_batch):
            data = frame(["failed"])
            data["nemotron_parse_v1_2"] = [{"error": {"message": "Local Parse startup failed"}}]
            validate_batch(data)
            pytest.fail("Failed local Parse must stop before uploading preserved native text")

    monkeypatch.setattr(ingest, "_ensure_batch_runtime", lambda: (None, cluster))
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", Executor)
    with pytest.raises(GraphIngestionError, match="Local Parse startup failed"):
        ingest.ingest(return_results=return_results)


def test_checkpoint_configuration_loads_offline_without_recursive_radio_import(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import shutil

    import yaml
    from transformers import AutoConfig

    fixture = Path(__file__).parent / "fixtures/parse_2_0"
    (tmp_path / "config.json").write_text(json.dumps(yaml.safe_load((fixture / "config.yaml").read_text())))
    shutil.copy2(fixture / "hf_nemotron_parse_config.py.txt", tmp_path / "hf_nemotron_parse_config.py")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "requests.sessions.Session.request", lambda *a, **k: pytest.fail("Offline loading made a request")
    )
    config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True)
    assert config.encoder.model_type == "radio"
    assert type(config.encoder).__name__ == "NemotronParseEncoderConfig"
    assert config.tie_word_embeddings


@pytest.mark.parametrize(
    "top_tied,decoder_tied,explicit,should_tie",
    [
        (True, False, False, True),
        (False, True, False, True),
        (False, False, False, False),
        (True, True, True, False),
    ],
)
def test_checkpoint_tied_head_patch_guards_and_idempotence(monkeypatch, top_tied, decoder_tied, explicit, should_tie):
    from pathlib import Path
    import runpy
    from types import ModuleType

    class Model:
        def load_weights(self, weights):
            self.loaded = list(weights)
            return {name for name, _ in self.loaded}

    names = ["vllm", "vllm.model_executor", "vllm.model_executor.models", "vllm.model_executor.models.nemotron_parse"]
    for name in names:
        module = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(sys.modules[parent], attr, module)
    sys.modules[names[-1]].NemotronParseForConditionalGeneration = Model
    patch = Path(__file__).parent / "fixtures/parse_2_0/sitecustomize.py.txt"
    runpy.run_path(str(patch))
    patched = Model.load_weights
    runpy.run_path(str(patch))
    assert Model.load_weights is patched
    model = Model()
    model.config = SimpleNamespace(
        tie_word_embeddings=top_tied, decoder=SimpleNamespace(tie_word_embeddings=decoder_tied)
    )
    decoder_weight, head_weight = object(), object()
    model.decoder = SimpleNamespace(embed_tokens=SimpleNamespace(weight=decoder_weight))
    model.lm_head = SimpleNamespace(weight=head_weight)
    weights = [("decoder.embed_tokens.weight", decoder_weight)]
    if explicit:
        weights.append(("lm_head.weight", head_weight))
    assert model.load_weights(iter(weights)) == {name for name, _ in weights}
    assert model.lm_head.weight is (decoder_weight if should_tie else head_weight)


def test_bad_image_does_not_claim_success_or_hide_other_page_success(monkeypatch):
    def decode(payload):
        if payload == "bad":
            raise ValueError("Corrupt page image")
        return object()

    monkeypatch.setattr("nemo_retriever.operators.extract.parse.nemotron_parse._decode_page_image", decode)
    data = pd.DataFrame({"page_image": [{"image_b64": "bad"}, {"image_b64": "good"}], "text": ["Native", ""]})
    out = nemotron_parse_pages(data, model=SimpleNamespace(invoke_batch=lambda *a, **k: [""]), extract_text=True)
    assert [p["inference_status"] for p in out["nemotron_parse_v1_2"]] == ["failed", "succeeded"]
    assert out.at[0, "nemotron_parse_v1_2"]["error"]["message"] == "Corrupt page image"


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "response,status",
    [
        ({}, "failed"),
        ({"choices": []}, "failed"),
        ({"choices": [{"message": {"content": None}}]}, "failed"),
        ({"choices": [{"message": {"content": ""}}]}, "succeeded"),
    ],
)
def test_remote_parse_requires_completed_response_but_accepts_blank_text(monkeypatch, persistent, response, status):
    from nemo_retriever.models.nim.nim import NIMClient

    monkeypatch.setattr("nemo_retriever.models.nim.nim._post_with_retries", lambda **k: response)
    client = NIMClient() if persistent else None
    try:
        out = nemotron_parse_pages(
            pd.DataFrame({"page_image": [{"image_b64": "aW1hZ2U="}], "text": ["Native"]}),
            invoke_url="http://parse:8000/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse-v2.0",
            nim_client=client,
            extract_text=True,
        )
        assert out.at[0, "nemotron_parse_v1_2"]["inference_status"] == status
    finally:
        if client is not None:
            client.shutdown()
