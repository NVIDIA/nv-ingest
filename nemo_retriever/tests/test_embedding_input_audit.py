# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small CPU-only diagnostic contracts; no model weights, network or CUDA calls."""

import base64
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
from PIL import Image

from nemo_retriever.common.params import EmbedInputAuditParams, EmbedParams
from nemo_retriever.common.params.utils import build_embed_option_kwargs
from nemo_retriever.models.inference.input_audit import (
    InputAuditSink,
    _canonical,
    audit_rows,
    current_rows,
    invoke_with_rows,
)
from nemo_retriever.models.inference.runtime import _embed_group
from nemo_retriever.models.inference.shared import build_embed_kwargs
from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder


def sink(tmp_path, **options):
    return InputAuditSink(EmbedInputAuditParams(enabled=True, output_dir=str(tmp_path / "private"), **options))


def documents(value):
    return list(yaml.safe_load_all(value.path.read_text()))


def frame(texts, images=None):
    return pd.DataFrame(
        {
            "text": texts,
            "metadata": [
                {
                    "source_metadata": {"source_id": "private-document"},
                    "content_metadata": {"page_number": i, "type": "text"},
                }
                for i in range(len(texts))
            ],
            "_image_b64": images or [None] * len(texts),
        }
    )


def encoded_image(width, height, *, compress_level=6):
    with Image.new("RGB", (width, height), color=(20, 40, 60)) as image:
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=compress_level)
        return base64.b64encode(buffer.getvalue()).decode()


class Batch(dict):
    def to(self, device):
        assert str(device) == "cpu"
        return Batch(
            {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in self.items()}
        )


class Processor:
    def to_dict(self):
        return {"processor_class": "fake-CPU-only"}

    def tokenizer(self, prompts, **kwargs):
        lengths = [len(prompt.split()) + 1 for prompt in prompts]
        length = max(lengths)
        return Batch(
            input_ids=torch.ones(len(prompts), length, dtype=torch.int64),
            attention_mask=torch.tensor([[1] * n + [0] * (length - n) for n in lengths]),
        )

    def __call__(self, *, text, images, **kwargs):
        batch = self.tokenizer(text)
        height = max(image.height for image in images)
        width = max(image.width for image in images)
        pixels = torch.zeros(len(images), 3, height, width)
        for i, image in enumerate(images):
            values = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width, 3)
            pixels[i, :, : image.height, : image.width] = values.permute(2, 0, 1) / 255
        batch.update(pixel_values=pixels, image_sizes=torch.tensor([[image.height, image.width] for image in images]))
        return batch


class Model(torch.nn.Module):
    device = torch.device("cpu")
    config = SimpleNamespace(
        to_dict=lambda: {"model_type": "fake"}, _commit_hash="fake-revision", _attn_implementation="fake"
    )

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()

    def forward(self, **batch):
        size, tokens = batch["input_ids"].shape
        return SimpleNamespace(last_hidden_state=torch.arange(1, 5).float().expand(size, tokens, 4))


def model():
    result = Nemotron35Embedder(
        "fake-private-model", "fake-revision", "query: ", "passage: ", backend="hf", output_dimension=4
    )
    result._template = (
        "{{ messages[0].content }}{% if messages[1].content is string %}"
        "{{ messages[1].content }}{% else %}{{ messages[1].content[1].text }}{% endif %}"
    )
    result._processor = Processor()
    result._model = Model()
    return result


def embed(data, model, modality="text_image", role="passage"):
    return _embed_group(
        data,
        group_modality=modality,
        model=model,
        endpoint=None,
        api_key=None,
        text_column="text",
        inference_batch_size=8,
        output_column="vectors",
        resolved_model_name="fake-private-model",
        input_type=role,
    )


def test_disabled_path_does_not_select_rows_hash_or_write(tmp_path, monkeypatch):
    class NoFrame:
        @property
        def iloc(self):
            raise AssertionError("disabled diagnostic selected rows")

    assert invoke_with_rows(NoFrame(), lambda: 7, text_column="text", positions=[0]) == 7
    assert current_rows() is None
    assert list(tmp_path.iterdir()) == []


def test_typed_config_and_constructor_identity_are_separate(tmp_path):
    settings = EmbedInputAuditParams(enabled=True, output_dir=str(tmp_path / "private"))
    params = EmbedParams(local_ingest_embed_backend="hf", embed_input_audit=settings)
    restored = EmbedParams.model_validate(params.model_dump())
    assert restored.embed_input_audit == settings
    assert "embed_input_audit" not in build_embed_kwargs(params)
    assert build_embed_option_kwargs(None, None, embed_input_audit=settings)["embed_input_audit"] == settings
    assert EmbedParams().embed_input_audit is None


@pytest.mark.parametrize(
    "values",
    [{"enabled": True}, {"enabled": True, "output_dir": "relative"}, {"chunk_bytes": 9 * 1024**2}, {"max_records": 0}],
)
def test_invalid_diagnostic_contract_rejected(values):
    with pytest.raises(ValueError):
        EmbedInputAuditParams(**values)


def test_private_directory_and_shard_permissions(tmp_path):
    value = sink(tmp_path)
    assert value.path.stat().st_mode & 0o777 == 0o600
    assert value.path.parent.stat().st_mode & 0o777 == 0o700
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(ValueError, match="private directory"):
        InputAuditSink(EmbedInputAuditParams(enabled=True, output_dir=str(public)))


def test_tensor_hash_preserves_bf16_shape_and_logical_order(tmp_path):
    value = sink(tmp_path)
    original = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    assert value.tensor(original.t())["sha256"] == value.tensor(original.t().contiguous())["sha256"]
    assert value.tensor(original)["sha256"] != value.tensor(original.reshape(4, 3))["sha256"]
    assert value.tensor(original)["sha256"] != value.tensor(original.float())["sha256"]
    changed = original.clone()
    changed[0, 0] = 1
    assert value.tensor(original)["sha256"] != value.tensor(changed)["sha256"]


def test_image_compression_and_decoded_rgb_are_distinct_evidence(tmp_path):
    value = sink(tmp_path)
    data = frame(
        ["secret text", "secret text"], [encoded_image(2, 3, compress_level=0), encoded_image(2, 3, compress_level=9)]
    )
    with value.activate(role="passage"):
        embed(data, model())
    images = [event for event in documents(value) if event["event"] == "decoded_image"]
    assert len(images) == 2
    assert images[0]["compressed_sha256"] != images[1]["compressed_sha256"]
    assert images[0]["rgb_sha256"] == images[1]["rgb_sha256"]
    output = value.path.read_text()
    assert "secret text" not in output
    assert "private-document" not in output
    assert data.iloc[0]["_image_b64"] not in output


def test_peer_padding_and_valid_pixels_have_separate_fingerprints(tmp_path):
    value = sink(tmp_path)
    with value.activate(role="passage"):
        embed(frame(["one", "two"], [encoded_image(2, 3), encoded_image(2, 3)]), model())
        embed(frame(["one", "two"], [encoded_image(2, 3), encoded_image(4, 5)]), model())
    processed = [e for e in documents(value) if e["event"] == "processor"]
    assert len(processed) == 2
    assert processed[0]["valid_items"][0]["valid_pixels"] == processed[1]["valid_items"][0]["valid_pixels"]
    assert (
        processed[0]["padded_processor_fields"]["pixel_values"]["sha256"]
        != processed[1]["padded_processor_fields"]["pixel_values"]["sha256"]
    )


def test_mixed_fallback_reassembly_and_handoff_match_final_vectors(tmp_path):
    value = sink(tmp_path)
    data = frame(["first secret", " fallback ", "third secret"], [encoded_image(2, 3), None, encoded_image(4, 5)])
    with value.activate(role="passage"):
        out = embed(data, model())
    events = documents(value)
    assert events[-1]["event"] == "call_end" and events[-1]["complete"] is True
    assert value.records == 3  # Handoff does not admit the same rows twice.
    finals = {
        item["identity"]: item["final"]["sha256"]
        for event in events
        if event["event"] == "model_output"
        for item in event["items"]
    }
    handoff = next(event for event in events if event["event"] == "row_handoff")
    assert len(finals) == len(handoff["items"]) == len(out) == 3
    assert [item["vector"]["sha256"] for item in handoff["items"]] == [
        finals[item["identity"]] for item in handoff["items"]
    ]
    assert all("position_ids" in event["fields"] for event in events if event["event"] == "model_call")


def test_same_warmed_model_uses_independent_call_contexts(tmp_path):
    shared = model()
    left = sink(tmp_path / "left")
    right = sink(tmp_path / "right")
    for value, text in ((left, "private left"), (right, "private right")):
        with value.activate(role="query"):
            embed(frame([text]), shared, modality="text", role="query")
    assert left.path != right.path
    assert current_rows() is None
    assert not hasattr(shared, "_input_audit")
    assert len([e for e in documents(left) if e["event"] == "model_identity"]) == 1
    assert len([e for e in documents(right) if e["event"] == "model_identity"]) == 1


def test_exact_text_and_duplicate_query_positions_remain_distinct(tmp_path):
    value = sink(tmp_path)
    data = frame(["same query", "same query", "same query "])
    data["_nrl_query_index"] = [0, 1, 2]
    with value.activate(role="query"):
        embed(data, model(), modality="text", role="query")
    inputs = next(e for e in documents(value) if e["event"] == "row_inputs")
    assert len({item["identity"] for item in inputs["items"]}) == 3
    assert inputs["items"][0]["selected_text_sha256"] == inputs["items"][1]["selected_text_sha256"]
    assert inputs["items"][1]["selected_text_sha256"] != inputs["items"][2]["selected_text_sha256"]


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_records": 1}, "records"),
        ({"max_hashed_bytes": 1}, "hashed_bytes"),
        ({"max_output_bytes": 4096}, "output_bytes"),
    ],
)
def test_budget_stop_is_explicit_and_does_not_change_embeddings(tmp_path, options, reason):
    value = sink(tmp_path, **options)
    data = frame(["first", "second"])
    with value.activate(role="passage"):
        out = embed(data, model(), modality="text")
    stop = next(e for e in documents(value) if e["event"] == "audit_stopped")
    assert stop["complete"] is False and stop["reason"] == reason
    assert len(out) == 2
    assert value.path.stat().st_size <= value.params.max_output_bytes


def test_capture_error_exports_only_type_not_exception_content(tmp_path):
    value = sink(tmp_path)

    def fail():
        raise ValueError("secret corpus credential do not export")

    value.capture("broken", fail)
    text = value.path.read_text()
    assert "secret corpus" not in text and "credential" not in text
    assert documents(value)[-1]["error_type"] == "ValueError"
    assert documents(value)[-1]["complete"] is False


@pytest.mark.parametrize(
    "config",
    [
        {"embedding_endpoint": "http://private", "local_ingest_embed_backend": "hf"},
        {"local_ingest_embed_backend": "vllm"},
    ],
)
def test_unsupported_paths_rejected_before_model_metadata_resolution(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("should reject before resolving metadata")

    monkeypatch.setattr("nemo_retriever.models.local_embedder_spec.LocalEmbedderSpec.from_config", forbidden)
    with pytest.raises(ValueError, match="local HF"):
        InputAuditSink.validate_model_config(config)


def test_missing_model_output_cannot_claim_complete_evidence(tmp_path):
    value = sink(tmp_path)
    with value.activate(role="passage"):
        with audit_rows(frame(["private text"]), text_column="text"):
            pass  # Deliberately omit the model result in a synthetic failure.
    assert documents(value)[-1]["event"] == "audit_stopped"
    assert documents(value)[-1]["reason"] == "missing_model_outputs"
    assert documents(value)[-1]["complete"] is False


def test_disabled_frame_and_positions_factories_are_never_evaluated():
    def forbidden():
        raise AssertionError("disabled diagnostics allocated frame/positions")

    assert invoke_with_rows(forbidden, lambda: 11, text_column="text", positions=forbidden) == 11


def test_canonical_hash_metadata_uses_stable_yaml_key_order():
    left = {"unicode": "café", "batch": {"shape": [2, 3], "dtype": "torch.bfloat16"}}
    right = {"batch": {"dtype": "torch.bfloat16", "shape": [2, 3]}, "unicode": "café"}
    assert _canonical(left) == _canonical(right)
    assert _canonical(left).startswith(b"---\n")
    assert _canonical(left).endswith(b"...\n")
    assert yaml.safe_load(_canonical(left)) == left


def test_canonical_yaml_ignores_shared_container_aliasing():
    shared = [1, 2]
    assert _canonical({"a": shared, "b": shared}) == _canonical({"a": [1, 2], "b": [1, 2]})
    assert b"&id" not in _canonical({"a": shared, "b": shared})
