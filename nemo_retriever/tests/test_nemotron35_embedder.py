# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N3.5 compatibility regressions without network access or model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

from nemo_retriever.models import (
    NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_35_EMBED_BF16_MODEL,
    NEMOTRON_35_EMBED_NVFP4_MODEL,
    resolve_embed_model,
    resolve_local_embed_model,
)
from nemo_retriever.models.embed_model_spec import (
    _spec_from_config,
    validate_embed_model_backend,
)
from nemo_retriever.models.inference.retrieval_template import render_retrieval_input
from nemo_retriever.models.local.nemotron35_embedder import Nemotron35Embedder

TEMPLATE = (
    "{{ messages[0].content.rstrip(' ') ~ ' ' }}{% for m in messages[1:] %}"
    "{% if m.content is string %}{{ m.content | replace('[', '[‌') }}"
    "{% else %}[IMG] {{ m.content[1].text | replace('[', '[‌') }}{% endif %}{% endfor %}"
)


def config(quantized=False):
    return dict(
        model_type="mistral3",
        architectures=["Mistral3Model"],
        pooling="avg",
        text_config=dict(model_type="ministral3", is_causal=False, hidden_size=2048),
        quantization_config=({"quant_method": "compressed-tensors"} if quantized else None),
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_resolve_mistral3_metadata(quantized):
    spec = _spec_from_config(
        "model",
        None,
        config(quantized),
        {"prompts": {"query": "query: ", "document": "passage: "}},
        {"max_seq_length": 32768},
    )
    assert spec.family == "vl" and spec.model_type == "mistral3"
    assert spec.output_dimension == 2048 and spec.max_input_tokens == 32768
    assert spec.requires_vllm == quantized
    validate_embed_model_backend(spec, "vllm")
    if quantized:
        with pytest.raises(ValueError, match="requires backend='vllm'"):
            validate_embed_model_backend(spec, "hf")


def test_reject_causal_text_encoder():
    checkpoint_config = config()
    checkpoint_config["text_config"]["is_causal"] = True
    with pytest.raises(ValueError, match="bidirectional"):
        _spec_from_config("model", None, checkpoint_config)


@pytest.mark.parametrize("prefix", ["query:", "query: "])
def test_template_restores_one_space_and_escapes_user_tokens(prefix):
    assert render_retrieval_input(TEMPLATE, prefix, "text") == "query: text"
    assert render_retrieval_input(TEMPLATE, prefix, "Literal [IMG]") == "query: Literal [‌IMG]"


def test_image_is_structured_and_template_only_applied_once():
    assert render_retrieval_input(TEMPLATE, "passage: ", "passage: text", image=True) == "passage: [IMG] text"


def test_local_defaults_preserve_remote_and_explicit_precisions(monkeypatch):
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: True)
    assert resolve_local_embed_model(None, backend="vllm") == NEMOTRON_35_EMBED_NVFP4_MODEL
    assert resolve_local_embed_model(None, backend="hf") == NEMOTRON_35_EMBED_BF16_MODEL
    assert resolve_local_embed_model(NEMOTRON_35_EMBED_BF16_MODEL, backend="vllm") == NEMOTRON_35_EMBED_BF16_MODEL
    assert resolve_embed_model(None) == NEMOTRON_3_EMBED_MODEL
    monkeypatch.setattr("nemo_retriever.models._cuda_supports_native_nvfp4", lambda: False)
    assert resolve_local_embed_model(None, backend="vllm") == NEMOTRON_35_EMBED_BF16_MODEL


def test_hf_masked_pooling_normalization_and_positions():
    embedder = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="hf")
    embedder._template = TEMPLATE
    seen = []

    class Batch(dict):
        def to(self, device):
            return self

    def tokenize(prompts, **kwargs):
        assert prompts == ["passage: text", "passage: shorter"]
        return Batch(
            input_ids=torch.tensor([[1, 2, 3], [1, 2, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        )

    def forward(**kwargs):
        seen.append(kwargs)
        hidden = torch.ones(2, 3, 2048)
        hidden[1, 2] = 999  # PAD must not influence pooling.
        return SimpleNamespace(last_hidden_state=hidden)

    embedder._processor = SimpleNamespace(tokenizer=tokenize)
    embedder._model = SimpleNamespace(device="cpu", __call__=forward)

    class Model:
        device = "cpu"

        def __call__(self, **kwargs):
            return forward(**kwargs)

    embedder._model = Model()
    out = embedder.embed(["text", "shorter"])
    assert torch.equal(out[0], out[1])
    assert torch.allclose(out.norm(dim=1), torch.ones(2))
    assert seen[0]["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]


def test_invalid_image_counts_fail_before_loading():
    embedder = Nemotron35Embedder("model", None, "query: ", "passage: ")
    with pytest.raises(ValueError, match="counts must match"):
        embedder._encode(["a"], images=[])


@pytest.mark.parametrize("cuda_available", [False, True])
def test_vllm_unload_shuts_down_before_reload(monkeypatch, cuda_available):
    import importlib

    import transformers
    from nemo_retriever.models.inference import nemotron35_vllm, vllm

    embedder_module = importlib.import_module(Nemotron35Embedder.__module__)
    events = []
    engines = []
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="vllm")

    def create_engine(*args, **kwargs):
        engine = MagicMock()
        engine_index = len(engines)
        events.append(f"load-{engine_index}")

        def shutdown(*, timeout):
            assert timeout == 30.0
            assert embedder._model is engine
            events.append(f"shutdown-{engine_index}")

        engine.llm_engine.engine_core.shutdown.side_effect = shutdown
        engine.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=[1.0] * 2048))]
        engines.append(engine)
        return engine

    processor = SimpleNamespace(
        tokenizer=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
    )
    monkeypatch.setattr(embedder_module, "load_retrieval_template", lambda *args: TEMPLATE)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", MagicMock(return_value=processor))
    monkeypatch.setattr(nemotron35_vllm, "register_nemotron35_vllm", lambda: None)
    monkeypatch.setattr(vllm, "create_vllm_llm", create_engine)
    monkeypatch.setattr(vllm, "_pooling_params_for_normalize", lambda normalize: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    first = embedder.embed(["document"])
    embedder.unload()
    assert events == ["load-0", "shutdown-0"]
    assert embedder._model is embedder._processor is embedder._template is None
    assert empty_cache.call_count == int(cuda_available)
    embedder.unload()
    engines[0].llm_engine.engine_core.shutdown.assert_called_once_with(timeout=30.0)
    second = embedder.embed(["document"])
    assert events == ["load-0", "shutdown-0", "load-1"]
    assert embedder._model is engines[1]
    assert first.shape == second.shape == (1, 2048)
    for engine in engines:
        engine.embed.assert_called_once()
    embedder.unload()
    assert events[-1] == "shutdown-1"


def test_vllm_unload_retains_references_if_shutdown_fails(monkeypatch):
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="vllm")
    engine = MagicMock()
    engine.llm_engine.engine_core.shutdown.side_effect = RuntimeError("shutdown failed")
    processor = object()
    embedder._model = engine
    embedder._processor = processor
    embedder._template = TEMPLATE
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        embedder.unload()
    assert embedder._model is engine
    assert embedder._processor is processor
    assert embedder._template == TEMPLATE
    empty_cache.assert_not_called()
    engine.llm_engine.engine_core.shutdown.side_effect = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    embedder.unload()
    assert embedder._model is None


def test_hf_unload_does_not_attempt_vllm_shutdown(monkeypatch):
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="hf")
    model = MagicMock()
    embedder._model = model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    embedder.unload()
    model.llm_engine.engine_core.shutdown.assert_not_called()
    assert embedder._model is None


def test_vllm_receives_rendered_tokens_and_normalizes_once(monkeypatch):
    embedder = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="vllm", normalize=False)
    embedder._template = TEMPLATE
    embedder._processor = SimpleNamespace(
        tokenizer=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
    )
    embedder._model = MagicMock()
    embedder._model.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=[2.0] * 2048))]
    params = object()
    monkeypatch.setattr(
        "nemo_retriever.models.inference.vllm._pooling_params_for_normalize",
        lambda normalize: params,
    )
    result = embedder.embed(["text"])
    assert embedder._model.embed.call_args.args[0] == [{"prompt_token_ids": [1, 2]}]
    assert embedder._model.embed.call_args.kwargs["pooling_params"] is params
    assert torch.equal(result, torch.full((1, 2048), 2.0))


def test_multimodal_expansion_overflow_fails_before_inference():
    embedder = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="vllm", max_length=2)
    embedder._template = TEMPLATE
    embedder._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 3, dtype=torch.long)})
    embedder._model = MagicMock()
    with pytest.raises(ValueError, match="after image expansion"):
        embedder._encode(["text"], images=[object()])
    embedder._model.embed.assert_not_called()
    assert embedder._processor.call_args.kwargs["truncation"] is False


def test_admission_uses_the_same_template():
    from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

    seen = []

    def encode(text, **kwargs):
        seen.append(text)
        return [1, 2, 3]

    policy = EmbeddingInputPolicy(SimpleNamespace(encode=encode), 128, "query: ", retrieval_template=TEMPLATE)
    assert policy._formatted_token_count("Literal [IMG]") == 3
    assert seen == ["query: Literal [‌IMG]"]


def test_admission_does_not_add_special_tokens_twice():
    from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2]
    policy = EmbeddingInputPolicy(tokenizer, 10, "query:", retrieval_template=TEMPLATE, add_special_tokens=False)
    assert policy._formatted_token_count("text") == 2
    tokenizer.encode.assert_called_once_with("query: text", add_special_tokens=False)


@pytest.mark.parametrize("causal", [True, None])
def test_bidirectional_attention_rejects_other_profiles(causal):
    from nemo_retriever.models.inference.nemotron35_vllm import (
        create_bidirectional_attention,
    )

    model = SimpleNamespace(
        config=SimpleNamespace(model_type="mistral3"),
        text_config=SimpleNamespace(model_type="ministral3", is_causal=causal),
    )
    with pytest.raises(ValueError, match="bidirectional"):
        create_bidirectional_attention(
            model,
            attention_cls=MagicMock(),
            attention_type="encoder",
            pp_indices=MagicMock(),
        )


def test_bidirectional_attention_preserves_shards_sliding_window_and_quantization():
    from nemo_retriever.models.inference.nemotron35_vllm import (
        create_bidirectional_attention,
    )

    model = SimpleNamespace(
        config=SimpleNamespace(model_type="mistral3"),
        text_config=SimpleNamespace(
            model_type="ministral3",
            is_causal=False,
            num_hidden_layers=4,
            layer_types=[
                "full_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            sliding_window=128,
        ),
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda parallel: 4,
            get_head_size=lambda: 64,
            get_num_kv_heads=lambda parallel: 2,
        ),
        parallel_config=object(),
        cache_config=object(),
        quant_config=object(),
        pp_group=SimpleNamespace(rank_in_group=1, world_size=2),
    )
    attention = MagicMock(side_effect=lambda **kwargs: kwargs)
    result = create_bidirectional_attention(
        model,
        attention_cls=attention,
        attention_type="encoder_only",
        pp_indices=lambda layers, rank, size: (2, 4),
    )
    assert set(result) == {2, 3}
    assert result[2]["attn_type"] == "encoder_only"
    assert result[2]["per_layer_sliding_window"] == 128
    assert result[3]["per_layer_sliding_window"] is None
    assert result[2]["quant_config"] is model.quant_config


def test_multimodal_admission_splits_losslessly_and_preserves_full_image():
    import pandas as pd
    from nemo_retriever.models.inference.embedding_input import (
        EmbeddingInputPolicy,
        prepare_embedding_inputs,
    )

    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: list(text.encode("ascii")),
        decode=lambda ids, **kwargs: bytes(ids).decode("ascii"),
    )
    # The image contributes five tokens; the template/prefix contributes nine.
    policy = EmbeddingInputPolicy(
        tokenizer,
        18,
        "passage:",
        retrieval_template=TEMPLATE,
        add_special_tokens=False,
        image=True,
        formatted_token_counter=lambda prompt: len(prompt.replace("[IMG] ", "")) + 5,
    )
    source = pd.DataFrame(
        [
            {
                "text": "abcdefghij",
                "_image_b64": "FULL_PAGE",
                "_embed_modality": pd.NA,
                "metadata": {"id": "page"},
            },
        ]
    )
    result = prepare_embedding_inputs(
        source,
        policy=None,
        default_modality="text_image",
        multimodal_planner=lambda text, image: policy.plan([text])[0],
    )
    assert "".join(result.frame.text) == "abcdefghij"
    assert len(result.frame) > 1
    assert result.frame._image_b64.tolist() == ["FULL_PAGE"] * len(result.frame)
    assert all(policy._formatted_token_count(text) <= 18 for text in result.frame.text)
    assert all(meta["embedding_split"]["chunk_count"] == len(result.frame) for meta in result.frame.metadata)
    assert source.iloc[0].metadata == {"id": "page"}
    assert result.split_child_positions == frozenset(range(len(result.frame)))


def test_image_budget_counter_is_used_instead_of_tokenizer_batch_fast_path():
    from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

    tokenizer = MagicMock()
    policy = EmbeddingInputPolicy(tokenizer, 20, "p:", formatted_token_counter=lambda prompt: len(prompt) + 10)
    assert policy.plan(["short"])[0].formatted_tokens == 17
    tokenizer.encode_batch.assert_not_called()


def test_registration_uses_a_unique_lazy_architecture(monkeypatch):
    import sys
    from nemo_retriever.models.inference.nemotron35_vllm import register_nemotron35_vllm

    registry = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    register_nemotron35_vllm()
    registry.register_model.assert_called_once_with(
        "NRLNemotron35EmbeddingModel",
        "nemo_retriever.models.local.nemotron35_vllm:NRLNemotron35EmbeddingModel",
    )


@pytest.mark.parametrize("missing_parent_export", [False, True])
def test_hf_load_disables_causality_on_text_attention_only(monkeypatch, missing_parent_export):
    import importlib

    import transformers

    embedder_module = importlib.import_module(Nemotron35Embedder.__module__)
    if missing_parent_export:
        # Import-safety tests can reload the parent while this child stays cached.
        local_package = importlib.import_module("nemo_retriever.models.local")
        monkeypatch.delattr(local_package, "nemotron35_embedder", raising=False)
    text_attention = SimpleNamespace(is_causal=True)
    vision_attention = SimpleNamespace(is_causal=True)
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.language_model.modules.return_value = [text_attention]
    model.vision_tower = vision_attention
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", MagicMock(return_value=model))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", MagicMock())
    monkeypatch.setattr(
        embedder_module,
        "load_retrieval_template",
        lambda *args: TEMPLATE,
    )
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="hf")
    embedder._ensure_loaded()
    assert text_attention.is_causal is False
    assert vision_attention.is_causal is True


def test_image_without_remaining_text_budget_fails_before_embedding(monkeypatch):
    from nemo_retriever.models.inference import image as image_utils

    image = MagicMock()
    monkeypatch.setattr(image_utils, "decode_base64_image", lambda value: image)
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", max_length=10)
    embedder._model = MagicMock()
    embedder._template = TEMPLATE
    embedder._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 10)})
    with pytest.raises(ValueError, match="no text budget"):
        embedder.plan_text_image("content", "image")
    image.close.assert_called_once()
    embedder._model.embed.assert_not_called()
    assert embedder._processor.call_args.kwargs["truncation"] is False


def test_runtime_runs_multimodal_admission_without_text_policy(monkeypatch):
    import pandas as pd
    from nemo_retriever.models.inference import runtime
    from nemo_retriever.models.inference.main_text_embed import assign_embedding_columns
    from nemo_retriever.models.inference.embedding_input import (
        EmbeddingSplitChild,
        EmbeddingSplitPlan,
    )

    calls = []

    class Model:
        def plan_text_image(self, text, image, *, query):
            calls.append((text, image, query))
            return EmbeddingSplitPlan(20, (EmbeddingSplitChild("abc", 0, 3), EmbeddingSplitChild("def", 3, 6)))

    def embed_group(frame, **kwargs):
        return assign_embedding_columns(
            frame, [[1.0, 0.0]] * len(frame), [None] * len(frame), output_column=kwargs["output_column"]
        )

    monkeypatch.setattr(runtime, "_embed_group", embed_group)
    frame = pd.DataFrame([{"text": "abcdef", "_image_b64": "full_page", "metadata": {}}])
    result = runtime.embed_text_main_text_embed(frame, model=Model(), embed_modality="text_image")
    assert calls == [("abcdef", "full_page", False)]
    assert result.text.tolist() == ["abc", "def"]
    assert result._image_b64.tolist() == ["full_page", "full_page"]
    assert result["embedding"].notna().all()


@pytest.mark.parametrize("endpoint_key", ["embedding_endpoint", "embed_invoke_url"])
def test_remote_default_admission_retains_n3_tokenizer(monkeypatch, endpoint_key):
    from nemo_retriever.models.inference import embedding_input

    resolve = MagicMock()
    monkeypatch.setattr(embedding_input, "resolve_embedding_input_policy", resolve)
    embedding_input.configure_embedding_input_policy({endpoint_key: "https://nim.example/v1"})
    assert resolve.call_args.args[0] == NEMOTRON_3_EMBED_MODEL


def test_vllm_multimodal_disables_implicit_truncation_and_special_tokens(monkeypatch):
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="vllm", max_length=20)
    embedder._template = TEMPLATE
    embedder._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 10)})
    embedder._model = MagicMock()
    embedder._model.embed.return_value = [
        SimpleNamespace(prompt_token_ids=[1] * 10, outputs=SimpleNamespace(embedding=[1.0] * 2048))
    ]
    monkeypatch.setattr("nemo_retriever.models.inference.vllm._pooling_params_for_normalize", lambda normalize: None)
    embedder._encode(["text"], images=[object()])
    assert embedder._processor.call_args.kwargs["truncation"] is False
    assert embedder._model.embed.call_args.kwargs["tokenization_kwargs"] == {
        "add_special_tokens": False,
        "truncation": False,
    }
    request = embedder._model.embed.call_args.args[0][0]
    assert request["mm_processor_kwargs"]["text_kwargs"] == {"add_special_tokens": False, "truncation": False}


def test_vllm_multimodal_rejects_silent_processor_truncation(monkeypatch):
    embedder = Nemotron35Embedder("model", None, "query:", "passage:", backend="vllm", max_length=20)
    embedder._template = TEMPLATE
    embedder._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 10, dtype=torch.long)})
    embedder._model = MagicMock()
    embedder._model.embed.return_value = [
        SimpleNamespace(prompt_token_ids=[1] * 5, outputs=SimpleNamespace(embedding=[1.0] * 2048))
    ]
    monkeypatch.setattr("nemo_retriever.models.inference.vllm._pooling_params_for_normalize", lambda normalize: None)
    with pytest.raises(ValueError, match=r"expected=\[10\], actual=\[5\]"):
        embedder._encode(["text"], images=[object()])
