"""N3.5 compatibility regressions without network access or model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_retriever.models import (
    NEMOTRON_3_EMBED_MODEL,
    NEMOTRON_35_EMBED_BF16_MODEL,
    NEMOTRON_35_EMBED_NVFP4_MODEL,
    resolve_embed_model,
    resolve_local_embed_model,
)
from nemo_retriever.models.embed_model_spec import _spec_from_config, validate_embed_model_backend
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
        quantization_config={"quant_method": "compressed-tensors"} if quantized else None,
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
    c = config()
    c["text_config"]["is_causal"] = True
    with pytest.raises(ValueError, match="bidirectional"):
        _spec_from_config("model", None, c)


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
    e = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="hf")
    e._template = TEMPLATE
    seen = []

    class Batch(dict):
        def to(self, device):
            return self

    def tokenize(prompts, **kwargs):
        assert prompts == ["passage: text", "passage: shorter"]
        return Batch(
            input_ids=torch.tensor([[1, 2, 3], [1, 2, 0]]), attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]])
        )

    def forward(**kwargs):
        seen.append(kwargs)
        hidden = torch.ones(2, 3, 2048)
        hidden[1, 2] = 999  # PAD must not influence pooling.
        return SimpleNamespace(last_hidden_state=hidden)

    e._processor = SimpleNamespace(tokenizer=tokenize)
    e._model = SimpleNamespace(device="cpu", __call__=forward)

    class Model:
        device = "cpu"

        def __call__(self, **kwargs):
            return forward(**kwargs)

    e._model = Model()
    out = e.embed(["text", "shorter"])
    assert torch.equal(out[0], out[1])
    assert torch.allclose(out.norm(dim=1), torch.ones(2))
    assert seen[0]["position_ids"].tolist() == [[0, 1, 2], [0, 1, 0]]


def test_invalid_image_counts_fail_before_loading():
    e = Nemotron35Embedder("model", None, "query: ", "passage: ")
    with pytest.raises(ValueError, match="counts must match"):
        e._encode(["a"], images=[])


def test_vllm_receives_rendered_tokens_and_normalizes_once(monkeypatch):
    e = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="vllm", normalize=False)
    e._template = TEMPLATE
    e._processor = SimpleNamespace(
        tokenizer=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
    )
    e._model = MagicMock()
    e._model.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=[2.0] * 2048))]
    params = object()
    monkeypatch.setattr("nemo_retriever.models.inference.vllm._pooling_params_for_normalize", lambda normalize: params)
    result = e.embed(["text"])
    assert e._model.embed.call_args.args[0] == [{"prompt_token_ids": [1, 2]}]
    assert e._model.embed.call_args.kwargs["pooling_params"] is params
    assert torch.equal(result, torch.full((1, 2048), 2.0))


def test_multimodal_expansion_overflow_fails_before_inference():
    e = Nemotron35Embedder("model", None, "query: ", "passage: ", backend="vllm", max_length=2)
    e._template = TEMPLATE
    e._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 3, dtype=torch.long)})
    e._model = MagicMock()
    with pytest.raises(ValueError, match="after image expansion"):
        e._encode(["text"], images=[object()])
    e._model.embed.assert_not_called()


def test_admission_uses_the_same_template():
    from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy

    seen = []

    def encode(text, **kwargs):
        seen.append(text)
        return [1, 2, 3]

    policy = EmbeddingInputPolicy(SimpleNamespace(encode=encode), 128, "query: ", retrieval_template=TEMPLATE)
    assert policy._formatted_token_count("Literal [IMG]") == 3
    assert seen == ["query: Literal [‌IMG]"]
