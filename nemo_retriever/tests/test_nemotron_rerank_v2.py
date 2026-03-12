# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for NemotronRerankV2 and the rerank module helpers.

All heavy dependencies (torch, transformers, nemo_retriever.utils.hf_cache)
are stubbed via sys.modules injection so no GPU or model download is required.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build lightweight torch / transformers stubs
# ---------------------------------------------------------------------------


def _make_tensor_stub(values: list[float]) -> MagicMock:
    """Return a mock that mimics a 1-D torch.Tensor view(-1).cpu().tolist()."""
    t = MagicMock()
    t.view.return_value = t
    t.cpu.return_value = t
    t.tolist.return_value = values
    return t


def _make_model_output_stub(logits_values: list[float]) -> MagicMock:
    out = MagicMock()
    out.logits = _make_tensor_stub(logits_values)
    return out


def _build_torch_stub() -> MagicMock:
    torch_mod = MagicMock()
    torch_mod.cuda.is_available.return_value = False
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.inference_mode.return_value.__enter__ = lambda s: None
    torch_mod.inference_mode.return_value.__exit__ = MagicMock(return_value=False)
    return torch_mod


def _build_transformers_stub(model_output_values: list[float]) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (transformers_mod, tokenizer_instance, model_instance)."""
    tokenizer_inst = MagicMock()
    tokenizer_inst.pad_token = "pad"
    tokenizer_inst.eos_token_id = 0
    # __call__ on the tokenizer returns a dict of tensors
    tokenizer_inst.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}

    model_inst = MagicMock()
    model_inst.eval.return_value = model_inst
    model_inst.to.return_value = model_inst
    model_inst.config.pad_token_id = 1
    model_inst.return_value = _make_model_output_stub(model_output_values)

    AutoTokenizer = MagicMock()
    AutoTokenizer.from_pretrained.return_value = tokenizer_inst

    AutoModelForSequenceClassification = MagicMock()
    AutoModelForSequenceClassification.from_pretrained.return_value = model_inst

    transformers_mod = MagicMock()
    transformers_mod.AutoTokenizer = AutoTokenizer
    transformers_mod.AutoModelForSequenceClassification = AutoModelForSequenceClassification

    return transformers_mod, tokenizer_inst, model_inst


@pytest.fixture()
def _patch_heavy_deps(monkeypatch):
    """Inject torch + transformers stubs and disable hf_cache setup."""
    torch_stub = _build_torch_stub()
    transformers_stub, tok, mdl = _build_transformers_stub([1.5, -0.3])

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    # Stub hf_cache so configure_global_hf_cache_base() is a no-op.
    hf_cache_mod = ModuleType("nemo_retriever.utils.hf_cache")
    hf_cache_mod.configure_global_hf_cache_base = MagicMock()
    monkeypatch.setitem(sys.modules, "nemo_retriever.utils.hf_cache", hf_cache_mod)

    # Also stub the parent model module so BaseModel import works.
    # We bypass by importing NemotronRerankV2 after patching.
    yield torch_stub, transformers_stub, tok, mdl


# ---------------------------------------------------------------------------
# _prompt_template
# ---------------------------------------------------------------------------


def test_prompt_template_format():
    from nemo_retriever.rerank.rerank import _rerank_via_endpoint  # noqa: F401 — just ensure importable
    from nemo_retriever.model.local.nemotron_rerank_v2 import _prompt_template

    result = _prompt_template("What is ML?", "Machine learning is a branch of AI.")
    assert "question:What is ML?" in result
    assert "passage:Machine learning is a branch of AI." in result


# ---------------------------------------------------------------------------
# NemotronRerankV2 — properties & initialisation
# ---------------------------------------------------------------------------


class TestNemotronRerankV2Properties:
    """Test BaseModel properties without loading real weights."""

    def _make_instance(self, model_name: str = "nvidia/llama-nemotron-rerank-1b-v2") -> object:
        """Instantiate NemotronRerankV2 with all heavy ops mocked out."""
        from nemo_retriever.model.local import nemotron_rerank_v2 as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.AutoTokenizer") as MockTok,
            patch("transformers.AutoModelForSequenceClassification") as MockModel,
        ):
            tok = MockTok.from_pretrained.return_value
            tok.pad_token = "pad"
            tok.eos_token_id = 0
            mdl = MockModel.from_pretrained.return_value
            mdl.eval.return_value = mdl
            mdl.to.return_value = mdl
            mdl.config.pad_token_id = 1
            obj = mod.NemotronRerankV2(model_name=model_name)
        return obj

    def test_model_name(self):
        obj = self._make_instance()
        assert obj.model_name == "nvidia/llama-nemotron-rerank-1b-v2"

    def test_model_type(self):
        obj = self._make_instance()
        assert obj.model_type == "reranker"

    def test_model_runmode(self):
        obj = self._make_instance()
        assert obj.model_runmode == "local"

    def test_input_batch_size(self):
        obj = self._make_instance()
        assert obj.input_batch_size == 32

    def test_custom_model_name_stored(self):
        obj = self._make_instance("my-org/my-reranker")
        assert obj.model_name == "my-org/my-reranker"

    def test_device_defaults_to_cpu_when_no_cuda(self):
        obj = self._make_instance()
        assert obj._device == "cpu"


# ---------------------------------------------------------------------------
# NemotronRerankV2 — score() logic (batch chunking, empty input)
# ---------------------------------------------------------------------------


class TestNemotronRerankV2Score:
    """Test score() and score_pairs() without real model weights."""

    @pytest.fixture()
    def reranker(self):
        from nemo_retriever.model.local import nemotron_rerank_v2 as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.AutoTokenizer") as MockTok,
            patch("transformers.AutoModelForSequenceClassification") as MockModel,
        ):
            tok_inst = MockTok.from_pretrained.return_value
            tok_inst.pad_token = "pad"
            tok_inst.eos_token_id = 0
            mdl_inst = MockModel.from_pretrained.return_value
            mdl_inst.eval.return_value = mdl_inst
            mdl_inst.to.return_value = mdl_inst
            mdl_inst.config.pad_token_id = 1
            obj = mod.NemotronRerankV2()

        return obj

    def test_score_empty_documents_returns_empty(self, reranker):
        assert reranker.score("q", []) == []

    def test_score_pairs_empty_returns_empty(self, reranker):
        assert reranker.score_pairs([]) == []

    def test_score_calls_model_and_returns_flat_list(self, reranker):
        """score() should return one float per document."""
        logit_tensor = MagicMock()
        logit_tensor.view.return_value = logit_tensor
        logit_tensor.cpu.return_value = logit_tensor
        logit_tensor.tolist.return_value = [3.5, -1.2]

        model_out = MagicMock()
        model_out.logits = logit_tensor

        reranker._tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        reranker._model.return_value = model_out

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            scores = reranker.score("What is ML?", ["Machine learning is…", "Paris is…"])

        assert len(scores) == 2
        assert scores == [3.5, -1.2]

    def test_score_prompts_are_formatted_correctly(self, reranker):
        """The tokenizer must receive the templated text, not the raw document."""
        captured_texts = []

        def fake_tokenizer(texts, **kwargs):
            captured_texts.extend(texts)
            m = MagicMock()
            m.items.return_value = []
            return m

        reranker._tokenizer.side_effect = fake_tokenizer

        logit_tensor = MagicMock()
        logit_tensor.view.return_value = logit_tensor
        logit_tensor.cpu.return_value = logit_tensor
        logit_tensor.tolist.return_value = [0.0]

        model_out = MagicMock()
        model_out.logits = logit_tensor
        reranker._model.return_value = model_out

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            reranker.score("my query", ["my document"])

        assert len(captured_texts) == 1
        assert "question:my query" in captured_texts[0]
        assert "passage:my document" in captured_texts[0]

    def test_score_splits_into_batches(self, reranker):
        """With batch_size=2 and 5 documents, model should be called 3 times."""
        call_count = [0]

        def fake_tokenizer(texts, **kwargs):
            m = MagicMock()
            m.items.return_value = [("input_ids", MagicMock())]
            return m

        reranker._tokenizer.side_effect = fake_tokenizer

        def fake_model(**kwargs):
            # Count items in the batch by inspecting how many texts were tokenized
            call_count[0] += 1
            logit_tensor = MagicMock()
            logit_tensor.view.return_value = logit_tensor
            logit_tensor.cpu.return_value = logit_tensor
            logit_tensor.tolist.return_value = [1.0] * 2  # Return 2 scores per call
            out = MagicMock()
            out.logits = logit_tensor
            return out

        reranker._model.side_effect = fake_model

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            # 5 documents, batch_size=2 → ceil(5/2) = 3 forward passes
            reranker.score("q", ["d1", "d2", "d3", "d4", "d5"], batch_size=2)

        assert call_count[0] == 3

    def test_score_pairs_uses_query_per_pair(self, reranker):
        """score_pairs() must use each pair's own query, not a shared one."""
        captured = []

        def fake_tokenizer(texts, **kwargs):
            captured.extend(texts)
            m = MagicMock()
            m.items.return_value = []
            return m

        reranker._tokenizer.side_effect = fake_tokenizer

        logit_tensor = MagicMock()
        logit_tensor.view.return_value = logit_tensor
        logit_tensor.cpu.return_value = logit_tensor
        logit_tensor.tolist.return_value = [0.0, 0.0]

        model_out = MagicMock()
        model_out.logits = logit_tensor
        reranker._model.return_value = model_out

        with patch("torch.inference_mode") as inf_mode:
            inf_mode.return_value.__enter__ = lambda s: None
            inf_mode.return_value.__exit__ = MagicMock(return_value=False)
            reranker.score_pairs([("q1", "doc A"), ("q2", "doc B")])

        assert any("question:q1" in t for t in captured)
        assert any("question:q2" in t for t in captured)


# ---------------------------------------------------------------------------
# rerank_hits() — standalone helper
# ---------------------------------------------------------------------------


class TestRerankHits:
    """Test the public rerank_hits() convenience function."""

    def _make_hits(self, n: int, prefix: str = "doc") -> list[dict]:
        return [{"text": f"{prefix}{i}", "_distance": float(i)} for i in range(n)]

    def test_empty_hits_returns_empty(self):
        from nemo_retriever.rerank import rerank_hits

        model = MagicMock()
        assert rerank_hits("q", [], model=model) == []

    def test_results_sorted_by_score_descending(self):
        from nemo_retriever.rerank import rerank_hits

        hits = self._make_hits(3)
        model = MagicMock()
        model.score.return_value = [0.1, 5.0, -1.0]

        out = rerank_hits("q", hits, model=model)

        scores = [h["_rerank_score"] for h in out]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_score_added_to_each_hit(self):
        from nemo_retriever.rerank import rerank_hits

        hits = [{"text": "hello"}, {"text": "world"}]
        model = MagicMock()
        model.score.return_value = [2.0, 3.0]

        out = rerank_hits("q", hits, model=model)
        assert all("_rerank_score" in h for h in out)

    def test_top_n_truncates_output(self):
        from nemo_retriever.rerank import rerank_hits

        hits = self._make_hits(5)
        model = MagicMock()
        model.score.return_value = [5.0, 4.0, 3.0, 2.0, 1.0]

        out = rerank_hits("q", hits, model=model, top_n=3)
        assert len(out) == 3

    def test_model_score_called_with_query_and_texts(self):
        from nemo_retriever.rerank import rerank_hits

        hits = [{"text": "first"}, {"text": "second"}]
        model = MagicMock()
        model.score.return_value = [1.0, 2.0]

        rerank_hits("my query", hits, model=model)

        model.score.assert_called_once_with("my query", ["first", "second"], max_length=512, batch_size=32)

    def test_raises_without_model_or_endpoint(self):
        from nemo_retriever.rerank import rerank_hits

        with pytest.raises(ValueError, match="model.*invoke_url"):
            rerank_hits("q", [{"text": "doc"}])

    def test_custom_text_key(self):
        from nemo_retriever.rerank import rerank_hits

        hits = [{"content": "alpha"}, {"content": "beta"}]
        model = MagicMock()
        model.score.return_value = [1.0, 2.0]

        out = rerank_hits("q", hits, model=model, text_key="content")
        assert len(out) == 2

    def test_original_hit_keys_preserved(self):
        from nemo_retriever.rerank import rerank_hits

        hits = [{"text": "t", "metadata": "m", "_distance": 0.5}]
        model = MagicMock()
        model.score.return_value = [7.0]

        out = rerank_hits("q", hits, model=model)
        assert out[0]["metadata"] == "m"
        assert out[0]["_distance"] == 0.5


# ---------------------------------------------------------------------------
# _rerank_via_endpoint()
# ---------------------------------------------------------------------------


class TestRerankViaEndpoint:
    def test_posts_to_rerank_url(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            scores = _rerank_via_endpoint(
                "What is ML?",
                ["Machine learning is…", "Paris is…"],
                endpoint="http://localhost:8000",
                model_name="nvidia/llama-nemotron-rerank-1b-v2",
            )

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == "http://localhost:8000/rerank"
        assert call_kwargs[1]["json"]["query"] == "What is ML?"
        assert len(call_kwargs[1]["json"]["documents"]) == 2

        assert scores == [0.9, 0.3]

    def test_scores_aligned_with_input_order(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        # Server returns results in reversed order
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.1},
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.5},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            scores = _rerank_via_endpoint(
                "q",
                ["d0", "d1", "d2"],
                endpoint="http://localhost:8000",
            )

        assert scores[0] == 0.8  # index 0
        assert scores[1] == 0.5  # index 1
        assert scores[2] == 0.1  # index 2

    def test_authorization_header_sent_when_api_key_provided(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 1.0}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint(
                "q",
                ["d"],
                endpoint="http://localhost:8000",
                api_key="my-secret-key",
            )

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    def test_trailing_slash_on_endpoint_normalized(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint("q", ["d"], endpoint="http://localhost:8000/")

        url = mock_post.call_args[0][0]
        assert url == "http://localhost:8000/rerank"

    def test_top_n_sent_in_payload_when_specified(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint("q", ["d"], endpoint="http://localhost:8000", top_n=5)

        payload = mock_post.call_args[1]["json"]
        assert payload["top_n"] == 5

    def test_top_n_not_in_payload_when_not_specified(self):
        from nemo_retriever.rerank.rerank import _rerank_via_endpoint

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            _rerank_via_endpoint("q", ["d"], endpoint="http://localhost:8000")

        payload = mock_post.call_args[1]["json"]
        assert "top_n" not in payload


# ---------------------------------------------------------------------------
# NemotronRerankActor
# ---------------------------------------------------------------------------


class TestNemotronRerankActor:
    """Test the Ray Data-compatible actor."""

    def test_actor_with_invoke_url_skips_local_model(self):
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(invoke_url="http://localhost:8000")
        assert actor._model is None

    def test_actor_with_rerank_invoke_url_alias(self):
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(rerank_invoke_url="http://localhost:8000")
        assert actor._model is None
        assert actor._kwargs.get("invoke_url") == "http://localhost:8000"

    def test_actor_call_scores_dataframe(self):
        import pandas as pd
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(invoke_url="http://localhost:8000")

        df = pd.DataFrame({"query": ["q1", "q2"], "text": ["doc A", "doc B"]})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = [
            {"results": [{"index": 0, "relevance_score": 0.9}]},
            {"results": [{"index": 0, "relevance_score": 0.4}]},
        ]

        with patch("requests.post", return_value=mock_resp):
            out = actor(df)

        assert "rerank_score" in out.columns
        assert len(out) == 2

    def test_actor_call_sorts_descending_by_default(self):
        import pandas as pd
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(invoke_url="http://localhost:8000")
        df = pd.DataFrame({"query": ["q", "q"], "text": ["low relevance", "high relevance"]})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = [
            {"results": [{"index": 0, "relevance_score": 0.1}]},
            {"results": [{"index": 0, "relevance_score": 0.9}]},
        ]

        with patch("requests.post", return_value=mock_resp):
            out = actor(df)

        scores = out["rerank_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_actor_call_returns_error_payload_on_exception(self):
        import pandas as pd
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(invoke_url="http://localhost:8000")
        df = pd.DataFrame({"query": ["q"], "text": ["doc"]})

        with patch("requests.post", side_effect=RuntimeError("connection failed")):
            out = actor(df)

        # Should not raise; should return a DataFrame with error payload
        assert isinstance(out, pd.DataFrame)
        assert "rerank_score" in out.columns
        payload = out["rerank_score"].iloc[0]
        assert payload["status"] == "error"

    def test_actor_custom_score_column_name(self):
        import pandas as pd
        from nemo_retriever.rerank.rerank import NemotronRerankActor

        actor = NemotronRerankActor(invoke_url="http://localhost:8000", score_column="my_score")
        df = pd.DataFrame({"query": ["q"], "text": ["doc"]})

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.7}]}

        with patch("requests.post", return_value=mock_resp):
            out = actor(df)

        assert "my_score" in out.columns
