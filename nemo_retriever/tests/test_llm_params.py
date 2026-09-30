# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified LLM params layer and client / judge composition.

Covers:
    * LLMRemoteClientParams validation and api_key auto-resolution
    * LiteLLMClient(transport, sampling) and .from_kwargs(...) parity
    * top_p omission from litellm call kwargs when unset
    * LLMJudge default sampling and .from_kwargs(...) back-compat
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_litellm_response(text: str = "ok") -> SimpleNamespace:
    """Mimic the litellm.completion() response shape used by LiteLLMClient."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class TestLLMRemoteClientParams:
    """Validate LLMRemoteClientParams validators, defaults, and api_key auto-resolution."""

    def test_defaults(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="nvidia_nim/meta/llama-3.1-70b-instruct")
        assert p.model == "nvidia_nim/meta/llama-3.1-70b-instruct"
        assert p.api_base is None
        assert p.num_retries == 3
        assert p.timeout == 120.0
        assert p.extra_params == {}

    def test_model_is_required(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError):
            LLMRemoteClientParams()  # type: ignore[call-arg]

    def test_negative_num_retries_rejected(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="num_retries must be >= 0"):
            LLMRemoteClientParams(model="m", num_retries=-1)

    def test_zero_timeout_rejected(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="timeout must be > 0"):
            LLMRemoteClientParams(model="m", timeout=0.0)

    def test_negative_timeout_rejected(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="timeout must be > 0"):
            LLMRemoteClientParams(model="m", timeout=-1.0)

    def test_extra_forbid(self):
        """Unknown kwargs should be rejected by _ParamsModel(extra='forbid')."""
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError):
            LLMRemoteClientParams(model="m", unknown_field=123)  # type: ignore[call-arg]

    def test_api_key_none_defers_to_provider_native_lookup(self, monkeypatch):
        """Generic LiteLLM transport must not substitute NVIDIA credentials."""
        from nemo_retriever.common.params import models as params_models

        def unexpected_resolution():
            raise AssertionError("generic transport must not resolve NVIDIA credentials")

        monkeypatch.setattr(params_models, "resolve_remote_api_key", unexpected_resolution)
        p = params_models.LLMRemoteClientParams(model="m")
        assert p.api_key is None

    def test_api_key_no_api_key_sentinel_yields_none(self):
        """Explicit NO_API_KEY sentinel suppresses auto-resolution."""
        from nemo_retriever.common.params.models import (
            NO_API_KEY,
            LLMRemoteClientParams,
        )

        p = LLMRemoteClientParams(model="m", api_key=NO_API_KEY)
        assert p.api_key is None


class TestLiteLLMClientConstruction:
    """LiteLLMClient should accept structured params and expose .model for back-compat."""

    def test_structured_construction(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.common.params.models import (
            LLMInferenceParams,
            LLMRemoteClientParams,
        )

        transport = LLMRemoteClientParams(model="openai/gpt-4o-mini", api_key="k")
        sampling = LLMInferenceParams(temperature=0.2, top_p=0.9, max_tokens=512)
        client = LiteLLMClient(transport=transport, sampling=sampling)

        assert client.transport is transport
        assert client.sampling is sampling
        assert client.model == "openai/gpt-4o-mini"

    def test_default_sampling_matches_from_kwargs_for_rag_determinism(self):
        """``LiteLLMClient`` is a RAG-eval client and must default to the
        same deterministic sampling regardless of which constructor path
        the caller picks.  The structured constructor therefore overrides
        ``LLMInferenceParams``'s general-purpose defaults
        (``temperature=1.0``, ``max_tokens=1024``) with the RAG-tuned
        ``temperature=0.0`` / ``max_tokens=4096`` so it agrees with
        :meth:`LiteLLMClient.from_kwargs`.
        """
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.common.params.models import (
            LLMInferenceParams,
            LLMRemoteClientParams,
        )

        client = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        assert isinstance(client.sampling, LLMInferenceParams)
        assert client.sampling.temperature == 0.0
        assert client.sampling.top_p is None
        assert client.sampling.max_tokens == 4096

    def test_from_kwargs_matches_explicit(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        flat = LiteLLMClient.from_kwargs(
            model="openai/gpt-4o-mini",
            api_key="k",
            temperature=0.3,
            top_p=0.8,
            max_tokens=256,
            num_retries=5,
            timeout=30.0,
            extra_params={"user": "tester"},
        )
        assert flat.transport.model == "openai/gpt-4o-mini"
        assert flat.transport.api_key == "k"
        assert flat.transport.num_retries == 5
        assert flat.transport.timeout == 30.0
        assert flat.transport.extra_params == {"user": "tester"}
        assert flat.sampling.temperature == 0.3
        assert flat.sampling.top_p == 0.8
        assert flat.sampling.max_tokens == 256

    def test_from_kwargs_defaults_top_p_to_none(self):
        """The old flat default of top_p=1.0 is now top_p=None (behavior fix)."""
        from nemo_retriever.models.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs(model="m")
        assert client.sampling.top_p is None

    def test_from_kwargs_preserves_provider_reasoning_by_default(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs(model="m")
        assert client.transport.reasoning_enabled is True


class TestLiteLLMCompleteCallKwargs:
    """Inspect the exact kwargs LiteLLMClient.complete() forwards to litellm."""

    @patch("litellm.completion")
    def test_top_p_omitted_when_none(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="openai/gpt-4o-mini", temperature=0.5)
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o-mini"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 4096
        assert "top_p" not in kwargs
        assert kwargs["num_retries"] == 3
        assert kwargs["timeout"] == 120.0

    @patch("litellm.completion")
    def test_top_p_forwarded_when_set(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="m", top_p=0.9)
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["top_p"] == 0.9

    @patch("litellm.completion")
    def test_max_tokens_override(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="m", max_tokens=4096)
        client.complete([{"role": "user", "content": "hi"}], max_tokens=128)

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["max_tokens"] == 128

    @patch("litellm.completion")
    def test_api_key_and_api_base_forwarded(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(
            model="openai/gpt-4o-mini",
            api_base="http://local-vllm:8000/v1",
            api_key="secret",
        )
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["api_base"] == "http://local-vllm:8000/v1"
        assert kwargs["api_key"] == "secret"

    @patch("litellm.completion")
    def test_allowed_nested_extra_params_merge_recursively(self, mock_completion):
        """Per-request extensions win without discarding sibling values."""
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(
            model="m",
            extra_params={
                "user": "tester",
                "provider": {"seed": 1, "mode": "stable"},
            },
        )
        client.complete(
            [{"role": "user", "content": "hi"}],
            extra_params={
                "provider": {"seed": 2, "extension": True},
                "stop": ["END"],
            },
        )

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["user"] == "tester"
        assert kwargs["provider"] == {
            "seed": 2,
            "mode": "stable",
            "extension": True,
        }
        assert kwargs["stop"] == ["END"]


class TestLiteLLMHardening:
    """Credential, protected-field, and text-only response contracts."""

    @pytest.mark.parametrize(
        "key",
        [
            "model",
            "messages",
            "api_key",
            "api_base",
            "timeout",
            "num_retries",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "functions",
            "function_call",
            "stream",
            "n",
        ],
    )
    def test_transport_rejects_every_protected_extra(self, key):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match=key):
            LLMRemoteClientParams(model="m", extra_params={key: "forbidden"})

    @pytest.mark.parametrize(
        "key",
        [
            "model",
            "messages",
            "api_key",
            "api_base",
            "timeout",
            "num_retries",
            "temperature",
            "top_p",
            "max_tokens",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "functions",
            "function_call",
            "stream",
            "n",
        ],
    )
    @patch("litellm.completion")
    def test_per_request_rejects_every_protected_extra(self, mock_completion, key):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs(model="m")
        with pytest.raises(ValueError, match=key):
            client.complete(
                [{"role": "user", "content": "hi"}],
                extra_params={key: "forbidden"},
            )
        mock_completion.assert_not_called()

    @pytest.mark.parametrize(
        ("model", "environment_name"),
        [
            ("nvidia_nim/model", "NVIDIA_API_KEY"),
            ("openai/model", "OPENAI_API_KEY"),
            ("huggingface/model", "HUGGINGFACE_API_KEY"),
            ("openai/custom", "MY_CUSTOM_PROVIDER_KEY"),
        ],
    )
    @patch("litellm.completion")
    def test_explicit_environment_reference_resolves_at_call_time(
        self,
        mock_completion,
        monkeypatch,
        model,
        environment_name,
    ):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        expected = f"value-for-{environment_name}"
        monkeypatch.setenv(environment_name, expected)
        mock_completion.return_value = _fake_litellm_response("ok")
        client = LiteLLMClient.from_kwargs(
            model=model,
            api_key=f"os.environ/{environment_name}",
        )

        assert client.transport.api_key == f"os.environ/{environment_name}"
        client.complete([{"role": "user", "content": "hi"}])
        assert mock_completion.call_args.kwargs["api_key"] == expected

    @pytest.mark.parametrize("value", [None, ""])
    @patch("litellm.completion")
    def test_missing_or_blank_explicit_environment_reference_fails_before_call(
        self,
        mock_completion,
        monkeypatch,
        value,
    ):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        if value is None:
            monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
        else:
            monkeypatch.setenv("MISSING_PROVIDER_KEY", value)
        client = LiteLLMClient.from_kwargs(
            model="openai/model",
            api_key="os.environ/MISSING_PROVIDER_KEY",
        )

        with pytest.raises(ValueError, match="MISSING_PROVIDER_KEY"):
            client.complete([{"role": "user", "content": "hi"}])
        mock_completion.assert_not_called()

    @patch("litellm.completion")
    def test_none_omits_api_key_and_no_auth_forwards_inert_nonempty_key(
        self,
        mock_completion,
        monkeypatch,
    ):
        from nemo_retriever.common.params.models import NO_API_KEY
        from nemo_retriever.models.llm.clients import LiteLLMClient

        monkeypatch.setenv("NVIDIA_API_KEY", "must-not-be-substituted")
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-substituted-either")
        mock_completion.return_value = _fake_litellm_response("ok")

        provider_native = LiteLLMClient.from_kwargs(model="openai/model")
        provider_native.complete([{"role": "user", "content": "hi"}])
        assert "api_key" not in mock_completion.call_args.kwargs

        no_auth = LiteLLMClient.from_kwargs(model="openai/local", api_key=NO_API_KEY)
        no_auth.complete([{"role": "user", "content": "hi"}])
        forwarded = mock_completion.call_args.kwargs["api_key"]
        assert forwarded
        assert forwarded not in {
            "must-not-be-substituted",
            "must-not-be-substituted-either",
        }

    @pytest.mark.parametrize(
        "reference",
        [
            "os.environ/",
            "os.environ/NOT-VALID",
            "os.environ/9INVALID",
            "os.environ/SPACE KEY",
            " os.environ/OPENAI_API_KEY ",
        ],
    )
    def test_invalid_environment_reference_names_are_rejected(self, reference):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="environment references"):
            LLMRemoteClientParams(model="m", api_key=reference)


class TestLiteLLMRAGPrompt:
    """RAG prompt customization for local OpenAI-compatible answer models."""

    @patch("litellm.completion")
    def test_generate_disables_reasoning_with_portable_request_controls(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("answer")
        client = LiteLLMClient.from_kwargs(
            model="m",
            extra_params={"chat_template_kwargs": {"reasoning_budget": 32}},
            reasoning_enabled=False,
        )
        result = client.generate(query="q", chunks=["ctx"])

        kwargs = mock_completion.call_args.kwargs
        messages = kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("/no_think\n")
        assert "precise question-answering assistant" in messages[0]["content"]
        assert kwargs["chat_template_kwargs"] == {
            "reasoning_budget": 32,
            "enable_thinking": False,
        }
        assert result.answer == "answer"

    @patch("litellm.completion")
    def test_generate_leaves_reasoning_request_defaults_when_enabled(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("answer")
        client = LiteLLMClient.from_kwargs(model="m", reasoning_enabled=True)
        client.generate(query="q", chunks=["ctx"])

        kwargs = mock_completion.call_args.kwargs
        messages = kwargs["messages"]
        assert not messages[0]["content"].startswith("/no_think\n")
        assert "chat_template_kwargs" not in kwargs

    @patch("litellm.completion")
    def test_generate_can_override_reasoning_per_call(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("answer")
        client = LiteLLMClient.from_kwargs(model="m", reasoning_enabled=True)
        client.generate(query="q", chunks=["ctx"], reasoning_enabled=False)

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["messages"][0]["content"].startswith("/no_think\n")
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}

    @patch("litellm.completion")
    def test_generate_uses_custom_rag_system_prompt(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("answer")
        client = LiteLLMClient.from_kwargs(
            model="m",
            rag_system_prompt="Use only context.",
            rag_system_prompt_prefix="/no_think",
        )
        client.generate(query="q", chunks=["ctx"])

        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[0]["content"] == "/no_think\nUse only context."


class TestLLMJudgeConstruction:
    """LLMJudge should use the current Nemotron judge defaults and expose .model."""

    def test_structured_construction_uses_defaults(self):
        from nemo_retriever.models.llm.clients import LLMJudge
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        transport = LLMRemoteClientParams(model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
        judge = LLMJudge(transport=transport)
        assert judge.model == "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
        assert judge.sampling.temperature == 0.1
        assert judge.sampling.max_tokens == 32768

    def test_custom_sampling_override(self):
        from nemo_retriever.models.llm.clients import LLMJudge
        from nemo_retriever.common.params.models import (
            LLMInferenceParams,
            LLMRemoteClientParams,
        )

        transport = LLMRemoteClientParams(model="m")
        sampling = LLMInferenceParams(temperature=0.4, max_tokens=1024)
        judge = LLMJudge(transport=transport, sampling=sampling)
        assert judge.sampling.temperature == 0.4
        assert judge.sampling.max_tokens == 1024

    def test_from_kwargs_matches_structured(self):
        from nemo_retriever.models.llm.clients import LLMJudge

        judge = LLMJudge.from_kwargs(
            model="m",
            api_key="k",
            num_retries=2,
            timeout=60.0,
            extra_params={"user": "t"},
        )
        assert judge.transport.model == "m"
        assert judge.transport.api_key == "k"
        assert judge.transport.num_retries == 2
        assert judge.transport.timeout == 60.0
        assert judge.transport.extra_params == {"user": "t"}
        # Sampling stays at judge defaults even when using flat constructor.
        assert judge.sampling.temperature == 0.1
        assert judge.sampling.max_tokens == 32768

    def test_from_kwargs_accepts_sampling_overrides(self):
        from nemo_retriever.models.llm.clients import LLMJudge

        judge = LLMJudge.from_kwargs(model="m", temperature=0.2, max_tokens=512)

        assert judge.sampling.temperature == 0.2
        assert judge.sampling.max_tokens == 512

    def test_from_kwargs_uses_default_model(self):
        from nemo_retriever.models.llm.clients import LLMJudge

        judge = LLMJudge.from_kwargs()
        assert judge.model == LLMJudge._DEFAULT_MODEL
        assert judge.model == "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"

    @patch("litellm.completion")
    def test_judge_returns_perfect_score(self, mock_completion):
        """Both judges rate 4 -> normalised 1.0 each -> averaged 1.0."""
        from nemo_retriever.models.llm.clients import LLMJudge

        mock_completion.return_value = _fake_litellm_response('{"rating": 4}')
        judge = LLMJudge.from_kwargs(model="m")
        verdict = judge.judge(query="q", reference="ref", candidate="cand")
        assert verdict.score == 1.0
        assert verdict.reasoning == ""
        assert verdict.error is None
        # Dual-judge: exactly two LLM calls per row.
        assert mock_completion.call_count == 2

    @patch("litellm.completion")
    def test_judge_averages_the_two_judges(self, mock_completion):
        """Judge 1 -> 4 (1.0), judge 2 -> 2 (0.5); average is 0.75."""
        from nemo_retriever.models.llm.clients import LLMJudge

        mock_completion.side_effect = [
            _fake_litellm_response('{"rating": 4}'),
            _fake_litellm_response('{"rating": 2}'),
        ]
        judge = LLMJudge.from_kwargs(model="m")
        verdict = judge.judge(query="q", reference="ref", candidate="cand")
        assert verdict.score == 0.75
        assert verdict.error is None

    @patch("litellm.completion")
    def test_judge_unparseable_rating_becomes_none(self, mock_completion):
        """When neither judge yields a valid 0/2/4 rating, score is None with an error."""
        from nemo_retriever.models.llm.clients import LLMJudge

        mock_completion.return_value = _fake_litellm_response("I cannot rate this answer.")
        judge = LLMJudge.from_kwargs(model="m", num_retries=2)
        verdict = judge.judge(query="q", reference="ref", candidate="cand")
        assert verdict.score is None
        assert verdict.error is not None and "judge_no_score" in verdict.error

    @patch("litellm.completion")
    def test_judge_transport_error_surfaced(self, mock_completion):
        """When every attempt raises, the last transport error is in the JudgeResult."""
        from nemo_retriever.models.llm.clients import LLMJudge

        mock_completion.side_effect = RuntimeError("connection refused")
        judge = LLMJudge.from_kwargs(model="m", num_retries=2)
        verdict = judge.judge(query="q", reference="ref", candidate="cand")
        assert verdict.score is None
        assert "judge_no_score" in verdict.error
        assert "connection refused" in verdict.error

    def test_judge_empty_candidate_short_circuits(self):
        """Empty candidate is handled locally with no LLM call."""
        from nemo_retriever.models.llm.clients import LLMJudge

        with patch("litellm.completion") as mock_completion:
            judge = LLMJudge.from_kwargs(model="m")
            verdict = judge.judge(query="q", reference="r", candidate="   ")
            mock_completion.assert_not_called()

        assert verdict.score is None
        assert verdict.error == "empty_candidate"


class TestBackCompatCallSites:
    """The four migrated call sites all use .from_kwargs, so they must still work."""

    @patch("litellm.completion")
    def test_qa_generation_operator_constructs_cleanly(self, mock_completion):
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        mock_completion.return_value = _fake_litellm_response("answer")
        op = QAGenerationOperator(model="m", temperature=0.0, max_tokens=128)
        assert op._client.transport.model == "m"
        assert op._client.sampling.temperature == 0.0
        assert op._client.sampling.max_tokens == 128

    def test_qa_generation_operator_forwards_reasoning_enabled(self):
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        op = QAGenerationOperator(model="m", reasoning_enabled=True)
        assert op._client.transport.reasoning_enabled is True

    def test_pipeline_builder_generate_forwards_reasoning_enabled(self):
        from unittest.mock import MagicMock

        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.graph.retriever import RetrieverPipelineBuilder

        retriever = MagicMock()
        retriever.top_k = 5
        builder = RetrieverPipelineBuilder(retriever, top_k=5)

        llm = LiteLLMClient.from_kwargs(model="m", reasoning_enabled=True)
        builder.generate(llm)

        generation_ops = [s for s in builder._steps if isinstance(s, QAGenerationOperator)]
        assert len(generation_ops) == 1
        assert generation_ops[0]._client.transport.reasoning_enabled is True

    def test_pipeline_builder_generate_defaults_reasoning_enabled_for_legacy_client(
        self,
    ):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator
        from nemo_retriever.graph.retriever import RetrieverPipelineBuilder

        retriever = MagicMock()
        retriever.top_k = 5
        builder = RetrieverPipelineBuilder(retriever, top_k=5)
        llm = SimpleNamespace(
            transport=SimpleNamespace(
                model="m",
                api_base=None,
                api_key=None,
                extra_params={},
                num_retries=3,
                timeout=120.0,
                rag_system_prompt=None,
                rag_system_prompt_prefix=None,
            ),
            sampling=SimpleNamespace(temperature=0.0, top_p=None, max_tokens=128),
        )

        builder.generate(llm)

        generation_ops = [s for s in builder._steps if isinstance(s, QAGenerationOperator)]
        assert len(generation_ops) == 1
        assert generation_ops[0]._client.transport.reasoning_enabled is True

    def test_judging_operator_constructs_cleanly(self):
        from nemo_retriever.tools.evaluation.judging import JudgingOperator

        op = JudgingOperator(model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")
        assert op._judge.model == "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
        assert op._judge.sampling.temperature == 0.1
        assert op._judge.sampling.max_tokens == 32768

    def test_judging_operator_plumbs_num_retries_to_inner_judge(self):
        """JudgingOperator(num_retries=...) must flow down to the LLMJudge it
        instantiates internally.

        Before this fix, ``JudgingOperator.__init__`` had no ``num_retries``
        parameter, so the pre-built ``LLMJudge.transport.num_retries`` set by
        a pipeline caller was silently dropped at the operator boundary and
        the operator always ran with ``LLMJudge``'s default (3)."""
        from nemo_retriever.tools.evaluation.judging import JudgingOperator

        op = JudgingOperator(
            model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
            num_retries=7,
        )
        assert op._judge.transport.num_retries == 7

    def test_pipeline_builder_judge_forwards_transport_num_retries(self):
        """RetrieverPipelineBuilder.judge(judge) unpacks transport.* onto the
        operator. num_retries must be in that unpack, symmetric with the
        identical .generate() branch at retriever.py:762."""
        from unittest.mock import MagicMock

        from nemo_retriever.tools.evaluation.judging import JudgingOperator
        from nemo_retriever.models.llm.clients import LLMJudge
        from nemo_retriever.graph.retriever import RetrieverPipelineBuilder

        retriever = MagicMock()
        retriever.top_k = 5
        builder = RetrieverPipelineBuilder(retriever, top_k=5)

        judge = LLMJudge.from_kwargs(
            model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
            num_retries=7,
        )
        builder.judge(judge)

        judging_ops = [s for s in builder._steps if isinstance(s, JudgingOperator)]
        assert len(judging_ops) == 1
        assert judging_ops[0]._judge.transport.num_retries == 7

    def test_pipeline_builder_judge_defaults_num_retries_when_flat_kwargs(self):
        """The flat ``model=...`` branch of .judge() must still default
        num_retries to 3, preserving the current default behaviour."""
        from unittest.mock import MagicMock

        from nemo_retriever.tools.evaluation.judging import JudgingOperator
        from nemo_retriever.graph.retriever import RetrieverPipelineBuilder

        retriever = MagicMock()
        retriever.top_k = 5
        builder = RetrieverPipelineBuilder(retriever, top_k=5)

        builder.judge(model="nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b")

        judging_ops = [s for s in builder._steps if isinstance(s, JudgingOperator)]
        assert len(judging_ops) == 1
        assert judging_ops[0]._judge.transport.num_retries == 3


class TestApiKeyRedaction:
    """Guard the repr/str of every transport params object against key leakage.

    The ``_ParamsModel`` base redacts ``api_key`` + ``*_api_key`` fields in
    ``__repr__`` / ``__str__`` so that logging a transport object (or
    letting Pydantic's default error formatter echo one back) never
    prints a bearer token.  Consumers still read the plain ``str`` via
    attribute access, so no downstream litellm/NIM call is affected.
    """

    def test_api_key_masked_in_repr(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        rendered = repr(p)
        assert "nvapi-SECRET-TOKEN" not in rendered
        assert "api_key=***" in rendered

    def test_api_key_masked_in_str(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        assert "nvapi-SECRET-TOKEN" not in str(p)

    def test_api_key_attribute_is_plain_str(self):
        """Redaction is display-only -- attribute access still yields the raw string."""
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        assert p.api_key == "nvapi-SECRET-TOKEN"
        assert isinstance(p.api_key, str)

    def test_empty_api_key_not_masked(self):
        """Redaction only fires when a key is actually present."""
        from nemo_retriever.common.params.models import (
            NO_API_KEY,
            LLMRemoteClientParams,
        )

        p = LLMRemoteClientParams(model="m", api_key=NO_API_KEY)
        assert p.api_key is None
        assert "api_key=***" not in repr(p)
        assert "api_key=None" in repr(p)

    @patch("litellm.completion")
    def test_plain_str_reaches_litellm_call_site(self, mock_completion):
        """The redacted __repr__ must not break the wire-format contract."""
        from nemo_retriever.models.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("ok")
        client = LiteLLMClient.from_kwargs(model="m", api_key="nvapi-SECRET-TOKEN")
        client.generate(query="q", chunks=[])

        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["api_key"] == "nvapi-SECRET-TOKEN"
        assert isinstance(call_kwargs["api_key"], str)

    def test_nested_api_key_fields_also_masked(self):
        """Fields matching *_api_key (not only bare api_key) get redacted."""
        from nemo_retriever.common.params.models import ExtractParams

        p = ExtractParams(
            page_elements_api_key="nvapi-PAGE-ELEM-TOKEN",
            ocr_api_key="nvapi-OCR-TOKEN",
        )
        rendered = repr(p)
        assert "nvapi-PAGE-ELEM-TOKEN" not in rendered
        assert "nvapi-OCR-TOKEN" not in rendered
        assert "page_elements_api_key=***" in rendered
        assert "ocr_api_key=***" in rendered

    def test_nested_extra_param_secrets_are_redacted_from_repr(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        secret = "sk-NESTED-MUST-NOT-LEAK"
        params = LLMRemoteClientParams(
            model="m",
            extra_params={
                "provider": {"api_key": secret},
                "authorization": f"Bearer {secret}",
                "authorizationHeader": f"Bearer {secret}",
            },
        )

        rendered = repr(params)
        assert secret not in rendered
        assert "***" in rendered

    def test_protected_extra_validation_error_hides_raw_input(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        secret = "sk-VALIDATION-MUST-NOT-LEAK"
        with pytest.raises(ValueError, match="protected request fields") as exc_info:
            LLMRemoteClientParams(
                model="m",
                extra_params={"api_key": secret},
            )

        assert secret not in str(exc_info.value)

    def test_mutated_literal_invalidates_environment_provenance(self):
        from nemo_retriever.common.params.models import LLMRemoteClientParams

        params = LLMRemoteClientParams(
            model="m",
            api_key="os.environ/OPENAI_API_KEY",
        )
        params.api_key = "sk-LITERAL-MUTATION"

        assert params._api_key_env_reference("api_key") is None


class TestLiteLLMDefaultModel:
    """Mirror of LLMJudge._DEFAULT_MODEL coverage for LiteLLMClient."""

    def test_from_kwargs_uses_default_model(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs()
        assert client.model == LiteLLMClient._DEFAULT_MODEL
        assert client.model == "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"

    def test_default_model_is_a_non_empty_string(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient

        assert isinstance(LiteLLMClient._DEFAULT_MODEL, str)
        assert LiteLLMClient._DEFAULT_MODEL


class TestLiteLLMDefaultSamplingAlignment:
    """Both constructor paths must default to the same deterministic sampling.

    Regression test for the Greptile P1 finding that
    ``LiteLLMClient(transport=...)`` with ``sampling=None`` silently
    fell through to ``LLMInferenceParams()`` (``temperature=1.0``) while
    ``LiteLLMClient.from_kwargs(...)`` explicitly defaulted to
    ``temperature=0.0``.  For RAG-eval reproducibility the two paths
    must converge on the same default.
    """

    def test_structured_constructor_defaults_to_zero_temperature(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.common.params import LLMRemoteClientParams

        client = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        assert client.sampling.temperature == 0.0

    def test_structured_and_flat_paths_agree_on_defaults(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.common.params import LLMRemoteClientParams

        structured = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        flat = LiteLLMClient.from_kwargs(model="m")
        assert structured.sampling.temperature == flat.sampling.temperature
        assert structured.sampling.max_tokens == flat.sampling.max_tokens
        assert structured.sampling.top_p == flat.sampling.top_p

    def test_explicit_sampling_is_not_overridden(self):
        """Passing an explicit ``LLMInferenceParams`` must win over the default."""
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.common.params import (
            LLMInferenceParams,
            LLMRemoteClientParams,
        )

        client = LiteLLMClient(
            transport=LLMRemoteClientParams(model="m"),
            sampling=LLMInferenceParams(temperature=0.7),
        )
        assert client.sampling.temperature == 0.7
