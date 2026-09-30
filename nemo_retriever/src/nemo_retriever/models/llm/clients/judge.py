# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM-as-judge scoring -- a from-scratch port of ragas' ``AnswerAccuracy``.

``LLMJudge`` reproduces the dual-judge logic of ragas'
:class:`ragas.metrics.collections.AnswerAccuracy` directly on
:class:`~nemo_retriever.models.llm.clients.litellm.LiteLLMClient`, with no ragas /
instructor / openai dependency:

* Two paraphrased judge prompts (verbatim from ragas), each rating the answer
  against the reference on a ``0/2/4`` scale via few-shot prompting.
* Judge 2 evaluates with the reference/candidate roles **swapped**, giving a
  bidirectional equivalence check.
* Each rating is normalised (``/4``) and the two are averaged (NaN-aware),
  yielding a final score on a ``0.0-1.0`` scale (higher is better).

The prompts, few-shot examples, and prompt-string layout mirror ragas'
``AnswerAccuracyJudge1Prompt`` / ``AnswerAccuracyJudge2Prompt`` and
``BasePrompt.to_string`` so behaviour tracks the upstream metric. ``instructor``
constrained decoding is replaced by prompt-instructed JSON plus a tolerant
parser (the prompts already demand ``{"rating": X}``).
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Optional

from nemo_retriever.models.llm.clients.litellm import LiteLLMClient
from nemo_retriever.models.llm.types import JudgeResult
from nemo_retriever.common.params.models import LLMInferenceParams, LLMRemoteClientParams

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompts -- verbatim from ragas AnswerAccuracy (util.py), rendered through a
# faithful reproduction of ragas BasePrompt.to_string.
# --------------------------------------------------------------------------- #

_JUDGE1_INSTRUCTION = (
    "You are a world class state of the art assistant for rating a User Answer given a Question. "
    "The Question is completely answered by the Reference Answer.\n"
    "Say 4, if User Answer is full contained and equivalent to Reference Answer in all terms, "
    "topics, numbers, metrics, dates and units.\n"
    "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer in all "
    "terms, topics, numbers, metrics, dates and units.\n"
    "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, "
    "topics, numbers, metrics, dates and units or the User Answer do not answer the question.\n"
    "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the "
    "instructions above.\n"
    'Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4.'
)

_JUDGE2_INSTRUCTION = (
    "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
    "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, "
    "covering all aspects, topics, numbers, metrics, dates, and units.\n"
    "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with "
    "minor discrepancies in some areas.\n"
    "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the "
    "Reference Answer, or it fails to address the Question.\n"
    "I will provide the rating without any explanation or justification, adhering to the following "
    "scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
    "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n"
    'Return your response as JSON in this format: {"rating": X} where X is 0, 2, or 4.'
)

_JUDGE1_EXAMPLES: tuple[tuple[dict[str, str], int], ...] = (
    (
        {
            "query": "When was Albert Einstein born?",
            "user_answer": "Albert Einstein was born in 1879.",
            "reference_answer": "Albert Einstein was born on March 14, 1879.",
        },
        2,
    ),
    (
        {
            "query": "What is the capital of France?",
            "user_answer": "Paris is the capital of France.",
            "reference_answer": "Paris is the capital of France.",
        },
        4,
    ),
    (
        {
            "query": "What is the highest mountain?",
            "user_answer": "The Eiffel Tower is a famous landmark.",
            "reference_answer": "Mount Everest is the highest mountain.",
        },
        0,
    ),
)

_JUDGE2_EXAMPLES: tuple[tuple[dict[str, str], int], ...] = (
    (
        {
            "query": "When was Albert Einstein born?",
            "user_answer": "Einstein was born in 1879 in Germany.",
            "reference_answer": "Albert Einstein was born on March 14, 1879 in Ulm, Germany.",
        },
        2,
    ),
    (
        {
            "query": "What is the capital of France?",
            "user_answer": "The capital of France is Paris.",
            "reference_answer": "Paris is the capital of France.",
        },
        4,
    ),
    (
        {
            "query": "What is the speed of light?",
            "user_answer": "The sun is a star.",
            "reference_answer": "The speed of light is approximately 299,792,458 meters per second.",
        },
        0,
    ),
)

# json.dumps(AnswerAccuracyOutput.model_json_schema()) -- the rating-only schema
# (the leading description is the ragas model's docstring).
_OUTPUT_SCHEMA = (
    '{"description": "Structured output for answer accuracy evaluation.", '
    '"properties": {"rating": {"description": "Accuracy rating (0, 2, or 4)", '
    '"title": "Rating", "type": "integer"}}, "required": ["rating"], '
    '"title": "AnswerAccuracyOutput", "type": "object"}'
)

# Valid ratings the judge may emit, and their normalised 0.0-1.0 values.
_VALID_RATINGS = (0, 2, 4)


def _render_examples(examples: tuple[tuple[dict[str, str], int], ...]) -> str:
    """Reproduce ragas BasePrompt._generate_examples for the given few-shots."""
    blocks = [
        "Example {n}\nInput: {inp}\nOutput: {out}".format(
            n=i + 1,
            inp=json.dumps(inp, indent=4),
            out=json.dumps({"rating": rating}, indent=4),
        )
        for i, (inp, rating) in enumerate(examples)
    ]
    return "--------EXAMPLES-----------\n" + "\n\n".join(blocks)


def _build_prefix(instruction: str, examples: tuple[tuple[dict[str, str], int], ...]) -> str:
    """Reproduce the static head of ragas BasePrompt.to_string (everything before the input)."""
    return (
        f"{instruction}\n"
        "Please return the output in a JSON format that complies with the following schema as "
        "specified in JSON Schema:\n"
        f"{_OUTPUT_SCHEMA}Do not use single quotes in your response but double quotes,properly "
        "escaped with a backslash.\n\n"
        f"{_render_examples(examples)}\n"
        "-----------------------------\n\n"
        "Now perform the same with the following input\n"
    )


_JUDGE1_PREFIX = _build_prefix(_JUDGE1_INSTRUCTION, _JUDGE1_EXAMPLES)
_JUDGE2_PREFIX = _build_prefix(_JUDGE2_INSTRUCTION, _JUDGE2_EXAMPLES)


def _render_prompt(prefix: str, query: str, user_answer: str, reference_answer: str) -> str:
    """Fill the input block exactly as ragas BasePrompt.to_string does."""
    input_json = json.dumps(
        {"query": query, "user_answer": user_answer, "reference_answer": reference_answer},
        indent=4,
    )
    return f"{prefix}input: {input_json}\nOutput: "


def _parse_rating(raw: str) -> Optional[int]:
    """Extract the integer ``rating`` from a judge response, or ``None``."""
    text = raw.strip()
    # Reasoning models (e.g. Nemotron) may emit a <think>...</think> block first.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
        return int(data["rating"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    match = re.search(r'"rating"\s*:\s*([0-9]+)', text)
    if match:
        return int(match.group(1))
    # Last resort: a lone 0/2/4 token.
    match = re.search(r"\b([024])\b", text)
    if match:
        return int(match.group(1))
    return None


def _average_scores(score1: float, score2: float) -> float:
    """Average two normalised judge scores, preferring a valid score over NaN."""
    n1, n2 = math.isnan(score1), math.isnan(score2)
    if not n1 and not n2:
        return (score1 + score2) / 2.0
    if not n1:
        return score1
    if not n2:
        return score2
    return float("nan")


class LLMJudge:
    """LLM-as-judge that scores candidate answers on a ``0.0-1.0`` scale.

    Ports ragas' dual-judge ``AnswerAccuracy`` onto ``LiteLLMClient``. Two
    paraphrased judges each rate the answer ``0/2/4`` against the reference --
    the second with reference/candidate roles swapped -- and the normalised
    ratings are averaged.

    Configuration is split into two Pydantic objects:

    * ``transport``: :class:`~nemo_retriever.common.params.LLMRemoteClientParams` owns
      the endpoint, api_key, retries, and timeout. ``num_retries`` is reused as
      the per-judge attempt budget for obtaining a valid ``0/2/4`` rating.
    * ``sampling``: :class:`~nemo_retriever.common.params.LLMInferenceParams` owns
      ``temperature`` / ``top_p`` / ``max_tokens``. Defaults to
      ``temperature=0.1, max_tokens=4096`` for judge consistency.

    Use :meth:`from_kwargs` for a flat, backwards-compatible constructor.
    """

    _DEFAULT_MODEL: str = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
    # max_tokens must accommodate the Nemotron reasoning block + the final
    # {"rating": X}; NVIDIA's llm-judge recipe uses 32768. 4096 truncated mid-think.
    _DEFAULT_SAMPLING: LLMInferenceParams = LLMInferenceParams(temperature=0.1, max_tokens=32768)

    def __init__(
        self,
        transport: LLMRemoteClientParams,
        sampling: Optional[LLMInferenceParams] = None,
    ):
        self.transport = transport
        self.sampling = sampling if sampling is not None else self._DEFAULT_SAMPLING
        self._client = LiteLLMClient(transport=transport, sampling=self.sampling)

    @property
    def model(self) -> str:
        """Return the judge model identifier from the transport params."""
        return self.transport.model

    @classmethod
    def from_kwargs(
        cls,
        *,
        model: str = _DEFAULT_MODEL,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "LLMJudge":
        """Flat-kwarg constructor for zero-churn migration from the old signature.

        Sampling is left at the class default unless ``temperature`` or
        ``max_tokens`` is supplied. Use the two-arg constructor to override
        the full sampling object.
        """
        transport = LLMRemoteClientParams(
            model=model,
            api_base=api_base,
            api_key=api_key,
            num_retries=num_retries,
            timeout=timeout,
            extra_params=extra_params or {},
        )
        sampling = None
        if temperature is not None or max_tokens is not None:
            sampling = LLMInferenceParams(
                temperature=cls._DEFAULT_SAMPLING.temperature if temperature is None else temperature,
                top_p=cls._DEFAULT_SAMPLING.top_p,
                max_tokens=cls._DEFAULT_SAMPLING.max_tokens if max_tokens is None else max_tokens,
            )
        return cls(transport=transport, sampling=sampling)

    def _rate(self, prefix: str, query: str, user_answer: str, reference_answer: str) -> tuple[float, Optional[str]]:
        """Run one judge prompt.

        Returns ``(score, error)`` where ``score`` is the normalised
        0.0/0.5/1.0 rating (or NaN) and ``error`` is the last transport error
        string when every attempt failed, else ``None``. Mirrors ragas
        ``_get_judge_rating``: retry on an invalid rating or a transport error
        up to ``num_retries`` attempts, then give up with NaN.
        """
        # The prompt already forbids explanation and demands `{"rating": X}`, and
        # _parse_rating strips any <think> block — i.e. the judge is designed for the
        # Nemotron reasoning model. The only requirement is a large enough max_tokens
        # for the reasoning block to finish and still emit the rating (NVIDIA's
        # llm-judge recipe uses 32768); 4096 truncated mid-think -> null content.
        messages = [{"role": "user", "content": _render_prompt(prefix, query, user_answer, reference_answer)}]
        attempts = max(1, self.transport.num_retries)
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                raw, _ = self._client.complete(messages)
            except Exception as exc:  # noqa: BLE001 - retried; surfaced below if all attempts fail
                last_exc = exc
                logger.warning("Judge transport error on attempt %d/%d: %s", attempt + 1, attempts, exc)
                continue
            rating = _parse_rating(raw)
            if rating in _VALID_RATINGS:
                return rating / 4.0, None
        if last_exc is not None:
            logger.warning("All %d judge attempts failed; last error: %s", attempts, last_exc)
            return float("nan"), str(last_exc)
        # Attempts succeeded at the transport layer but never produced a valid rating.
        return float("nan"), None

    def judge(self, query: str, reference: str, candidate: str) -> JudgeResult:
        """Score a candidate answer against the reference answer (0.0-1.0)."""
        if not candidate or not candidate.strip():
            return JudgeResult(score=None, reasoning="Candidate answer was empty.", error="empty_candidate")

        try:
            # Judge 1: candidate as the user answer, reference as ground truth.
            rating1, err1 = self._rate(_JUDGE1_PREFIX, query, candidate, reference)
            # Judge 2: roles swapped (bidirectional check) under the paraphrased rubric.
            rating2, err2 = self._rate(_JUDGE2_PREFIX, query, reference, candidate)
        except Exception as exc:
            return JudgeResult(score=None, reasoning="", error=f"judge_api_error: {exc}")

        score = _average_scores(rating1, rating2)
        if math.isnan(score):
            transport_err = err1 or err2
            error = "judge_no_score: neither judge produced a valid rating"
            if transport_err is not None:
                error += f" (last transport error: {transport_err})"
            return JudgeResult(score=None, reasoning="", error=error)
        # AnswerAccuracy emits only a numeric rating, so there is no rationale.
        return JudgeResult(score=float(score), reasoning="")
