# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JudgingOperator -- DataFrame-in/out LLM-as-judge scoring."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, ClassVar, Optional

from nemo_retriever.operators.graph_ops.eval_operator import EvalOperator
from nemo_retriever.models.llm.clients import LLMJudge
from nemo_retriever.models.llm.types import JudgeResult

logger = logging.getLogger(__name__)


class JudgingOperator(EvalOperator):
    """Score candidate answers against reference answers using an LLM judge.

    Input DataFrame must have ``query``, ``reference_answer``, ``answer``.
    Adds columns: ``judge_score``, ``judge_reasoning``, ``judge_error``.
    """

    required_columns: ClassVar[tuple[str, ...]] = ("query", "reference_answer", "answer")
    output_columns: ClassVar[tuple[str, ...]] = ("judge_score", "judge_reasoning", "judge_error")

    def __init__(
        self,
        model: str = "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_params: Optional[dict[str, Any]] = None,
        num_retries: int = 3,
        timeout: float = 120.0,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_workers: int = 8,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            extra_params=extra_params,
            num_retries=num_retries,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
        )
        self._judge = LLMJudge.from_kwargs(
            model=model,
            api_base=api_base,
            api_key=api_key,
            extra_params=extra_params,
            num_retries=num_retries,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._max_workers = max_workers

    def process(self, data: Any, **kwargs: Any) -> Any:
        df = data
        results: list = [None] * len(df)

        def _do_judge(idx: int, query: str, reference: str, answer: str):
            verdict = self._judge.judge(query, reference, answer)
            return idx, verdict

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_do_judge, i, row["query"], row["reference_answer"], row["answer"]): i
                for i, (_, row) in enumerate(df.iterrows())
            }
            for future in as_completed(futures):
                try:
                    idx, verdict = future.result()
                    results[idx] = verdict
                except Exception as exc:
                    idx = futures[future]
                    logger.warning("Row %d judging failed: %s", idx, exc)
                    results[idx] = JudgeResult(score=None, reasoning="", error=str(exc))

        out = df.copy()
        out["judge_score"] = [r.score for r in results]
        out["judge_reasoning"] = [r.reasoning for r in results]
        out["judge_error"] = [r.error for r in results]
        return out
