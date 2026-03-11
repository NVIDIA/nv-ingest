# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local wrapper for nvidia/llama-nemotron-rerank-1b-v2 cross-encoder reranker."""

from __future__ import annotations

from typing import List, Optional

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from ..model import BaseModel, RunMode


_DEFAULT_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
_DEFAULT_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 32


def _prompt_template(query: str, passage: str) -> str:
    """Format a (query, passage) pair as the model expects."""
    return f"question:{query} \n \n passage:{passage}"


class NemotronRerankV2(BaseModel):
    """
    Local cross-encoder reranker wrapping nvidia/llama-nemotron-rerank-1b-v2.

    The model scores (query, document) pairs and returns raw logits; higher
    values indicate greater relevance.  It is fine-tuned from
    meta-llama/Llama-3.2-1B with bi-directional attention and supports 26
    languages with sequences up to 8 192 tokens.

    Example::

        reranker = NemotronRerankV2()
        scores = reranker.score("What is ML?", ["Machine learning is…", "Paris is…"])
        # scores -> [20.6, -23.1]  (higher = more relevant)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        configure_global_hf_cache_base()

        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        kwargs: dict = {"trust_remote_code": True}
        if hf_cache_dir:
            kwargs["cache_dir"] = hf_cache_dir

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            **kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                **kwargs,
            )
            .eval()
            .to(self._device)
        )

        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.eos_token_id

    # ------------------------------------------------------------------
    # BaseModel abstract properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "reranker"

    @property
    def model_runmode(self) -> RunMode:
        return "local"

    @property
    def input(self):
        return "List[Tuple[str, str]]"

    @property
    def output(self):
        return "List[float]"

    @property
    def input_batch_size(self) -> int:
        return _DEFAULT_BATCH_SIZE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        documents: List[str],
        *,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """
        Score relevance of *documents* to *query*.

        Parameters
        ----------
        query:
            The search query.
        documents:
            Candidate passages/documents to score.
        max_length:
            Tokenizer truncation length (default 512; max supported 8 192).
        batch_size:
            Number of (query, doc) pairs to process per GPU forward pass.

        Returns
        -------
        List[float]
            Raw logit scores aligned with *documents* (higher = more relevant).
        """
        import torch

        if not documents:
            return []

        texts = [_prompt_template(query, d) for d in documents]
        all_scores: List[float] = []

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                batch = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = self._model(**batch).logits
                all_scores.extend(logits.view(-1).cpu().tolist())

        return all_scores

    def score_pairs(
        self,
        pairs: List[tuple],
        *,
        max_length: int = _DEFAULT_MAX_LENGTH,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> List[float]:
        """
        Score a list of (query, document) pairs.

        Parameters
        ----------
        pairs:
            Sequence of ``(query, document)`` tuples.
        max_length:
            Tokenizer truncation length.
        batch_size:
            GPU forward-pass batch size.

        Returns
        -------
        List[float]
            Raw logit scores (higher = more relevant).
        """
        import torch

        if not pairs:
            return []

        texts = [_prompt_template(q, d) for q, d in pairs]
        all_scores: List[float] = []

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                batch = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                batch = {k: v.to(self._device) for k, v in batch.items()}
                logits = self._model(**batch).logits
                all_scores.extend(logits.view(-1).cpu().tolist())

        return all_scores
