# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision


def _prompt_template(query: str, passage: str) -> str:
    """Format query and passage into the prompt expected by the reranker."""
    return f"question:{query} \n \n passage:{passage}"


@dataclass
class LlamaNemotronReranker1BV2:
    """
    Minimal reranker wrapper for local-only HuggingFace execution of
    ``nvidia/llama-nemotron-rerank-1b-v2``.

    Produces a relevance score (raw logit) for each (query, document) pair.
    This intentionally contains **no remote invocation logic**.
    """

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    max_length: int = 8192
    model_id: Optional[str] = None

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_id = self.model_id or "nvidia/llama-nemotron-rerank-1b-v2"
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        hf_cache_dir = configure_global_hf_cache_base(self.hf_cache_dir)
        _revision = get_hf_revision(model_id)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.eos_token_id

        self._model = self._model.to(dev)
        self._model.eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int = 64,
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Score documents against *query* and return ``(index, score)`` pairs
        sorted by descending relevance.

        Parameters
        ----------
        query:
            The search query string.
        documents:
            Candidate passages to score against *query*.
        batch_size:
            Tokenisation / inference batch size.
        top_n:
            If given, return only the top-n results.

        Returns
        -------
        list[tuple[int, float]]
            ``(original_index, raw_logit_score)`` pairs sorted best-first.
        """
        if not documents:
            return []

        scores = self._score_pairs(query, list(documents), batch_size=batch_size)

        indexed = sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
        if top_n is not None:
            indexed = indexed[: int(top_n)]
        return indexed

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> List[float]:
        """Return raw relevance scores (logits) for each document, preserving
        the original document order."""
        if not documents:
            return []
        return self._score_pairs(query, list(documents), batch_size=batch_size)

    def _score_pairs(
        self,
        query: str,
        documents: List[str],
        *,
        batch_size: int,
    ) -> List[float]:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Local reranker was not initialized.")

        dev = self._device
        texts = [_prompt_template(query, doc) for doc in documents]
        all_scores: List[float] = []

        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`input_embeds` is deprecated", category=FutureWarning)
            for i in range(0, len(texts), max(1, int(batch_size))):
                chunk = texts[i : i + max(1, int(batch_size))]
                batch = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max(1, int(self.max_length)),
                    return_tensors="pt",
                ).to(dev)

                logits = self._model(**batch).logits
                scores = logits.view(-1).cpu().tolist()
                all_scores.extend(scores)

        return all_scores
