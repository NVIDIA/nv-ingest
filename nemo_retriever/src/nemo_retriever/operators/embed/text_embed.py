# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Lightweight text embedding helpers (local vLLM for the default 1b-v2 text model).

This module is intentionally independent of `nemo_retriever.api` so it can be used in
environments that don't have the full schema/transform stack installed.

It mirrors the "pure pandas batch fn + Ray-friendly actor" pattern used by
the extraction operators.
"""

import logging
from typing import Any, List, Optional

import pandas as pd
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.models.inference.main_text_embed import assign_embedding_columns, embedding_error_message

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _text_from_row(row: pd.Series, *, text_column: str) -> str:
    """
    Extract text from a row with a small set of fallbacks.
    """
    v = row.get(text_column)
    if isinstance(v, str) and v.strip():
        return v

    # Common alternative keys.
    for k in ("text", "content", "chunk", "page_text"):
        v2 = row.get(k)
        if isinstance(v2, str) and v2.strip():
            return v2

    return ""


def embed_text_1b_v2(
    batch_df: Any,
    *,
    model: Any,
    # Optional compatibility args (e.g. passed by `.embed(model_name=..., embedding_endpoint=...)`).
    # This lightweight implementation uses the provided local `model` regardless.
    model_name: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    input_type: str = "passage",
    text_column: str = "text",
    inference_batch_size: int = 32,
    output_column: str = "embedding",
    **_: Any,
) -> Any:
    """
    Embed a batch of text rows using the local `LlamaNemotronEmbed1BV2Embedder`.

    Input:
      - `batch_df`: pandas.DataFrame (Ray Data `batch_format="pandas"` compatible)
      - `text_column`: preferred column name to read text from (defaults to `"text"`)

    Output:
      - Returns a pandas.DataFrame with original columns preserved, plus:
        - `output_column`: list<float32> vector or null
        - `f"{output_column}_error"`: string or null
    """
    _ = (model_name, embedding_endpoint)  # reserved for future remote execution support
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_1b_v2 currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    vectors: List[Any] = [None] * len(batch_df.index)
    errors: List[Optional[str]] = [None] * len(batch_df.index)
    texts: List[str] = []
    text_row_idxs: List[int] = []

    for i, (_, row) in enumerate(batch_df.iterrows()):
        try:
            txt = _text_from_row(row, text_column=str(text_column))
            if not txt.strip():
                continue
            # VL model handles formatting internally; skip prefix.
            if input_type and not hasattr(model, "embed_queries"):
                texts.append(f"{input_type}: {txt}")
            else:
                texts.append(txt)
            text_row_idxs.append(i)
        except BaseException as e:
            logger.error("Text extraction failed for row %d: %s", i, e, exc_info=e)
            errors[i] = embedding_error_message("extract_text", e)

    # Run inference in chunks.
    for start in range(0, len(texts), int(inference_batch_size)):
        chunk_texts = texts[start : start + int(inference_batch_size)]
        chunk_idxs = text_row_idxs[start : start + int(inference_batch_size)]
        if not chunk_texts:
            continue

        try:
            vecs = model.embed(chunk_texts, batch_size=int(inference_batch_size))
            if torch is not None and isinstance(vecs, torch.Tensor):
                rows = vecs.detach().to("cpu").float().numpy()
            else:
                rows = vecs

            if len(rows) != len(chunk_idxs):
                raise RuntimeError("Embedder returned unexpected output shape/type.")

            for local_i, row_i in enumerate(chunk_idxs):
                vectors[row_i] = rows[local_i]
        except BaseException as e:
            logger.error("Text embedding failed for chunk starting at row %d: %s", start, e, exc_info=e)
            for row_i in chunk_idxs:
                errors[row_i] = embedding_error_message("embed", e)

    return assign_embedding_columns(batch_df, vectors, errors, output_column=output_column)


@designer_component(
    name="Text Embedder",
    category="Embeddings & Ranking",
    compute="gpu",
    description="Generates text embeddings using a language model",
    category_color="#e06cff",
)
class TextEmbedGPUActor(AbstractOperator, GPUOperator):
    """
    Ray-friendly callable that initializes `LlamaNemotronEmbed1BV2Embedder` once.
    """

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)
        self.detect_kwargs = dict(detect_kwargs)

        self.detect_kwargs.pop("device", None)  # unused for local text vLLM embedder
        hf_cache_dir = self.detect_kwargs.pop("hf_cache_dir", None)
        normalize = bool(self.detect_kwargs.pop("normalize", True))
        max_length = self.detect_kwargs.pop("max_length", 4096)

        from nemo_retriever.models import create_local_embedder

        self._model = create_local_embedder(
            self.detect_kwargs.get("model_name"),
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            normalize=normalize,
            max_length=int(max_length),
        )

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, batch_df: Any, **override_kwargs: Any) -> Any:
        return embed_text_1b_v2(
            batch_df,
            model=self._model,
            **self.detect_kwargs,
            **override_kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(batch_df, **override_kwargs)
        except BaseException as e:
            logger.error("Text embedding actor failed: %s", e, exc_info=e)
            output_column = override_kwargs.get("output_column", self.detect_kwargs.get("output_column", "embedding"))
            error_column = f"{output_column}_error"
            message = embedding_error_message("actor_call", e)
            if isinstance(batch_df, pd.DataFrame):
                return assign_embedding_columns(
                    batch_df,
                    [None] * len(batch_df),
                    [message] * len(batch_df),
                    output_column=output_column,
                )
            return [{output_column: None, error_column: message}]


class TextEmbedActor(ArchetypeOperator):
    _gpu_variant_class = TextEmbedGPUActor

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)
