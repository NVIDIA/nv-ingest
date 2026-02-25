from __future__ import annotations

"""
Lightweight text embedding helpers (local HF).

This module is intentionally independent of `nv-ingest-api` so it can be used in
environments that don't have the full schema/transform stack installed.

It mirrors the "pure pandas batch fn + Ray-friendly actor" pattern used by:
- `retriever.page_elements.detect_page_elements_v3`
- `retriever.chart.chart_detection.detect_graphic_elements_v1`
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence  # noqa: F401

import time
import traceback

import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _error_payload(*, stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "embedding": None,
        "error": {
            "stage": str(stage),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        },
    }


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
    output_column: str = "text_embeddings_1b_v2",
    embedding_dim_column: str = "text_embeddings_1b_v2_dim",
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding",
    **_: Any,
) -> Any:
    """
    Embed a batch of text rows using the local `LlamaNemotronEmbed1BV2Embedder`.

    Input:
      - `batch_df`: pandas.DataFrame (Ray Data `batch_format="pandas"` compatible)
      - `text_column`: preferred column name to read text from (defaults to `"text"`)

    Output:
      - Returns a pandas.DataFrame with original columns preserved, plus:
        - `output_column`: dict payload `{"embedding": list[float]|None, "timing": {...}, "error": ...}`
        - `embedding_dim_column`: int
        - `has_embedding_column`: bool
    """
    _ = (model_name, embedding_endpoint)  # reserved for future remote execution support
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_1b_v2 currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    payloads: List[Dict[str, Any]] = [{"embedding": None, "error": None} for _ in range(len(batch_df.index))]
    texts: List[str] = []
    text_row_idxs: List[int] = []

    for i, (_, row) in enumerate(batch_df.iterrows()):
        try:
            txt = _text_from_row(row, text_column=str(text_column))
            if not txt.strip():
                # Keep placeholder but mark as "no text".
                payloads[i] = {"embedding": None, "error": None}
                continue
            texts.append(f"{input_type}: {txt}" if input_type else txt)
            text_row_idxs.append(i)
        except BaseException as e:
            payloads[i] = _error_payload(stage="extract_text", exc=e)

    # Nothing to embed.
    if not texts:
        out0 = batch_df.copy()
        out0[output_column] = payloads
        out0[embedding_dim_column] = [0 for _ in range(len(out0.index))]
        out0[has_embedding_column] = [False for _ in range(len(out0.index))]
        return out0

    # Run inference in chunks.
    for start in range(0, len(texts), int(inference_batch_size)):
        chunk_texts = texts[start : start + int(inference_batch_size)]
        chunk_idxs = text_row_idxs[start : start + int(inference_batch_size)]
        if not chunk_texts:
            continue

        t0 = time.perf_counter()
        try:
            vecs = model.embed(chunk_texts, batch_size=int(inference_batch_size))
            elapsed = time.perf_counter() - t0

            if torch is not None and isinstance(vecs, torch.Tensor):
                vecs_list = vecs.detach().to("cpu").tolist()
            elif isinstance(vecs, list):
                vecs_list = vecs
            else:
                # Best-effort conversion.
                tolist = getattr(vecs, "tolist", None)
                vecs_list = tolist() if callable(tolist) else vecs  # type: ignore[assignment]

            if not isinstance(vecs_list, list) or len(vecs_list) != len(chunk_idxs):
                raise RuntimeError("Embedder returned unexpected output shape/type.")

            for local_i, row_i in enumerate(chunk_idxs):
                emb = vecs_list[local_i]
                if not isinstance(emb, list):
                    # Allow numpy arrays or tensors that slipped through.
                    tolist = getattr(emb, "tolist", None)
                    emb = tolist() if callable(tolist) else emb
                payloads[row_i] = {
                    "embedding": emb,
                    "timing": {"seconds": float(elapsed)},
                    "error": None,
                }
        except BaseException as e:
            elapsed = time.perf_counter() - t0
            for row_i in chunk_idxs:
                payloads[row_i] = _error_payload(stage="embed", exc=e) | {"timing": {"seconds": float(elapsed)}}

    out = batch_df.copy()
    out[output_column] = payloads
    out[embedding_dim_column] = [
        (
            int(len((p or {}).get("embedding") or []))
            if isinstance(p, dict) and isinstance(p.get("embedding"), list)
            else 0
        )
        for p in payloads
    ]
    out[has_embedding_column] = [bool(d > 0) for d in out[embedding_dim_column].tolist()]
    return out


@dataclass(slots=True)
class TextEmbedActor:
    """
    Ray-friendly callable that initializes `LlamaNemotronEmbed1BV2Embedder` once.
    """

    detect_kwargs: Dict[str, Any]

    def __init__(self, **detect_kwargs: Any) -> None:
        self.detect_kwargs = dict(detect_kwargs)
        from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

        device = self.detect_kwargs.pop("device", None)
        hf_cache_dir = self.detect_kwargs.pop("hf_cache_dir", None)
        normalize = bool(self.detect_kwargs.pop("normalize", True))
        max_length = self.detect_kwargs.pop("max_length", 4096)

        self._model = LlamaNemotronEmbed1BV2Embedder(
            device=str(device) if device is not None else None,
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            normalize=normalize,
            max_length=int(max_length),
        )

    def __call__(self, batch_df: Any, **override_kwargs: Any) -> Any:
        try:
            return embed_text_1b_v2(
                batch_df,
                model=self._model,
                **self.detect_kwargs,
                **override_kwargs,
            )
        except BaseException as e:
            if isinstance(batch_df, pd.DataFrame):
                out = batch_df.copy()
                payload = _error_payload(stage="actor_call", exc=e)
                out["text_embeddings_1b_v2"] = [payload for _ in range(len(out.index))]
                out["text_embeddings_1b_v2_dim"] = [0 for _ in range(len(out.index))]
                out["text_embeddings_1b_v2_has_embedding"] = [False for _ in range(len(out.index))]
                return out
            return [{"text_embeddings_1b_v2": _error_payload(stage="actor_call", exc=e)}]
