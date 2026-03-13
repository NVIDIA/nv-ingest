# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared backend initialization for dense retrieval singletons.

Used by both DenseRetrievalPipeline and AgenticRetrievalPipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union


def infer_bright_task_key(dataset_name: Any) -> Optional[str]:
    """Extract the BRIGHT task key (e.g. 'biology') from a dataset name like 'bright/biology'."""
    try:
        ds = str(dataset_name or "").strip()
    except Exception:
        return None
    if not ds:
        return None
    parts = [p for p in ds.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "bright":
        return parts[1]
    return None


VALID_BACKENDS = {
    "llama-nv-embed-reasoning-3b",
    "llama-nemoretriever-colembed-3b-v1",
    "llama-nemotron-embed-vl-1b-v2",
    "nemotron-colembed-vl-8b-v2",
}

_BACKEND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "llama-nv-embed-reasoning-3b": {
        "model_id": "nvidia/llama-nv-embed-reasoning-3b",
        "max_length": 8192,
        "pooling": "mean",
        "score_scale": 100.0,
        "corpus_batch_size": 1,
        "max_scoring_batch_size": 4096,
        "query_prefix_fallback": (
            "Instruct: Given the following post, retrieve relevant passages that help answer the post.\nQuery:"
        ),
    },
    "llama-nemoretriever-colembed-3b-v1": {
        "model_id": "nvidia/llama-nemoretriever-colembed-3b-v1",
        "batch_size": 32,
        "corpus_batch_size": 32,
        "max_scoring_batch_size": 256,
    },
    "llama-nemotron-embed-vl-1b-v2": {
        "model_id": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "device": "auto",
        "doc_modality": "image_text",
        "doc_max_length": "auto",
        "query_max_length": 10240,
        "corpus_batch_size": 32,
        "max_scoring_batch_size": 4096,
        "max_input_tiles": 6,
        "use_thumbnail": True,
    },
    "nemotron-colembed-vl-8b-v2": {
        "model_id": "nvidia/nemotron-colembed-vl-8b-v2",
        "corpus_batch_size": 32,
        "max_scoring_batch_size": 3000,
        "scoring_chunk_size": 1311,
        "max_input_tiles": 8,
        "use_thumbnail": True,
        "cache_dir": "cache/nemotron_colembed_vl_v2",
    },
}


class _NemotronEmbedVLAdapter:
    """Adapter so nemotron_embed_vl matches the retrieve(excluded_ids=...) interface."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def retrieve(
        self,
        query: str,
        *,
        return_markdown: bool = False,
        excluded_ids: Optional[Sequence[str]] = None,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
        return self._inner.retrieve(
            str(query),
            return_markdown=bool(return_markdown),
            excluded_ids=excluded_ids,
        )

    def unload(self) -> None:
        self._inner.unload()


def _import_retriever(backend: str) -> Any:
    """Lazily import the singleton retriever for a given backend."""
    if backend == "llama-nv-embed-reasoning-3b":
        from retrieval_bench.singletons.hf_dense_retriever import retriever

        return retriever
    elif backend == "llama-nemoretriever-colembed-3b-v1":
        from retrieval_bench.singletons.colembed_retriever import retriever

        return retriever
    elif backend == "llama-nemotron-embed-vl-1b-v2":
        from retrieval_bench.singletons.nemotron_embed_vl_dense_retriever import retriever

        return retriever
    elif backend == "nemotron-colembed-vl-8b-v2":
        from retrieval_bench.singletons.nemotron_colembed_vl_v2_retriever import retriever

        return retriever
    else:
        raise ValueError(f"Unknown backend {backend!r}. Must be one of: {', '.join(sorted(VALID_BACKENDS))}")


def get_backend_defaults(backend: str) -> Dict[str, Any]:
    """Return a copy of the default kwargs for a backend."""
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
    return dict(_BACKEND_DEFAULTS[backend])


def init_backend(
    backend: str,
    *,
    dataset_name: str,
    corpus_ids: Any,
    corpus: Any,
    top_k: int = 100,
    task_key: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Initialize a retriever backend and return (active_retriever, effective_model_id, init_info).

    The returned retriever object has .retrieve() and .unload() methods.
    ``init_info`` contains backend-specific metadata for the infos dict.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Must be one of: {', '.join(sorted(VALID_BACKENDS))}")

    cfg = get_backend_defaults(backend)
    if overrides:
        cfg.update(overrides)

    model_id = os.path.expanduser(str(cfg.pop("model_id")))
    retriever = _import_retriever(backend)
    init_info: Dict[str, Any] = {"backend": backend}

    if backend == "llama-nv-embed-reasoning-3b":
        from retrieval_bench.prompts.bright_instructions import (
            NEMO_REASONING_PASSAGE_PREFIX,
            get_bright_query_prefix_nemo,
        )

        query_prefix_fallback = str(cfg.pop("query_prefix_fallback"))
        query_prefix = get_bright_query_prefix_nemo(task_key=task_key, fallback=query_prefix_fallback)

        pooling = str(cfg.pop("pooling", "mean"))
        max_length = int(cfg.pop("max_length", 8192))
        score_scale = float(cfg.pop("score_scale", 100.0))
        corpus_batch_size = int(cfg.pop("corpus_batch_size", 1))
        max_scoring_batch_size = int(cfg.pop("max_scoring_batch_size", 4096))

        if cfg:
            raise ValueError(f"Unknown pipeline arg(s) for backend {backend!r}: {', '.join(sorted(cfg))}")

        retriever.init(
            dataset_name=dataset_name,
            corpus_ids=corpus_ids,
            corpus=corpus,
            model_id=model_id,
            device="cuda",
            top_k=top_k,
            max_length=max_length,
            pooling=pooling,
            doc_prefix=str(NEMO_REASONING_PASSAGE_PREFIX),
            query_prefix=str(query_prefix),
            task_description="Given the following post, retrieve relevant passages that help answer the post.",
            score_scale=score_scale,
            batch_size=1,
            corpus_batch_size=corpus_batch_size,
            max_scoring_batch_size=max_scoring_batch_size,
            cache_dir="cache/hf_dense",
        )
        init_info.update(
            {
                "model_id": model_id,
                "pooling": pooling,
                "task_key": task_key,
                "query_prefix": str(query_prefix),
                "doc_prefix": str(NEMO_REASONING_PASSAGE_PREFIX),
                "max_length": max_length,
                "score_scale": score_scale,
            }
        )
        return retriever, model_id, init_info

    elif backend == "llama-nemoretriever-colembed-3b-v1":
        batch_size = int(cfg.pop("batch_size", 32))
        corpus_batch_size = int(cfg.pop("corpus_batch_size", 32))
        max_scoring_batch_size = int(cfg.pop("max_scoring_batch_size", 256))

        if cfg:
            raise ValueError(f"Unknown pipeline arg(s) for backend {backend!r}: {', '.join(sorted(cfg))}")

        retriever.init(
            dataset_name=dataset_name,
            corpus_ids=corpus_ids,
            corpus=corpus,
            model_id=model_id,
            top_k=top_k,
            batch_size=batch_size,
            corpus_batch_size=corpus_batch_size,
            max_scoring_batch_size=max_scoring_batch_size,
            cache_dir="cache",
        )
        init_info.update({"model_id": model_id})
        return retriever, model_id, init_info

    elif backend == "llama-nemotron-embed-vl-1b-v2":
        device = str(cfg.pop("device", "auto"))
        doc_modality = str(cfg.pop("doc_modality", "image_text"))
        doc_max_length = cfg.pop("doc_max_length", "auto")

        # Auto-detect: fall back to text-only when the corpus has no images
        # (e.g. BRIGHT text-only datasets).
        if doc_modality != "text" and not any(doc.get("image") is not None for doc in corpus[:5]):
            doc_modality = "text"

        query_max_length = int(cfg.pop("query_max_length", 10240))
        corpus_batch_size = int(cfg.pop("corpus_batch_size", 4))
        max_scoring_batch_size = int(cfg.pop("max_scoring_batch_size", 4096))
        max_input_tiles = int(cfg.pop("max_input_tiles", 6))
        use_thumbnail = bool(cfg.pop("use_thumbnail", True))

        if cfg:
            raise ValueError(f"Unknown pipeline arg(s) for backend {backend!r}: {', '.join(sorted(cfg))}")

        retriever.init(
            dataset_name=dataset_name,
            corpus_ids=corpus_ids,
            corpus=corpus,
            model_id=model_id,
            device=device,
            top_k=top_k,
            doc_modality=doc_modality,
            doc_max_length=doc_max_length,
            query_max_length=query_max_length,
            corpus_batch_size=corpus_batch_size,
            max_scoring_batch_size=max_scoring_batch_size,
            cache_dir="cache/nemotron_vl_dense",
            max_input_tiles=max_input_tiles,
            use_thumbnail=use_thumbnail,
        )
        init_info.update(
            {
                "model_id": model_id,
                "device": device,
                "doc_modality": doc_modality,
                "doc_max_length": doc_max_length,
                "query_max_length": query_max_length,
                "max_input_tiles": max_input_tiles,
                "use_thumbnail": use_thumbnail,
                "corpus_batch_size": corpus_batch_size,
                "max_scoring_batch_size": max_scoring_batch_size,
            }
        )
        active = _NemotronEmbedVLAdapter(retriever)
        return active, model_id, init_info

    else:  # nemotron-colembed-vl-8b-v2
        corpus_batch_size = int(cfg.pop("corpus_batch_size", 32))
        max_scoring_batch_size = int(cfg.pop("max_scoring_batch_size", 3000))
        scoring_chunk_size = int(cfg.pop("scoring_chunk_size", 1311))
        max_input_tiles = int(cfg.pop("max_input_tiles", 8))
        use_thumbnail = bool(cfg.pop("use_thumbnail", True))
        cache_dir = str(cfg.pop("cache_dir", "cache/nemotron_colembed_vl_v2"))

        if cfg:
            raise ValueError(f"Unknown pipeline arg(s) for backend {backend!r}: {', '.join(sorted(cfg))}")

        retriever.init(
            dataset_name=str(dataset_name),
            corpus_ids=corpus_ids,
            corpus=corpus,
            model_id=str(model_id),
            device="cuda",
            top_k=top_k,
            corpus_batch_size=corpus_batch_size,
            max_scoring_batch_size=max_scoring_batch_size,
            scoring_chunk_size=scoring_chunk_size,
            cache_dir=cache_dir,
            max_input_tiles=max_input_tiles,
            use_thumbnail=use_thumbnail,
        )
        init_info.update({"model_id": model_id})
        return retriever, model_id, init_info
