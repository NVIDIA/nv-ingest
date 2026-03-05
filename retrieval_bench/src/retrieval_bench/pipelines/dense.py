# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified dense retrieval pipeline supporting multiple backends.

Replaces the four standalone example pipelines:
- nemo_reasoning_dense_pipeline.py
- llama_nemotron_embed_vl_1b_v2_dense_pipeline.py
- nemotron_colembed_vl_8b_v2_pipeline.py
- colembed_pipeline.py
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    print("Error: Required GPU dependencies not installed.")
    print("Please install: pip install torch")
    sys.exit(1)

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from retrieval_bench.pipelines.backends import VALID_BACKENDS, infer_bright_task_key, init_backend


class DenseRetrievalPipeline(BasePipeline):
    """
    Dense retrieval pipeline with pluggable backend.

    Backends:
      - llama-nv-embed-reasoning-3b   (text-only dense, BRIGHT-style prefixes)
      - llama-nemoretriever-colembed-3b-v1   (text-only late-interaction ColBERT)
      - llama-nemotron-embed-vl-1b-v2        (multimodal dense, image+text)
      - nemotron-colembed-vl-8b-v2           (multimodal late-interaction ColBERT)
    """

    def __init__(self, *, backend: str, top_k: int = 100, **kwargs: Any) -> None:
        if backend not in VALID_BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. " f"Must be one of: {', '.join(sorted(VALID_BACKENDS))}")
        self.backend = backend
        self.model_id = backend
        self.top_k = int(top_k)
        self._backend_overrides = dict(kwargs)

        if not torch.cuda.is_available():
            print("Error: CUDA is not available. This pipeline requires a GPU.")
            sys.exit(1)

    def index(self, corpus_ids: List[str], corpus_images: List[Any], corpus_texts: List[str]) -> None:
        super().index(corpus_ids=corpus_ids, corpus_images=corpus_images, corpus_texts=corpus_texts)

        dataset_name = self.dataset_name
        task_key = infer_bright_task_key(dataset_name)

        corpus = [{"image": img, "markdown": md} for img, md in zip(corpus_images, corpus_texts)]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_init0 = time.perf_counter()

        active_retriever, effective_model_id, init_info = init_backend(
            self.backend,
            dataset_name=dataset_name,
            corpus_ids=corpus_ids,
            corpus=corpus,
            top_k=self.top_k,
            task_key=task_key,
            overrides=self._backend_overrides or None,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        retriever_init_ms = (time.perf_counter() - t_init0) * 1000.0

        print(f"Using backend {self.backend} ({effective_model_id})")
        print(f"Retriever init: {retriever_init_ms / 1000.0:.2f}s")

        self._active_retriever = active_retriever
        self._init_info = init_info
        self._retriever_init_ms = retriever_init_ms

    def search(
        self,
        query_ids: List[str],
        queries: List[str],
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
        results: Dict[str, Dict[str, float]] = {}
        per_query_ms: Dict[str, float] = {}

        excluded_ids_by_query = getattr(self, "excluded_ids_by_query", None) or {}

        try:
            for q_idx, (query_id, query_text) in enumerate(zip(query_ids, queries)):
                if q_idx % 25 == 0:
                    print(f"  Retrieving for query {q_idx + 1}/{len(query_ids)}...")

                excluded_ids: Optional[List[str]] = None
                if isinstance(excluded_ids_by_query, dict):
                    ex = excluded_ids_by_query.get(str(query_id))
                    if isinstance(ex, list):
                        excluded_ids = [str(d) for d in ex if str(d) != "N/A"]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                try:
                    results[str(query_id)] = self._active_retriever.retrieve(str(query_text), excluded_ids=excluded_ids)
                except TypeError:
                    results[str(query_id)] = self._active_retriever.retrieve(str(query_text))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                per_query_ms[str(query_id)] = (t1 - t0) * 1000.0
        finally:
            self._active_retriever.unload()

        total_ms = sum(per_query_ms.values())
        print(f"\nRetrieval complete in {total_ms / 1000.0:.2f} seconds (sum of per-query times)")
        if query_ids:
            print(f"Average time per query: {total_ms / len(query_ids):.2f} ms")

        infos: Dict[str, Any] = {
            **self._init_info,
            "retriever_init_ms": float(self._retriever_init_ms),
            "per_query_retrieval_time_milliseconds": per_query_ms,
        }
        return results, infos
