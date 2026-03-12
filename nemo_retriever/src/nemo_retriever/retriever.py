# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
from tqdm import tqdm


@dataclass
class Retriever:
    """Simple query helper over LanceDB with configurable embedders.

    Retrieval pipeline
    ------------------
    1. Embed query strings (NIM endpoint or local HuggingFace model).
    2. Search LanceDB (vector or hybrid vector+BM25).
    3. Optionally rerank the results with ``nvidia/llama-nemotron-rerank-1b-v2``
       (NIM/vLLM endpoint or local HuggingFace model).

    Reranking
    ---------
    Set ``reranker`` to a model name (e.g.
    ``"nvidia/llama-nemotron-rerank-1b-v2"``) to enable post-retrieval
    reranking.  Results are re-sorted by the cross-encoder score and a
    ``"_rerank_score"`` key is added to each hit dict.

    Use ``reranker_endpoint`` to delegate to a running vLLM (>=0.14) or NIM
    server instead of loading the model locally::

        retriever = Retriever(
            reranker="nvidia/llama-nemotron-rerank-1b-v2",
            reranker_endpoint="http://localhost:8000",
        )
        results = retriever.query("What is machine learning?")
    """

    lancedb_uri: str = "lancedb"
    lancedb_table: str = "nv-ingest"
    embedder: str = "nvidia/llama-nemotron-embed-1b-v2"
    embedding_http_endpoint: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    top_k: int = 10
    vector_column_name: str = "vector"
    nprobes: int = 0
    refine_factor: int = 10
    hybrid: bool = False
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[Path] = None
    local_hf_batch_size: int = 64
    # Reranking -----------------------------------------------------------
    reranker: Optional[bool] = False
    """True to enable reranking with the default model, will use the reranker_model_name as hf model"""
    reranker_model_name: Optional[str] = "nvidia/llama-nemotron-rerank-1b-v2"
    """HuggingFace model ID for local reranking (e.g. 'nvidia/llama-nemotron-rerank-1b-v2').
    Set to None to skip reranking (default)."""
    reranker_endpoint: Optional[str] = None
    """Base URL of a vLLM / NIM /rerank endpoint.  Takes priority over local model."""
    reranker_api_key: str = ""
    """Bearer token for the remote rerank endpoint."""
    reranker_max_length: int = 512
    """Tokenizer truncation length for local reranking (max 8 192)."""
    reranker_batch_size: int = 32
    """GPU micro-batch size for local reranking."""
    reranker_refine_factor: int = 4
    """Number of candidates to rerank = top_k * reranker_refine_factor.
    Set to 1 to rerank only the top_k results."""
    # Internal cache for the local rerank model (not part of the public API).
    _reranker_model: Any = field(default=None, init=False, repr=False, compare=False)

    def _resolve_embedding_endpoint(self) -> Optional[str]:
        http_ep = self.embedding_http_endpoint.strip() if isinstance(self.embedding_http_endpoint, str) else None
        single = self.embedding_endpoint.strip() if isinstance(self.embedding_endpoint, str) else None

        if http_ep:
            return http_ep
        if single:
            if not single.lower().startswith("http"):
                raise ValueError("gRPC endpoints are not supported; provide an HTTP NIM endpoint URL.")
            return single
        return None

    def _embed_queries_nim(
        self,
        query_texts: list[str],
        *,
        endpoint: str,
        model: str,
    ) -> list[list[float]]:
        import numpy as np
        from nv_ingest_api.util.nim import infer_microservice

        embeddings = infer_microservice(
            query_texts,
            model_name=model,
            embedding_endpoint=endpoint,
            nvidia_api_key=(self.embedding_api_key or "").strip(),
            input_type="query",
        )
        out: list[list[float]] = []
        for embedding in embeddings:
            if isinstance(embedding, np.ndarray):
                out.append(embedding.astype("float32").tolist())
            else:
                out.append(list(embedding))
        return out

    def _embed_queries_local_hf(self, query_texts: list[str], *, model_name: str) -> list[list[float]]:
        from nemo_retriever.model import create_local_embedder, is_vl_embed_model

        cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None
        embedder = create_local_embedder(model_name, device=self.local_hf_device, hf_cache_dir=cache_dir)

        if is_vl_embed_model(model_name):
            vectors = embedder.embed_queries(query_texts, batch_size=int(self.local_hf_batch_size))
        else:
            vectors = embedder.embed(["query: " + q for q in query_texts], batch_size=int(self.local_hf_batch_size))
        return vectors.detach().to("cpu").tolist()

    def _search_lancedb(
        self,
        *,
        lancedb_uri: str,
        lancedb_table: str,
        query_vectors: list[list[float]],
        query_texts: list[str],
    ) -> list[list[dict[str, Any]]]:
        import lancedb  # type: ignore
        import numpy as np

        db = lancedb.connect(lancedb_uri)
        table = db.open_table(lancedb_table)

        effective_nprobes = int(self.nprobes)
        if effective_nprobes <= 0:
            try:
                for idx in table.list_indices():
                    num_parts = getattr(idx, "num_partitions", None)
                    if num_parts and int(num_parts) > 0:
                        effective_nprobes = int(num_parts)
                        break
            except Exception:
                pass
            if effective_nprobes <= 0:
                effective_nprobes = 16

        results: list[list[dict[str, Any]]] = []
        for i, vector in enumerate(query_vectors):
            q = np.asarray(vector, dtype="float32")
            # doubling top_k for both hybrid and dense search in order to have more to rerank
            top_k = self.top_k if not self.reranker else self.top_k * self.reranker_refine_factor
            if self.hybrid:
                from lancedb.rerankers import RRFReranker  # type: ignore

                hits = (
                    table.search(query_type="hybrid")
                    .vector(q)
                    .text(query_texts[i])
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                    .select(["text", "metadata", "source", "page_number"])
                    .limit(int(top_k))
                    .rerank(RRFReranker())
                    .to_list()
                )
            else:
                hits = (
                    table.search(q, vector_column_name=self.vector_column_name)
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                    .select(["text", "metadata", "source", "page_number", "_distance"])
                    .limit(int(top_k))
                    .to_list()
                )
            results.append(hits)
        return results

    # ------------------------------------------------------------------
    # Reranking helpers
    # ------------------------------------------------------------------

    def _get_reranker_model(self) -> Any:
        """Lazily load and cache the local NemotronRerankV2 model."""
        if self._reranker_model is None and self.reranker:
            from nemo_retriever.model.local import NemotronRerankV2

            cache_dir = str(self.local_hf_cache_dir) if self.local_hf_cache_dir else None
            self._reranker_model = NemotronRerankV2(
                model_name=self.reranker_model_name if self.reranker else None,
                device=self.local_hf_device,
                hf_cache_dir=cache_dir,
            )
        return self._reranker_model

    def _rerank_results(
        self,
        query_texts: list[str],
        results: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        """Rerank each per-query result list using the configured reranker."""
        from nemo_retriever.rerank import rerank_hits

        reranker_endpoint = (self.reranker_endpoint or "").strip() or None
        model = None if reranker_endpoint else self._get_reranker_model()

        reranked: list[list[dict[str, Any]]] = []
        for query, hits in tqdm(zip(query_texts, results), desc="Reranking", unit="query", total=len(query_texts)):
            reranked.append(
                rerank_hits(
                    query,
                    hits,
                    model=model,
                    invoke_url=reranker_endpoint,
                    model_name=str(self.reranker),
                    api_key=(self.reranker_api_key or "").strip(),
                    max_length=int(self.reranker_max_length),
                    batch_size=int(self.reranker_batch_size),
                    top_n=int(self.top_k),
                )
            )
        return reranked

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval for a single query string."""
        return self.queries(
            [query],
            embedder=embedder,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
    ) -> list[list[dict[str, Any]]]:
        """Run retrieval for multiple query strings.

        If ``reranker`` is set on this instance the initial vector-search
        results are re-scored with ``nvidia/llama-nemotron-rerank-1b-v2``
        (or the configured endpoint) and returned sorted by cross-encoder
        score.  Each hit gains a ``"_rerank_score"`` key.
        """
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        resolved_embedder = str(embedder or self.embedder)
        resolved_lancedb_uri = str(lancedb_uri or self.lancedb_uri)
        resolved_lancedb_table = str(lancedb_table or self.lancedb_table)

        endpoint = self._resolve_embedding_endpoint()
        if endpoint is not None:
            vectors = self._embed_queries_nim(
                query_texts,
                endpoint=endpoint,
                model=resolved_embedder,
            )
        else:
            vectors = self._embed_queries_local_hf(
                query_texts,
                model_name=resolved_embedder,
            )

        results = self._search_lancedb(
            lancedb_uri=resolved_lancedb_uri,
            lancedb_table=resolved_lancedb_table,
            query_vectors=vectors,
            query_texts=query_texts,
        )

        if self.reranker:
            assert self.top_k * self.reranker_refine_factor == len(
                results[0]
            ), "top_k must be at least 1/4 of the number of retrieved hits for reranking to work properly."
            results = self._rerank_results(query_texts, results)

        return results


# Backward compatibility alias.
retriever = Retriever
