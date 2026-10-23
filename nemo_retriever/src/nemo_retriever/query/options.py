# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from nemo_retriever.common.vdb.query_options import validate_hnsw_ef

QueryRetrievalMode = Literal["auto", "dense", "hybrid", "sparse"]


@dataclass(frozen=True)
class QueryRetrievalOptions:
    top_k: int = 10
    candidate_k: int | None = None
    page_dedup: bool = False
    content_types: str | Sequence[str] | None = None
    # ``auto`` lets LanceDB table capability detection choose dense, hybrid, or
    # sparse retrieval. Explicit modes are expert overrides.
    retrieval_mode: QueryRetrievalMode = "auto"
    # None preserves LanceDB's native HNSW search default.
    ef: int | None = None

    def __post_init__(self) -> None:
        validate_hnsw_ef(self.ef)
        if self.ef is not None and self.retrieval_mode == "sparse":
            raise ValueError("ef requires dense or hybrid LanceDB retrieval.")


@dataclass(frozen=True)
class QueryEmbedOptions:
    embed_invoke_url: str | None = None
    embed_model_name: str | None = None
    embed_model_provider_prefix: str | None = None
    embed_api_key: str | None = None


@dataclass(frozen=True)
class QueryRerankOptions:
    enabled: bool = False
    reranker_invoke_url: str | None = None
    reranker_model_name: str | None = None
    reranker_backend: str | None = None
    reranker_api_key: str | None = None


@dataclass(frozen=True)
class QueryStorageOptions:
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"


@dataclass(frozen=True)
class QueryServiceOptions:
    service_url: str = "http://localhost:7670"
    service_api_token: str | None = None


@dataclass(frozen=True)
class QueryAgenticOptions:
    """Options for the agentic (ReAct) retrieval strategy."""

    enabled: bool = False
    llm_model: str | None = None
    llm_backend: str | None = None
    invoke_url: str | None = None
    local_llm_backend: str = "vllm"
    local_hf_cache_dir: str | None = None
    local_gpu_memory_utilization: float = 0.8
    local_tensor_parallel_size: int = 1
    local_max_model_len: int | None = None
    local_max_num_seqs: int | None = None
    reasoning_effort: str | None = None
    react_max_steps: int = 50
    text_truncation: int = 0
    num_concurrent: int = 1
    temperature: float | None = None
    # LLM client (see AgenticRetrievalConfig.llm_client). Optional: defaults to
    # ``callable`` for both in-process and remote runs when unset.
    llm_client: str | None = None
    # Accepted for service-layer compatibility only. The agent derives its own
    # per-hop retrieval depth from ``top_k``, so this value is never read.
    backend_top_k: int | None = None


@dataclass(frozen=True)
class QueryRequest:
    query: str
    retrieval: QueryRetrievalOptions = field(default_factory=QueryRetrievalOptions)
    embed: QueryEmbedOptions = field(default_factory=QueryEmbedOptions)
    rerank: QueryRerankOptions = field(default_factory=QueryRerankOptions)
    storage: QueryStorageOptions = field(default_factory=QueryStorageOptions)
    agentic: QueryAgenticOptions = field(default_factory=QueryAgenticOptions)


@dataclass(frozen=True)
class ServiceQueryRequest:
    query: str
    retrieval: QueryRetrievalOptions = field(default_factory=QueryRetrievalOptions)
    service: QueryServiceOptions = field(default_factory=QueryServiceOptions)

    def __post_init__(self) -> None:
        if self.retrieval.ef is not None:
            raise ValueError("ef is not supported by service query requests.")
