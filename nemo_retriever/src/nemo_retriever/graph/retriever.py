# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast

import pandas as pd

from nemo_retriever.models import (
    NEMOTRON_3_EMBED_MODEL,
    VL_RERANK_MODEL,
    resolve_embed_model,
    resolve_embed_model_spec,
)
from nemo_retriever.graph.retriever_utils import (
    filter_retrieval_kwargs,
    rerank_long_dataframe_to_hits,
)
from nemo_retriever.common.vdb.hybrid_fusion import DEFAULT_HYBRID_FUSION_POLICY
from nemo_retriever.common.vdb.lancedb_capabilities import (
    LanceRetrievalMode,
    LanceTableCapabilities,
    inspect_lancedb_table,
)
from nemo_retriever.common.vdb.records import RetrievalHit, normalize_retrieval_results
from nemo_retriever.query.shaping import shape_query_hits
from nemo_retriever.operators.vdb import RetrieveVdbOperator

logger = logging.getLogger(__name__)

_QUERY_ROUTING_VDB_KWARGS = frozenset({"retrieval_mode"})

if TYPE_CHECKING:
    from nemo_retriever.models.llm.types import (
        AnswerJudge,
        AnswerResult,
        LLMClient,
        RetrievalResult,
    )


def _coerce_vdb_init(user: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``vdb_kwargs`` into :class:`RetrieveVdbOperator` constructor kwargs."""
    u = dict(user or {})
    for key in _QUERY_ROUTING_VDB_KWARGS:
        u.pop(key, None)
    if "vdb" in u or "vdb_op" in u:
        if isinstance(u.get("vdb_kwargs"), dict):
            nested = dict(u["vdb_kwargs"])
            for key in _QUERY_ROUTING_VDB_KWARGS:
                nested.pop(key, None)
            u["vdb_kwargs"] = nested
        return u
    return {"vdb_op": "lancedb", "vdb_kwargs": u}


def _default_rerank_actor_kwargs() -> dict[str, Any]:
    return {
        "model_name": VL_RERANK_MODEL,
        "query_column": "query",
        "text_column": "text",
        "score_column": "rerank_score",
        "max_length": 10240,
        "batch_size": 32,
        "sort_results": False,
        "api_key": "",
    }


@dataclass
class Retriever:
    """Graph-based query helper: batch embed → VDB retrieve [→ Nemotron rerank].

    Configuration is passed through ``embed_kwargs`` (:class:`~nemo_retriever.params.EmbedParams`),
    ``vdb_kwargs`` (constructor kwargs for :class:`~nemo_retriever.vdb.operators.RetrieveVdbOperator`),
    and optional ``rerank_kwargs`` for :class:`~nemo_retriever.rerank.rerank.NemotronRerankActor`.

    See ``retriever.md`` for examples.
    """

    run_mode: Literal["local", "service"] = "local"
    """``local`` uses archetype batch embed resolution; ``service`` forces CPU HTTP embed."""

    top_k: int = 10
    rerank: bool = False
    """When ``True``, append :class:`~nemo_retriever.rerank.rerank.NemotronRerankActor` after retrieval."""

    graph: Any = None
    """Custom :class:`~nemo_retriever.graph.pipeline_graph.Graph`. When set, ``embed_kwargs`` /
    ``vdb_kwargs`` default-graph fields are ignored for construction (you still pass execute kwargs)."""

    embed_kwargs: dict[str, Any] = field(default_factory=dict)
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    rerank_kwargs: dict[str, Any] = field(default_factory=dict)

    _cached_graph: Any = field(default=None, init=False, repr=False, compare=False)
    _cache_key: Any = field(default=None, init=False, repr=False, compare=False)
    _lancedb_capabilities_cache: dict[tuple[str, str], LanceTableCapabilities] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.run_mode not in ("local", "service"):
            raise ValueError("run_mode must be 'local' or 'service'")

    def _merge_embed_params(self, extra: Optional[dict[str, Any]] = None) -> Any:
        from nemo_retriever.models import _LOCAL_INGEST_EMBED_BACKENDS, normalize_backend
        from nemo_retriever.common.params import EmbedParams

        base: dict[str, Any] = {
            "model_name": NEMOTRON_3_EMBED_MODEL,
            "embed_model_name": NEMOTRON_3_EMBED_MODEL,
            "input_type": "query",
            "text_column": "text",
            "inference_batch_size": 32,
            "embed_inference_batch_size": 32,
            "local_ingest_embed_backend": "vllm",
        }
        overrides = {**dict(self.embed_kwargs or {}), **dict(extra or {})}
        merged = {**base, **overrides}
        endpoint = str(merged.get("embedding_endpoint") or merged.get("embed_invoke_url") or "").strip()
        if (
            self.run_mode == "local"
            and not endpoint
            and "local_ingest_embed_backend" in overrides
            and overrides["local_ingest_embed_backend"] is None
        ):
            model_id = str(merged.get("embed_model_name") or merged.get("model_name") or "").strip()
            if resolve_embed_model(model_id) == NEMOTRON_3_EMBED_MODEL:
                merged["local_ingest_embed_backend"] = "vllm"
            else:
                spec = resolve_embed_model_spec(model_id, revision=merged.get("embed_model_revision"))
                merged["local_ingest_embed_backend"] = "vllm" if spec.requires_vllm else "hf"
                merged["embed_model_revision"] = spec.revision
        if "local_ingest_embed_backend" in merged and merged["local_ingest_embed_backend"] is not None:
            merged["local_ingest_embed_backend"] = normalize_backend(
                str(merged["local_ingest_embed_backend"]),
                _LOCAL_INGEST_EMBED_BACKENDS,
                field_name="local_ingest_embed_backend",
                default="vllm",
            )
        params = EmbedParams.model_validate(merged)
        if self.run_mode == "service":
            url = (params.embedding_endpoint or params.embed_invoke_url or "").strip()
            if not url:
                raise ValueError(
                    "run_mode='service' requires a non-empty HTTP embedding URL. "
                    "Set ``embedding_endpoint`` or ``embed_invoke_url`` inside ``embed_kwargs``."
                )
        return params

    def _merge_rerank_actor_kwargs(self) -> dict[str, Any]:
        return {**_default_rerank_actor_kwargs(), **dict(self.rerank_kwargs or {})}

    def _refine_factor(self) -> int:
        if not self.rerank:
            return 1
        return int(self._merge_rerank_actor_kwargs().get("refine_factor", 4))

    def _build_default_graph(self, *, embed_extra: Optional[dict[str, Any]] = None) -> Any:
        from nemo_retriever.operators.rerank import NemotronRerankActor
        from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor
        from nemo_retriever.operators.embed.operators import _BatchEmbedActor

        embed_params = self._merge_embed_params(embed_extra)
        if self.run_mode == "service":
            embed_op = _BatchEmbedCPUActor(params=embed_params)
        else:
            embed_op = _BatchEmbedActor(params=embed_params)

        vdb_init = _coerce_vdb_init(self.vdb_kwargs)
        retrieve = RetrieveVdbOperator(
            explode_for_rerank=self.rerank,
            embedding_column=embed_params.output_column,
            **vdb_init,
        )

        chain = embed_op >> retrieve
        if self.rerank:
            rk = self._merge_rerank_actor_kwargs()
            rk.pop("refine_factor", None)
            chain = chain >> NemotronRerankActor(**rk)

        return chain

    def _get_graph(self, *, embed_extra: Optional[dict[str, Any]] = None) -> Any:
        if self.graph is not None:
            return self.graph

        key = (
            self.run_mode,
            self.rerank,
            json.dumps(self.vdb_kwargs, sort_keys=True, default=str),
            json.dumps(self.embed_kwargs, sort_keys=True, default=str),
            json.dumps(self.rerank_kwargs, sort_keys=True, default=str),
            json.dumps(embed_extra or {}, sort_keys=True, default=str),
        )
        if self._cached_graph is not None and self._cache_key == key:
            return self._cached_graph
        g = self._build_default_graph(embed_extra=embed_extra)
        self._cached_graph = g
        self._cache_key = key
        return g

    def _execute_queries_graph(
        self,
        query_texts: list[str],
        *,
        effective_top_k: int,
        retrieval_top_k: int,
        vdb_call_kwargs: Optional[dict[str, Any]],
        embed_extra: Optional[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if self.graph is None:
            embed_params = self._merge_embed_params(embed_extra)
            text_col = str(embed_params.text_column)
        else:
            # A caller-owned graph controls its own operators and does not
            # require default embedding configuration. In particular, avoid
            # resolving a local embedding model merely to choose its input
            # column. Agentic result graphs use ``query_text`` and delegate
            # actual retrieval to their configured inner retriever.
            text_col = str({**dict(self.embed_kwargs or {}), **dict(embed_extra or {})}.get("text_column") or "text")
        df = pd.DataFrame({text_col: query_texts})

        # Hybrid retrieval relies on these ordered query strings staying aligned
        # with the embedded rows produced from ``df``. If this query graph grows
        # distributed/shuffled stages, carry row-local query text or IDs instead.
        graph = self._get_graph(embed_extra=embed_extra)

        exec_kwargs: dict[str, Any] = {
            **filter_retrieval_kwargs(dict(vdb_call_kwargs or {})),
            "top_k": int(retrieval_top_k),
            "query_texts": query_texts,
        }
        if self.graph is None:
            leaves = graph.execute_in_place(df, **exec_kwargs)
        else:
            # Preserve resolve-per-query behavior for caller-owned graphs, which
            # may be mutated between calls.
            resolve = getattr(graph, "resolve_for_local_execution", None)
            if not callable(resolve):
                raise TypeError("graph must provide resolve_for_local_execution() (e.g. pipeline_graph.Graph)")
            resolved = resolve()
            leaves = resolved.execute(df, **exec_kwargs)
        if len(leaves) != 1:
            raise RuntimeError(
                f"Retriever query graph must yield exactly one leaf output; got {len(leaves)}. "
                "Use a linear graph or adjust your custom ``graph``."
            )
        out = leaves[0]

        if isinstance(out, pd.DataFrame):
            if not self.rerank:
                raise TypeError(
                    "Graph returned a DataFrame but ``rerank`` is False; expected list[list[dict]] from retrieval."
                )
            rk = self._merge_rerank_actor_kwargs()
            score_col = str(rk.get("score_column", "rerank_score"))
            return rerank_long_dataframe_to_hits(
                out, query_texts=query_texts, top_k=int(effective_top_k), score_column=score_col
            )
        if not isinstance(out, list):
            raise TypeError(f"Unexpected query graph output type: {type(out).__name__}")
        return out

    def _inspect_lancedb_capabilities(self, uri: str, table_name: str) -> LanceTableCapabilities:
        key = (uri, table_name)
        caps = self._lancedb_capabilities_cache.get(key)
        if caps is None:
            caps = inspect_lancedb_table(uri, table_name)
            self._lancedb_capabilities_cache[key] = caps
        return caps

    def _graph_lancedb_retrieve_operator(self) -> RetrieveVdbOperator | None:
        """Return the LanceDB retrieval operator from a caller-owned graph, if present."""
        if self.graph is None:
            return None

        from nemo_retriever.common.vdb.lancedb import LanceDB

        roots = getattr(self.graph, "roots", None)
        if not isinstance(roots, list):
            return None

        pending = list(roots)
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            operator = getattr(node, "operator", None)
            if isinstance(operator, RetrieveVdbOperator) and isinstance(getattr(operator, "_vdb", None), LanceDB):
                return operator
            children = getattr(node, "children", None)
            if isinstance(children, list):
                pending.extend(children)
        return None

    def _resolve_lancedb_query_mode(
        self,
        runtime_vdb_kwargs: Optional[dict[str, Any]],
    ) -> tuple[str, LanceTableCapabilities, str, str, bool] | None:
        graph_operator = self._graph_lancedb_retrieve_operator()
        if self.graph is not None and graph_operator is None:
            return None

        lancedb_kwargs = (
            {"vdb": graph_operator._vdb, "vdb_kwargs": dict(graph_operator._vdb_kwargs)}
            if graph_operator is not None
            else dict(self.vdb_kwargs or {})
        )
        if "vdb" in lancedb_kwargs:
            from nemo_retriever.common.vdb.lancedb import LanceDB

            vdb = lancedb_kwargs["vdb"]
            if not isinstance(vdb, LanceDB):
                return None
            nested_kwargs = dict(lancedb_kwargs.get("vdb_kwargs") or {})
            lancedb_kwargs = {
                "uri": vdb.uri,
                "table_name": vdb.table_name,
                "hybrid": vdb.hybrid,
                **nested_kwargs,
            }
        elif "vdb_op" in lancedb_kwargs:
            if str(lancedb_kwargs.get("vdb_op") or "").strip().lower() != "lancedb":
                return None
            lancedb_kwargs = dict(lancedb_kwargs.get("vdb_kwargs") or {})
        lancedb_kwargs.update(dict(runtime_vdb_kwargs or {}))

        uri = str(
            lancedb_kwargs.get("table_path")
            or lancedb_kwargs.get("uri")
            or lancedb_kwargs.get("lancedb_uri")
            or "lancedb"
        )
        table_name = str(lancedb_kwargs.get("table_name") or lancedb_kwargs.get("lancedb_table") or "nemo-retriever")
        caps = self._inspect_lancedb_capabilities(uri, table_name)

        mode_override = str(lancedb_kwargs.get("retrieval_mode") or "auto").strip().lower()
        if mode_override not in {"auto", "dense", "hybrid", "sparse"}:
            raise ValueError(
                f"Unsupported LanceDB retrieval mode {mode_override!r}; " "use 'auto', 'dense', 'hybrid', or 'sparse'."
            )
        if "hybrid" in lancedb_kwargs:
            mode_override = "hybrid" if bool(lancedb_kwargs["hybrid"]) else "dense"
        mode = caps.retrieval_mode if mode_override == "auto" else cast(LanceRetrievalMode, mode_override)

        if mode == "unknown":
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} is not queryable: "
                "no vector column or FTS index was detected."
            )
        if mode == "dense" and not caps.has_vector:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} cannot run dense retrieval: " "no vector column was detected."
            )
        if mode == "hybrid" and (not caps.has_vector or not caps.has_fts):
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} cannot run hybrid retrieval: "
                "both a vector column and FTS index are required."
            )
        if mode == "sparse" and not caps.has_fts:
            raise ValueError(
                f"LanceDB table {table_name!r} at {uri!r} cannot run sparse retrieval: " "no FTS index was detected."
            )

        return mode, caps, uri, table_name, mode_override != "auto"

    @staticmethod
    def _embedding_model_from_kwargs(kwargs: Optional[dict[str, Any]]) -> str | None:
        values = dict(kwargs or {})
        for key in ("model_name", "embed_model_name"):
            value = str(values.get(key) or "").strip()
            if value:
                return value
        return None

    def _resolve_embed_kwargs(
        self,
        index_model: str | None,
        runtime_embed_kwargs: Optional[dict[str, Any]],
        index_revision: str | None = None,
    ) -> dict[str, Any]:
        """Choose the query model snapshot: explicit override, index metadata, or default."""
        resolved = dict(runtime_embed_kwargs or {})
        runtime_model = self._embedding_model_from_kwargs(runtime_embed_kwargs)
        configured_model = self._embedding_model_from_kwargs(self.embed_kwargs)
        explicit_model = runtime_model or configured_model
        model_name = explicit_model or index_model
        model_name = resolve_embed_model(model_name)
        resolved["model_name"] = model_name
        resolved["embed_model_name"] = model_name
        if runtime_model and "embed_model_revision" not in resolved:
            resolved_configured = resolve_embed_model(configured_model) if configured_model else None
            if resolved_configured != model_name:
                resolved["embed_model_revision"] = None
        if explicit_model is None and index_revision:
            resolved.setdefault("embed_model_revision", index_revision)
        return resolved

    def _execute_sparse_lancedb_queries(
        self,
        query_texts: list[str],
        *,
        retrieval_top_k: int,
        vdb_call_kwargs: Optional[dict[str, Any]],
        caps: LanceTableCapabilities,
        uri: str,
        table_name: str,
    ) -> list[list[dict[str, Any]]]:
        from nemo_retriever.common.vdb.lancedb import LanceDB

        text_column = caps.text_column or "text"
        retrieval_kwargs = {
            **filter_retrieval_kwargs(dict(vdb_call_kwargs or {})),
            "top_k": int(retrieval_top_k),
            "table_path": uri,
            "table_name": table_name,
            "text_column_name": text_column,
        }
        vdb = LanceDB(uri=uri, table_name=table_name, overwrite=False, sparse=True)
        return normalize_retrieval_results(vdb.sparse_retrieval(query_texts, **retrieval_kwargs))

    def query(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        page_dedup: bool = False,
        content_types: str | Sequence[str] | None = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalHit]:
        """Run one retrieval query and return shaped hits.

        ``top_k`` is the final number of hits to return. ``candidate_k`` is the
        wider pre-filter/pre-dedup candidate pool and must be greater than or
        equal to ``top_k``. Increase it when page deduplication or content-type
        filtering would otherwise reduce the final hit count. ``page_dedup``
        keeps the first hit per document page. ``content_types`` accepts a
        comma-separated string or sequence of content types to keep, such as
        ``"text,table"``, and normalizes values to the canonical content types
        stored in hit metadata. Hits with missing or unknown content types are
        excluded while this filter is active. Page deduplication and
        content-type filtering are applied after vector retrieval, preserving
        retriever ranking order.
        """
        return self.queries(
            [query],
            top_k=top_k,
            candidate_k=candidate_k,
            page_dedup=page_dedup,
            content_types=content_types,
            vdb_kwargs=vdb_kwargs,
            embed_kwargs=embed_kwargs,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        page_dedup: bool = False,
        content_types: str | Sequence[str] | None = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[list[RetrievalHit]]:
        """Run retrieval for multiple query strings and return shaped hits.

        ``top_k`` is the final number of hits to return. ``candidate_k`` is the
        wider pre-filter/pre-dedup candidate pool and must be greater than or
        equal to ``top_k``. Increase it when page deduplication or content-type
        filtering would otherwise reduce the final hit count. ``page_dedup``
        keeps the first hit per document page. ``content_types`` accepts a
        comma-separated string or sequence of content types to keep, such as
        ``"text,table"``, and normalizes values to the canonical content types
        stored in hit metadata. Hits with missing or unknown content types are
        excluded while this filter is active. Page deduplication and
        content-type filtering are applied after vector retrieval, preserving
        retriever ranking order.
        """
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        candidate_top_k = int(candidate_k) if candidate_k is not None else effective_top_k
        if candidate_top_k < effective_top_k:
            raise ValueError(
                f"candidate_k ({candidate_top_k}) must be greater than or equal to top_k ({effective_top_k})."
            )
        refine = self._refine_factor()
        retrieval_top_k = candidate_top_k * refine if self.rerank else candidate_top_k

        vdb_call_kwargs = dict(vdb_kwargs or {})
        index_model: str | None = None
        index_revision: str | None = None
        explicit_model = self._embedding_model_from_kwargs(embed_kwargs) or self._embedding_model_from_kwargs(
            self.embed_kwargs
        )
        if self.graph is None:
            metadata_reader = RetrieveVdbOperator(**_coerce_vdb_init(self.vdb_kwargs))
        else:
            metadata_reader = self._graph_lancedb_retrieve_operator()
        if metadata_reader is not None:
            index_model = metadata_reader.get_index_metadata("embedding_model_name", **vdb_call_kwargs)
            index_revision = metadata_reader.get_index_metadata("embedding_model_revision", **vdb_call_kwargs)
            if explicit_model and index_model:
                resolved_explicit_model = resolve_embed_model(explicit_model)
                resolved_index_model = resolve_embed_model(index_model)
                if resolved_explicit_model != resolved_index_model:
                    raise ValueError(
                        f"Query embedding model {resolved_explicit_model!r} does not match model "
                        f"{resolved_index_model!r} recorded on the index. Use the index model or rebuild "
                        "the index with the configured query model."
                    )

        lancedb_mode = self._resolve_lancedb_query_mode(vdb_call_kwargs)
        if lancedb_mode is not None and lancedb_mode[0] != "sparse" and not index_model:
            raise ValueError(
                "The existing LanceDB table does not record its embedding model, so query compatibility "
                "cannot be verified. Rebuild the table with the configured embedding model before querying."
            )
        for key in _QUERY_ROUTING_VDB_KWARGS:
            vdb_call_kwargs.pop(key, None)
        if lancedb_mode is not None:
            mode, caps, uri, table_name, has_mode_override = lancedb_mode
            if mode == "sparse":
                raw_hits = self._execute_sparse_lancedb_queries(
                    query_texts,
                    retrieval_top_k=retrieval_top_k,
                    vdb_call_kwargs=vdb_call_kwargs,
                    caps=caps,
                    uri=uri,
                    table_name=table_name,
                )
                return [
                    shape_query_hits(
                        hits,
                        top_k=effective_top_k,
                        page_dedup=page_dedup,
                        content_types=content_types,
                    )
                    for hits in raw_hits
                ]
            if mode == "hybrid":
                vdb_call_kwargs["hybrid"] = True
                vdb_call_kwargs.setdefault("hybrid_fusion", DEFAULT_HYBRID_FUSION_POLICY)
            elif mode == "dense" and has_mode_override:
                vdb_call_kwargs["hybrid"] = False
            if caps.vector_column and caps.vector_column != "vector":
                vdb_call_kwargs.setdefault("vector_column_name", caps.vector_column)
        if self.graph is None:
            embed_kwargs = self._resolve_embed_kwargs(index_model, embed_kwargs, index_revision)

        raw_hits = self._execute_queries_graph(
            query_texts,
            effective_top_k=candidate_top_k,
            retrieval_top_k=retrieval_top_k,
            vdb_call_kwargs=vdb_call_kwargs,
            embed_extra=embed_kwargs,
        )
        return [
            shape_query_hits(
                hits,
                top_k=effective_top_k,
                page_dedup=page_dedup,
                content_types=content_types,
            )
            for hits in raw_hits
        ]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> "RetrievalResult":
        from nemo_retriever.models.llm.types import RetrievalResult

        hits = self.query(query, top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)

        chunks: list[str] = []
        metadata: list[dict[str, Any]] = []
        for hit in hits:
            chunks.append(str(hit.get("text", "")))
            metadata.append({k: v for k, v in hit.items() if k != "text"})
        return RetrievalResult(chunks=chunks, metadata=metadata)

    def retrieve_batch(
        self,
        queries: Sequence[str],
        *,
        top_k: Optional[int] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> list["RetrievalResult"]:
        from nemo_retriever.models.llm.types import RetrievalResult

        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        hits_per_query = self.queries(query_texts, top_k=top_k, vdb_kwargs=vdb_kwargs, embed_kwargs=embed_kwargs)

        results: list[RetrievalResult] = []
        for hits in hits_per_query:
            chunks = [str(hit.get("text", "")) for hit in hits]
            metadata = [{k: v for k, v in hit.items() if k != "text"} for hit in hits]
            results.append(RetrievalResult(chunks=chunks, metadata=metadata))
        return results

    def answer(
        self,
        query: str,
        *,
        llm: "LLMClient",
        judge: Optional["AnswerJudge"] = None,
        reference: Optional[str] = None,
        top_k: Optional[int] = None,
        reasoning_enabled: Optional[bool] = None,
        vdb_kwargs: Optional[dict[str, Any]] = None,
        embed_kwargs: Optional[dict[str, Any]] = None,
    ) -> "AnswerResult":
        from nemo_retriever.models.llm.types import (
            AnswerRequest,
            build_answer_result,
        )

        if judge is not None and reference is None:
            raise ValueError("judge requires reference")

        answer_req = AnswerRequest(
            query=query,
            top_k=int(top_k) if top_k is not None else int(self.top_k),
            reasoning_enabled=reasoning_enabled,
            reference=reference,
            judge_enabled=judge is not None,
        )
        retrieved = self.retrieve(
            answer_req.query,
            top_k=answer_req.top_k,
            vdb_kwargs=vdb_kwargs,
            embed_kwargs=embed_kwargs,
        )

        generate_kwargs: dict[str, Any] = {}
        if answer_req.reasoning_enabled is not None:
            generate_kwargs["reasoning_enabled"] = answer_req.reasoning_enabled
        gen = llm.generate(answer_req.query, retrieved.chunks, **generate_kwargs)

        return build_answer_result(
            query=answer_req.query,
            retrieval=retrieved,
            generation=gen,
            reference=answer_req.reference,
            judge=judge if answer_req.judge_enabled else None,
        )

    def pipeline(self, *, top_k: Optional[int] = None) -> "RetrieverPipelineBuilder":
        effective_top_k = int(top_k) if top_k is not None else int(self.top_k)
        return RetrieverPipelineBuilder(self, top_k=effective_top_k)


class RetrieverPipelineBuilder:
    """Fluent builder for live-RAG batch operator graphs.

    Returned from :meth:`Retriever.pipeline`.  Each builder method appends
    an :class:`~nemo_retriever.evaluation.eval_operator.EvalOperator` to an
    internal list; :meth:`run` composes them into a graph via the existing
    ``>>`` chaining and executes it on a DataFrame built from the provided
    queries.

    Example:
        >>> builder = retriever.pipeline()  # doctest: +SKIP
        >>> df = builder.generate(llm).score().judge(judge).run(  # doctest: +SKIP
        ...     queries=["q1", "q2"],
        ...     reference=["r1", "r2"],
        ... )
    """

    def __init__(self, retriever: "Retriever", *, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = int(top_k)
        self._steps: list[Any] = []

    def with_retrieval(self, *, top_k: int) -> "RetrieverPipelineBuilder":
        """Override the ``top_k`` used for the live retrieval source."""
        self._top_k = int(top_k)
        return self

    def generate(
        self,
        llm: Optional[Any] = None,
        /,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> "RetrieverPipelineBuilder":
        """Append a :class:`QAGenerationOperator` step.

        Accepts either a pre-built
        :class:`~nemo_retriever.llm.clients.LiteLLMClient` (whose transport
        and sampling params are unpacked onto the operator) or the flat
        ``model=..., api_base=..., ...`` kwargs forwarded to the operator
        constructor directly.

        Raises:
            ValueError: If neither ``llm`` nor ``model`` is provided.
        """
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        if llm is None and model is None:
            raise ValueError("generate() requires either llm= or model=")

        if llm is not None:
            transport = llm.transport
            sampling = llm.sampling
            operator = QAGenerationOperator(
                model=transport.model,
                api_base=transport.api_base,
                api_key=transport.api_key,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                extra_params=dict(transport.extra_params) if transport.extra_params else None,
                num_retries=transport.num_retries,
                timeout=transport.timeout,
                rag_system_prompt=transport.rag_system_prompt,
                rag_system_prompt_prefix=transport.rag_system_prompt_prefix,
                reasoning_enabled=getattr(transport, "reasoning_enabled", True),
            )
        else:
            operator = QAGenerationOperator(model=model, **kwargs)

        self._steps.append(operator)
        return self

    def score(self) -> "RetrieverPipelineBuilder":
        """Append a :class:`ScoringOperator` step (Tier 1 + Tier 2)."""
        from nemo_retriever.operators.graph_ops.scoring_operator import ScoringOperator

        self._steps.append(ScoringOperator())
        return self

    def judge(
        self,
        judge: Optional[Any] = None,
        /,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> "RetrieverPipelineBuilder":
        """Append a :class:`JudgingOperator` step (Tier 3).

        Accepts either a pre-built
        :class:`~nemo_retriever.llm.clients.judge.LLMJudge` (whose transport params
        are unpacked onto the operator) or the flat ``model=...`` kwargs
        forwarded to the operator constructor.

        Raises:
            ValueError: If neither ``judge`` nor ``model`` is provided.
        """
        from nemo_retriever.tools.evaluation.judging import JudgingOperator

        if judge is None and model is None:
            raise ValueError("judge() requires either judge= or model=")

        if judge is not None:
            transport = judge.transport
            operator = JudgingOperator(
                model=transport.model,
                api_base=transport.api_base,
                api_key=transport.api_key,
                extra_params=dict(transport.extra_params) if transport.extra_params else None,
                num_retries=transport.num_retries,
                timeout=transport.timeout,
            )
        else:
            operator = JudgingOperator(model=model, **kwargs)

        self._steps.append(operator)
        return self

    def run(
        self,
        queries: Any,
        *,
        reference: Any = None,
    ) -> "pd.DataFrame":
        """Execute the composed graph on ``queries``.

        Args:
            queries: A single query string, a list of query strings, or a
                pre-built ``pandas.DataFrame`` (which must contain a
                ``query`` column and, when judging/scoring, a
                ``reference_answer`` column).
            reference: Optional ground-truth answer(s).  Accepts a single
                string (applied to all queries), a list aligned with
                ``queries``, or ``None``.  Ignored when ``queries`` is
                already a DataFrame.

        Returns:
            A ``pandas.DataFrame`` with the columns contributed by each
            appended step (always ``query``, ``context``, and
            ``context_metadata``; plus ``answer``/``latency_s``/... when
            ``.generate()`` ran, and so on).

        Raises:
            ValueError: If ``reference`` is a list whose length does not
                match ``queries``.
        """
        import pandas as pd

        from nemo_retriever.tools.evaluation.live_retrieval import LiveRetrievalOperator

        if isinstance(queries, str):
            query_list = [queries]
            df = pd.DataFrame({"query": query_list})
            if reference is not None:
                refs = reference if isinstance(reference, list) else [reference]
                if len(refs) != len(query_list):
                    raise ValueError("reference length must match queries length")
                df["reference_answer"] = refs
        elif isinstance(queries, list):
            df = pd.DataFrame({"query": list(queries)})
            if reference is not None:
                refs = reference if isinstance(reference, list) else [reference] * len(queries)
                if len(refs) != len(queries):
                    raise ValueError("reference length must match queries length")
                df["reference_answer"] = refs
        elif isinstance(queries, pd.DataFrame):
            df = queries.copy()
        else:
            raise TypeError("queries must be a str, list[str], or pandas.DataFrame; " f"got {type(queries).__name__}")

        retrieval_op = LiveRetrievalOperator(self._retriever, top_k=self._top_k)
        if not self._steps:
            out = retrieval_op.run(df)
        else:
            graph = retrieval_op
            for step in self._steps:
                graph = graph >> step
            # Linear live-RAG pipelines have exactly one leaf.
            leaves = graph.execute(df)
            if len(leaves) != 1:
                raise RuntimeError(f"Unexpected pipeline fan-out: got {len(leaves)} leaf outputs")
            out = leaves[0]

        # Expose the generation failure rate on ``df.attrs`` for downstream aggregators.
        if "gen_error" in out.columns and len(out) > 0:
            out.attrs["generation_failure_rate"] = float(out["gen_error"].notna().mean())

        return out


# Backward compatibility alias.
retriever = Retriever
