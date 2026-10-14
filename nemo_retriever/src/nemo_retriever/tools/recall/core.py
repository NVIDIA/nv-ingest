# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL, VL_RERANK_MODEL
from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)
AUDIO_MATCH_TOLERANCE_SECS = 2.0

import pandas as pd


def _fmt_time(seconds: float) -> str:
    """Format *seconds* as ``raw / H:MM:SS.mmm``."""
    ms = int(round(seconds * 1000))
    h, remainder = divmod(ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, millis = divmod(remainder, 1000)
    return f"{seconds:.2f}s / {h}:{m:02d}:{s:02d}.{millis:03d}"


def _evaluation_metric_sort_key(item: tuple[str, float]) -> tuple[str, int, str]:
    """Sort metrics like ndcg@1, ndcg@3, ..., recall@1, recall@3, ... ."""
    key, _value = item
    metric_name, sep, suffix = str(key).partition("@")
    if sep:
        try:
            return metric_name, int(suffix), str(key)
        except ValueError:
            pass
    return metric_name, 0, str(key)


def print_run_summary(
    processed_pages: Optional[int],
    input_path: Path,
    vdb_op: str,
    vdb_kwargs: Optional[Dict[str, Any]],
    total_time: float,
    ingest_only_total_time: float,
    ray_dataset_download_total_time: float,
    vdb_upload_total_time: float,
    evaluation_total_time: float = 0.0,
    evaluation_metrics: Optional[Dict[str, float]] = None,
    recall_total_time: float = 0.0,
    recall_metrics: Optional[Dict[str, float]] = None,
    processed_files: Optional[int] = None,
    evaluation_label: str = "Recall",
    evaluation_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Print a human-readable run summary and return all metrics as a dict."""
    if recall_metrics is None:
        recall_metrics = {}
    if evaluation_metrics is None:
        evaluation_metrics = {}
    pages = processed_pages if processed_pages is not None else 0

    ingest_only_pps = pages / ingest_only_total_time if ingest_only_total_time > 0 else 0
    ingest_write_denom = ingest_only_total_time + vdb_upload_total_time
    ingest_and_vdb_upload_pps = pages / ingest_write_denom if ingest_write_denom > 0 else 0
    recall_qps = pages / recall_total_time if recall_total_time > 0 else 0
    total_pps = pages / total_time if total_time > 0 else 0
    vdb_kwargs = dict(vdb_kwargs or {})

    print(f"===== Run Summary - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC =====")

    print("Run Configuration:")
    print(f"\tInput path: {input_path}")
    print(f"\tVDB op: {vdb_op}")
    if vdb_kwargs:
        print(f"\tVDB kwargs: {json.dumps(vdb_kwargs, default=str, sort_keys=True)}")

    print("Runtimes:")
    if processed_files is not None:
        print(f"\tTotal files processed: {processed_files}")
    print(f"\tTotal pages processed: {pages} from {input_path}")
    print(f"\tIngestion only time: {_fmt_time(ingest_only_total_time)}")
    print(f"\tRay dataset download time: {_fmt_time(ray_dataset_download_total_time)}")
    print(f"\tVDB upload time: {_fmt_time(vdb_upload_total_time)}")
    if recall_total_time > 0:
        print(f"\tRecall time: {_fmt_time(recall_total_time)}")
    if evaluation_total_time > 0:
        print(f"\t{evaluation_label} time: {_fmt_time(evaluation_total_time)}")

    print("PPS:")
    print(f"\tIngestion only PPS: {ingest_only_pps:.2f}")
    print(f"\tIngestion + VDB upload PPS: {ingest_and_vdb_upload_pps:.2f}")
    if recall_total_time > 0:
        print(f"\tRecall QPS: {recall_qps:.2f}")
    print(f"\tTotal - Processed: {pages} pages in {_fmt_time(total_time)} @ {total_pps:.2f} PPS")

    if recall_metrics:
        print("Recall metrics:")
        for k, v in sorted(recall_metrics.items(), key=_evaluation_metric_sort_key):
            print(f"  {k}: {v:.4f}")
    elif not evaluation_metrics:
        print("Recall metrics: skipped (no query CSV configured)")

    if evaluation_metrics:
        print(f"{evaluation_label} metrics:")
        for k, v in sorted(evaluation_metrics.items(), key=_evaluation_metric_sort_key):
            print(f"  {k}: {v:.4f}")

    return {
        "pages": pages,
        "files": processed_files,
        "ingest_secs": round(ingest_only_total_time, 4),
        "pages_per_sec_ingest": round(ingest_only_pps, 4),
        "total_time_secs": round(total_time, 4),
        "total_pps": round(total_pps, 4),
        "ray_dataset_download_secs": round(ray_dataset_download_total_time, 4),
        "vdb_op": str(vdb_op),
        "vdb_kwargs": vdb_kwargs,
        "vdb_upload_secs": round(vdb_upload_total_time, 4),
        "recall_time_secs": round(recall_total_time, 4),
        "evaluation_time_secs": round(evaluation_total_time, 4),
        "evaluation_label": evaluation_label,
        "evaluation_count": evaluation_count,
        "recall_metrics": recall_metrics,
        "evaluation_metrics": evaluation_metrics,
    }


@dataclass(frozen=True)
class RecallConfig:
    vdb_op: str = "lancedb"
    vdb_kwargs: dict[str, Any] = field(default_factory=dict)
    query_embedder: str | None = None
    embedding_http_endpoint: Optional[str] = None
    embedding_grpc_endpoint: Optional[str] = None
    embedding_endpoint: Optional[str] = None
    embedding_api_key: str = ""
    embedding_use_grpc: Optional[bool] = None
    top_k: int = 10
    ks: Sequence[int] = (1, 3, 5, 10)
    local_hf_device: Optional[str] = None
    local_hf_cache_dir: Optional[str] = None
    local_hf_batch_size: int = 32
    local_query_max_length: int = 128
    # When using local query embedding (no HTTP endpoint), select backend for *queries* only.
    # ``vllm`` (default) uses the local vLLM text embedder;
    # ``hf`` uses the HF mean-pooled text embedder. Ignored when an
    # embedding HTTP endpoint is set.
    local_query_embed_backend: str = "vllm"
    # Gold/retrieval comparison mode:
    # - pdf_page: compare on "{pdf}_{page}" keys
    # - pdf_only: compare on "{pdf}" document keys
    # - audio_segment: compare on "media_id<TAB>start<TAB>end" segment keys
    match_mode: str = "pdf_page"
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS
    reranker: Optional[str] = None
    reranker_endpoint: Optional[str] = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32
    local_reranker_backend: str = "vllm"
    embed_modality: str = "text"

    def __post_init__(self) -> None:
        from nemo_retriever.models import (
            _LOCAL_QUERY_BACKENDS,
            _LOCAL_RERANKER_BACKENDS,
            normalize_backend,
        )

        # frozen=True: must use object.__setattr__ to write normalized values.
        object.__setattr__(
            self,
            "local_query_embed_backend",
            normalize_backend(
                self.local_query_embed_backend,
                _LOCAL_QUERY_BACKENDS,
                field_name="local_query_embed_backend",
                default="vllm",
            ),
        )
        object.__setattr__(
            self,
            "local_reranker_backend",
            normalize_backend(
                self.local_reranker_backend,
                _LOCAL_RERANKER_BACKENDS,
                field_name="local_reranker_backend",
                default="vllm",
            ),
        )


def _normalize_pdf_name(value: str) -> str:
    return str(value).replace(".pdf", "")


def _normalize_audio_media_id(value: object) -> str:
    basename = Path(str(value)).name
    return basename.split(".", 1)[0] if basename else ""


def _encode_audio_segment_key(media_id: str, start_time: float, end_time: float) -> str:
    return f"{media_id}\t{float(start_time):.6f}\t{float(end_time):.6f}"


def _parse_audio_segment_key(key: str) -> tuple[str, float, float]:
    parts = str(key).split("\t")
    if len(parts) != 3:
        raise ValueError(f"Invalid audio segment key: {key!r}")
    media_id, start_time, end_time = parts
    return media_id, float(start_time), float(end_time)


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def _normalize_audio_segment_times(
    start_time: Any,
    end_time: Any,
    *,
    duration_hint_secs: Any = None,
) -> tuple[float, float] | None:
    try:
        start_val = float(start_time)
        end_val = float(end_time)
    except (TypeError, ValueError):
        return None

    duration_secs: float | None
    try:
        duration_secs = float(duration_hint_secs) if duration_hint_secs is not None else None
    except (TypeError, ValueError):
        duration_secs = None

    # Audio stage metadata currently stores segment times in milliseconds.
    # Normalize those to seconds when they obviously exceed the chunk duration.
    if duration_secs is not None and duration_secs > 0 and end_val > (duration_secs + 1.0):
        return start_val / 1000.0, end_val / 1000.0

    return start_val, end_val


def _normalize_audio_query_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "query" not in df.columns and "question" in df.columns:
        df = df.rename(columns={"question": "query"})
    if "expected_media_id" not in df.columns and "name" in df.columns:
        df["expected_media_id"] = df["name"].astype(str).apply(_normalize_audio_media_id)
    if "expected_start_time" not in df.columns and "start_time" in df.columns:
        df["expected_start_time"] = pd.to_numeric(df["start_time"], errors="raise").astype(float)
    if "expected_end_time" not in df.columns and "end_time" in df.columns:
        df["expected_end_time"] = pd.to_numeric(df["end_time"], errors="raise").astype(float)

    required = {"query", "expected_media_id", "expected_start_time", "expected_end_time"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "For audio_segment mode, query data must contain "
            "['query','expected_media_id','expected_start_time','expected_end_time'] "
            "or ['question','name','start_time','end_time'] columns "
            f"(missing: {sorted(missing)})"
        )

    df["query"] = df["query"].astype(str)
    df["expected_media_id"] = df["expected_media_id"].astype(str).apply(_normalize_audio_media_id)
    df["expected_start_time"] = pd.to_numeric(df["expected_start_time"], errors="raise").astype(float)
    df["expected_end_time"] = pd.to_numeric(df["expected_end_time"], errors="raise").astype(float)
    df["golden_answer"] = df.apply(
        lambda row: _encode_audio_segment_key(
            row["expected_media_id"], row["expected_start_time"], row["expected_end_time"]
        ),
        axis=1,
    )
    return df


def _normalize_query_df(df: pd.DataFrame, *, match_mode: str) -> pd.DataFrame:
    """
    Normalize a query CSV into:
      - query (string)
      - golden_answer (string key that should match LanceDB `pdf_page`)

    Supported inputs by match mode:
      - pdf_page:
        - query,pdf_page
        - query,pdf,page (or query,pdf,gt_page)
      - pdf_only:
        - query,expected_pdf
        - query,pdf
      - audio_segment:
        - query,expected_media_id,expected_start_time,expected_end_time
        - question,name,start_time,end_time
    """
    if match_mode not in {"pdf_page", "pdf_only", "audio_segment"}:
        raise ValueError(f"Unsupported recall match mode: {match_mode}")

    if match_mode == "audio_segment":
        return _normalize_audio_query_df(df)

    df = df.copy()

    if "query" not in df.columns:
        raise KeyError("Query CSV must contain a 'query' column.")

    if match_mode == "pdf_only":
        if "expected_pdf" in df.columns:
            df["golden_answer"] = df["expected_pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        if "pdf" in df.columns:
            df["golden_answer"] = df["pdf"].astype(str).apply(_normalize_pdf_name)
            return df
        raise KeyError(
            "For pdf_only mode, query data must contain ['query','expected_pdf'] or ['query','pdf'] columns."
        )

    if "gt_page" in df.columns and "page" not in df.columns:
        df = df.rename(columns={"gt_page": "page"})

    if "pdf_page" in df.columns:
        df["golden_answer"] = df["pdf_page"].astype(str)
        return df

    required = {"pdf", "page"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            "Query CSV must contain either columns ['query','pdf_page'] or ['query','pdf','page'] "
            f"(missing: {sorted(missing)})"
        )

    df["pdf"] = df["pdf"].astype(str).str.replace(".pdf", "", regex=False)
    df["page"] = df["page"].astype(str)
    df["golden_answer"] = df.apply(lambda x: f"{x.pdf}_{x.page}", axis=1)
    return df


def _resolve_embedding_endpoint(cfg: RecallConfig) -> Tuple[Optional[str], Optional[bool]]:
    """
    Resolve which embedding endpoint to use.

    Returns (endpoint, use_grpc) where:
      - endpoint is either an http(s) URL or a host:port string for gRPC
      - use_grpc is True for gRPC, False for HTTP, None when no endpoint is configured
    """
    http_ep = (cfg.embedding_http_endpoint or "").strip() if isinstance(cfg.embedding_http_endpoint, str) else None
    grpc_ep = (cfg.embedding_grpc_endpoint or "").strip() if isinstance(cfg.embedding_grpc_endpoint, str) else None
    single = (cfg.embedding_endpoint or "").strip() if isinstance(cfg.embedding_endpoint, str) else None

    if http_ep:
        return http_ep, False
    if grpc_ep:
        return grpc_ep, True
    if single:
        if cfg.embedding_use_grpc is not None:
            return single, bool(cfg.embedding_use_grpc)
        # Infer protocol: if a URL scheme is present, treat as HTTP; otherwise gRPC.
        return single, (not single.lower().startswith("http"))

    return None, None


def _hits_to_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for h in hits:
            page_number = h["page_number"]
            source = h["source"]
            # Prefer explicit `pdf_page` column; fall back to derived form.
            if page_number is not None and source:
                filename = Path(source).stem
                keys.append(f"{filename}_{str(page_number)}")
            else:
                logger.warning(
                    "Skipping hit with missing page_number or source_id: metadata=%s source=%s",
                    h.get("metadata", ""),
                    h.get("source", ""),
                )
        retrieved_keys.append([k for k in keys if k])
    return retrieved_keys


def _hit_to_audio_segment_key(hit: Dict[str, Any]) -> str | None:
    metadata = _parse_mapping(hit.get("metadata"))
    source = _parse_mapping(hit.get("source"))

    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        source_id = hit.get("source_id") if isinstance(hit.get("source_id"), str) else hit.get("source")
    if not isinstance(source_id, str) or not source_id.strip():
        return None

    media_id = _normalize_audio_media_id(source_id)
    if not media_id:
        return None

    # Prefer the canonical "_seconds" keys (always wall-clock seconds).
    start_time = metadata.get("segment_start_seconds")
    end_time = metadata.get("segment_end_seconds")
    if start_time is not None and end_time is not None:
        try:
            return _encode_audio_segment_key(media_id, float(start_time), float(end_time))
        except (TypeError, ValueError):
            return None

    start_time = metadata.get("segment_start")
    end_time = metadata.get("segment_end")
    if start_time is not None and end_time is not None:
        normalized = _normalize_audio_segment_times(
            start_time,
            end_time,
            duration_hint_secs=metadata.get("duration"),
        )
        if normalized is None:
            return None
        start_secs, end_secs = normalized
        return _encode_audio_segment_key(media_id, start_secs, end_secs)

    content_metadata = metadata.get("content_metadata")
    if isinstance(content_metadata, dict):
        start_time = content_metadata.get("start_time")
        end_time = content_metadata.get("end_time")
        if start_time is not None and end_time is not None:
            normalized = _normalize_audio_segment_times(
                start_time,
                end_time,
                duration_hint_secs=metadata.get("duration"),
            )
            if normalized is None:
                return None
            start_secs, end_secs = normalized
            return _encode_audio_segment_key(media_id, start_secs, end_secs)

    return None


def _hits_to_audio_segment_keys(raw_hits: List[List[Dict[str, Any]]]) -> List[List[str]]:
    retrieved_keys: List[List[str]] = []
    for hits in raw_hits:
        keys: List[str] = []
        for hit in hits:
            encoded = _hit_to_audio_segment_key(hit)
            if encoded is not None:
                keys.append(encoded)
        retrieved_keys.append(keys)
    return retrieved_keys


def _extract_doc_from_pdf_page(key: str) -> str:
    parts = str(key).rsplit("_", 1)
    if len(parts) != 2:
        return str(key)
    return parts[0]


def _is_hit(
    golden_key: str,
    retrieved: List[str],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> bool:
    """Check if a golden key is found in the top-k retrieved keys.

    Handles filenames with underscores via ``rsplit`` and also accepts
    whole-document keys (page ``-1``).
    """
    if match_mode == "audio_segment":
        gold_media, gold_start, gold_end = _parse_audio_segment_key(golden_key)
        for encoded_hit in retrieved[:k]:
            try:
                hit_media, hit_start, hit_end = _parse_audio_segment_key(encoded_hit)
            except ValueError:
                continue
            hit_midpoint = (hit_start + hit_end) / 2.0
            if (
                hit_media == gold_media
                and hit_midpoint > (gold_start - float(audio_match_tolerance_secs))
                and hit_midpoint < (gold_end + float(audio_match_tolerance_secs))
            ):
                return True
        return False

    if match_mode == "pdf_only":
        gold_doc = _normalize_pdf_name(str(golden_key))
        top_docs = [_extract_doc_from_pdf_page(r) for r in retrieved[:k]]
        return gold_doc in top_docs

    parts = golden_key.rsplit("_", 1)
    if len(parts) != 2:
        return golden_key in retrieved[:k]
    filename, page = parts
    specific_page = f"{filename}_{page}"
    entire_document = f"{filename}_-1"
    top = retrieved[:k]
    return specific_page in top or entire_document in top


def is_hit_at_k(
    golden_key: str,
    retrieved: Sequence[str],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> bool:
    """Public wrapper for top-k hit checks across match modes."""
    return _is_hit(
        str(golden_key),
        list(retrieved),
        int(k),
        match_mode=str(match_mode),
        audio_match_tolerance_secs=float(audio_match_tolerance_secs),
    )


def _recall_at_k(
    gold: List[str],
    retrieved: List[List[str]],
    k: int,
    *,
    match_mode: str,
    audio_match_tolerance_secs: float = AUDIO_MATCH_TOLERANCE_SECS,
) -> float:
    hits = sum(
        is_hit_at_k(g, r, k, match_mode=match_mode, audio_match_tolerance_secs=audio_match_tolerance_secs)
        for g, r in zip(gold, retrieved)
    )
    return hits / max(1, len(gold))


def retrieve_and_score(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], List[List[Dict[str, Any]]], List[List[str]], Dict[str, float]]:
    """
    Run VDB retrieval for a query CSV.

    Returns:
      - normalized query DataFrame
      - gold keys
      - raw VDB hits
      - retrieved keys (pdf_page-like or audio-segment-like)
      - metrics dict (recall@k)
    """
    df_query = _normalize_query_df(pd.read_csv(query_csv), match_mode=str(cfg.match_mode))
    if limit is not None:
        df_query = df_query.head(int(limit)).copy()

    queries = df_query["query"].astype(str).tolist()
    gold = df_query["golden_answer"].astype(str).tolist()
    vdb_inner = dict(cfg.vdb_kwargs or {})
    embedding_endpoint, embedding_use_grpc = _resolve_embedding_endpoint(cfg)
    query_embedder = cfg.query_embedder or (NEMOTRON_3_EMBED_MODEL if embedding_endpoint else None)
    if embedding_use_grpc:
        raise ValueError(
            "RecallConfig gRPC query embedding is not supported with the graph-based Retriever. "
            "Use embedding_http_endpoint or an http(s) embedding_endpoint."
        )

    from nemo_retriever.common.params.models import ModelRuntimeParams

    embed_kwargs: dict[str, Any] = {
        "model_name": query_embedder,
        "embed_model_name": query_embedder,
        "inference_batch_size": int(cfg.local_hf_batch_size),
        "embed_inference_batch_size": int(cfg.local_hf_batch_size),
        "local_ingest_embed_backend": str(cfg.local_query_embed_backend),
        "embed_modality": str(cfg.embed_modality),
        "query_max_length": int(cfg.local_query_max_length),
    }
    if embedding_endpoint:
        embed_kwargs["embedding_endpoint"] = embedding_endpoint
        embed_kwargs["embed_invoke_url"] = embedding_endpoint
    if (cfg.embedding_api_key or "").strip():
        embed_kwargs["api_key"] = (cfg.embedding_api_key or "").strip()
    if cfg.local_hf_device or cfg.local_hf_cache_dir:
        embed_kwargs["runtime"] = ModelRuntimeParams(
            device=str(cfg.local_hf_device) if cfg.local_hf_device else None,
            hf_cache_dir=str(cfg.local_hf_cache_dir) if cfg.local_hf_cache_dir else None,
        )

    rerank_kw: dict[str, Any] = {
        "model_name": str(cfg.reranker or VL_RERANK_MODEL),
        "rerank_invoke_url": (cfg.reranker_endpoint or "").strip() or None,
        "api_key": (cfg.reranker_api_key or "").strip(),
        "batch_size": int(cfg.reranker_batch_size),
        "local_reranker_backend": str(cfg.local_reranker_backend),
    }

    retriever = Retriever(
        vdb_kwargs={"vdb_op": str(cfg.vdb_op), "vdb_kwargs": vdb_inner},
        embed_kwargs=embed_kwargs,
        top_k=cfg.top_k,
        rerank=bool(cfg.reranker),
        rerank_kwargs=rerank_kw,
    )
    start = time.time()
    raw_hits = retriever.queries(queries)
    end_queries = time.time() - start
    print(
        f"Retrieval time for {len(queries)} ",
        f"queries: {end_queries:.2f} seconds ",
        f"(average {len(queries)/end_queries:.2f} queries/second)",
    )

    if str(cfg.match_mode) == "audio_segment":
        retrieved_keys = _hits_to_audio_segment_keys(raw_hits)
    else:
        retrieved_keys = _hits_to_keys(raw_hits)
    metrics = {
        f"recall@{k}": _recall_at_k(
            gold,
            retrieved_keys,
            int(k),
            match_mode=str(cfg.match_mode),
            audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
        )
        for k in cfg.ks
    }
    return df_query, gold, raw_hits, retrieved_keys, metrics


def evaluate_recall(
    query_csv: Path,
    *,
    cfg: RecallConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    df_query, gold, raw_hits, retrieved_keys, metrics = retrieve_and_score(
        query_csv,
        cfg=cfg,
        limit=None,
    )

    # Build per-query analysis DataFrame
    rows = []
    for i, (q, g, r) in enumerate(zip(df_query["query"].astype(str).tolist(), gold, retrieved_keys)):
        row = {"query_id": i, "query": q, "golden_answer": g, "top_retrieved": r[: cfg.top_k]}
        for k in cfg.ks:
            k = int(k)
            row[f"hit@{k}"] = is_hit_at_k(
                g,
                r,
                k,
                match_mode=str(cfg.match_mode),
                audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
            )
            if str(cfg.match_mode) == "audio_segment":
                rank = None
                for index, encoded_hit in enumerate(r[: cfg.top_k], start=1):
                    if is_hit_at_k(
                        g,
                        [encoded_hit],
                        1,
                        match_mode="audio_segment",
                        audio_match_tolerance_secs=float(cfg.audio_match_tolerance_secs),
                    ):
                        rank = index
                        break
                row[f"rank@{k}"] = rank
            elif str(cfg.match_mode) == "pdf_only":
                top_docs = [_extract_doc_from_pdf_page(key) for key in r[: cfg.top_k]]
                try:
                    row[f"rank@{k}"] = top_docs.index(_normalize_pdf_name(str(g))) + 1
                except ValueError:
                    row[f"rank@{k}"] = None
            else:
                row[f"rank@{k}"] = (r[: cfg.top_k].index(g) + 1) if (g in r[: cfg.top_k]) else None
        rows.append(row)
    results_df = pd.DataFrame(rows)

    saved: Dict[str, str] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = output_dir / f"recall_results_{ts}.csv"
        results_df.to_csv(out, index=False)
        saved["results_csv"] = str(out)

    return {
        "n_queries": int(len(df_query)),
        "top_k": int(cfg.top_k),
        "metrics": metrics,
        "saved": saved,
    }
