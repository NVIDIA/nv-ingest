# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nemo_retriever.harness.artifact_writer import redact
from nemo_retriever.harness.artifacts import get_artifacts_root
from nemo_retriever.harness.benchmark_registry import (
    REPO_ROOT,
    benchmark_names,
    benchmark_payload,
    get_benchmark,
    get_dataset,
)
from nemo_retriever.harness.contracts import (
    EXIT_INVALID,
    EXIT_MISSING_INPUT,
    HarnessRunError,
    FailurePayload,
    MODE_TO_RUN_MODE,
)
from nemo_retriever.ingest.plan import (
    IngestCaptionOptions,
    IngestChunkOptions,
    IngestDedupOptions,
    IngestEmbedBatchOptions,
    IngestEmbedOptions,
    IngestExtractBatchOptions,
    IngestExtractOptions,
    IngestImageStoreOptions,
    IngestMediaOptions,
    IngestPlanRequest,
    IngestRuntimeOptions,
    IngestSourceOptions,
    IngestStorageOptions,
)
from nemo_retriever.ingest.service import (
    ServiceIngestCaptionOptions,
    ServiceIngestChunkOptions,
    ServiceIngestConnectionOptions,
    ServiceIngestDedupOptions,
    ServiceIngestEmbedOptions,
    ServiceIngestExtractOptions,
    ServiceIngestImageStoreOptions,
    ServiceIngestPlanRequest,
    ServiceIngestSourceOptions,
)
from nemo_retriever.query.options import (
    QueryAgenticOptions,
    QueryEmbedOptions,
    QueryRerankOptions,
    QueryRequest,
    QueryRetrievalOptions,
    QueryStorageOptions,
    QueryServiceOptions,
    ServiceQueryRequest,
)
from nemo_retriever.query.workflow import ResolvedQueryPlan


def _field_names(cls: type[Any]) -> set[str]:
    return {field.name for field in fields(cls)}


def _invalid_config(context: str, message: str) -> None:
    raise HarnessRunError(
        EXIT_INVALID,
        FailurePayload(
            failed_phase="resolve",
            failure_reason="invalid_benchmark_config",
            retryable=False,
            message=f"{context}: {message}",
        ),
    )


def _field_paths(prefix: str, cls: type[Any], *, exclude: set[str] | None = None) -> set[str]:
    excluded = exclude or set()
    return {f"{prefix}.{field.name}" for field in fields(cls) if field.name not in excluded}


def _override_child_keys(prefix: str, paths: set[str]) -> set[str]:
    dotted_prefix = f"{prefix}."
    return {path.removeprefix(dotted_prefix).split(".", 1)[0] for path in paths if path.startswith(dotted_prefix)}


INGEST_OVERRIDE_PATHS = {
    "ingest.profile",
    "ingest.input_type",
    "ingest.run_mode",
    "ingest.ray_address",
    "ingest.ray_log_to_driver",
    *_field_paths("ingest.extract", IngestExtractOptions, exclude={"batch"}),
    *_field_paths("ingest.extract.batch", IngestExtractBatchOptions),
    *_field_paths("ingest.media", IngestMediaOptions),
    *_field_paths("ingest.caption", IngestCaptionOptions),
    *_field_paths("ingest.dedup", IngestDedupOptions),
    *_field_paths("ingest.chunk", IngestChunkOptions),
    *_field_paths("ingest.embed", IngestEmbedOptions, exclude={"batch"}),
    *_field_paths("ingest.embed.batch", IngestEmbedBatchOptions),
    *_field_paths("ingest.image_store", IngestImageStoreOptions),
    *_field_paths("ingest.storage", IngestStorageOptions),
}
QUERY_OVERRIDE_PATHS = {
    "query.top_k",
    "query.candidate_k",
    "query.page_dedup",
    "query.content_types",
    "query.retrieval_mode",
    "query.embed_invoke_url",
    "query.embed_model_name",
    "query.rerank",
    "query.reranker_invoke_url",
    "query.reranker_model_name",
    "query.reranker_backend",
    "query.reranker_api_key",
    "query.lancedb_uri",
    "query.table_name",
    "query.agentic",
    "query.agentic_llm_model",
    "query.agentic_invoke_url",
    "query.agentic_local_gpu_memory_utilization",
    "query.agentic_local_tensor_parallel_size",
    "query.agentic_local_max_model_len",
    "query.agentic_local_max_num_seqs",
    "query.agentic_reasoning_effort",
    "query.agentic_react_max_steps",
    "query.agentic_text_truncation",
    "query.agentic_num_concurrent",
    "query.agentic_temperature",
    "query.agentic_llm_client",
}
EVALUATION_OVERRIDE_PATHS = {
    "evaluation.mode",
    "evaluation.loader",
    "evaluation.dataset_name",
    "evaluation.split",
    "evaluation.query_language",
    "evaluation.doc_id_field",
    "evaluation.ks",
}
DATASET_OVERRIDE_PATHS = {
    "dataset.path",
    "dataset.query_file",
    "dataset.input_type",
    "dataset.beir_loader",
    "dataset.beir_doc_id_field",
    "dataset.beir_ks",
}
ALLOWED_OVERRIDE_PATHS = (
    INGEST_OVERRIDE_PATHS | QUERY_OVERRIDE_PATHS | EVALUATION_OVERRIDE_PATHS | DATASET_OVERRIDE_PATHS
)


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_override",
                retryable=False,
                message=f"Override must be KEY=VALUE, got: {item}",
            ),
        )
    key, raw = item.split("=", 1)
    key = key.strip()
    if key not in ALLOWED_OVERRIDE_PATHS:
        suggestion = get_close_matches(key, sorted(ALLOWED_OVERRIDE_PATHS), n=1)
        suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_override",
                retryable=False,
                message=f"Unknown override key {key!r}.{suffix}",
            ),
        )
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_override",
                retryable=False,
                message=f"Could not parse override {key!r}: {exc}",
            ),
        ) from exc
    if value is None and raw.strip() not in {"null", "Null", "NULL", "none", "None", "~"}:
        value = ""
    return key, value


def _deep_set(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[parts[-1]] = value


def _mapping_payload(context: str, data: Mapping[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        _invalid_config(context, "expected an object")
    return dict(data)


def _validate_keys(context: str, data: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if not unknown:
        return
    suggestion = get_close_matches(unknown[0], sorted(allowed), n=1)
    suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    _invalid_config(context, f"unknown key {unknown[0]!r}.{suffix}")


def _resolve_repo_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def make_run_id(benchmark: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    return f"{benchmark}_{stamp}"


def resolve_artifact_dir(benchmark: str, run_id: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (get_artifacts_root() / run_id).resolve()


def resolve_benchmark(
    benchmark: str,
    *,
    mode: str = "local",
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    if mode not in MODE_TO_RUN_MODE:
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_override",
                retryable=False,
                message=f"--mode must be one of {sorted(MODE_TO_RUN_MODE)}, got {mode!r}.",
            ),
        )
    try:
        spec = get_benchmark(benchmark)
    except KeyError as exc:
        suggestion = get_close_matches(benchmark, benchmark_names(), n=1)
        suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Unknown benchmark {benchmark!r}.{suffix}",
            ),
        ) from exc

    dataset = get_dataset(spec.dataset)
    resolved: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "tags": list(spec.tags),
        "summary_keys": list(spec.summary_keys),
        "dataset": dataset.to_dict(),
        "ingest": deepcopy(dict(spec.ingest)),
        "query": deepcopy(dict(spec.query)),
        "evaluation": deepcopy(dict(spec.evaluation)),
    }
    resolved["ingest"]["run_mode"] = MODE_TO_RUN_MODE[mode]
    resolved["ingest"]["input_type"] = resolved["dataset"].get("input_type") or resolved["ingest"].get("input_type")
    for raw_override in overrides:
        key, value = _parse_override(raw_override)
        _deep_set(resolved, key, value)
    return resolved


def validate_dataset_inputs(resolved: dict[str, Any], *, dry_run: bool) -> tuple[Path, Path | None]:
    dataset = resolved["dataset"]
    dataset_path = _resolve_repo_path(dataset.get("path"))
    if dataset_path is None or not dataset_path.exists():
        raise HarnessRunError(
            EXIT_MISSING_INPUT,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="dataset_missing",
                retryable=False,
                message=f"Dataset path does not exist: {dataset.get('path')}",
            ),
        )

    query_file = dataset.get("query_file")
    evaluation = resolved.get("evaluation") or {}
    uses_remote_eval_dataset = evaluation.get("loader") == "vidore_hf"
    if evaluation.get("mode") == "beir" and not uses_remote_eval_dataset:
        query_file = evaluation.get("dataset_name") or query_file
    query_path = _resolve_repo_path(query_file) if query_file else None
    if (
        not dry_run
        and evaluation.get("mode") == "beir"
        and not uses_remote_eval_dataset
        and (query_path is None or not query_path.exists())
    ):
        raise HarnessRunError(
            EXIT_MISSING_INPUT,
            FailurePayload(
                failed_phase="query_plan",
                failure_reason="dataset_missing",
                retryable=False,
                message=f"BEIR query/qrels file does not exist: {query_file}",
            ),
        )
    dataset["resolved_path"] = str(dataset_path)
    dataset["query_file_exists"] = bool(query_path and query_path.exists())
    if query_path is not None:
        dataset["resolved_query_file"] = str(query_path)
        if evaluation.get("mode") == "beir" and not uses_remote_eval_dataset:
            evaluation["dataset_name"] = str(query_path)
    return dataset_path, query_path


def _dataclass_kwargs(context: str, cls: type[Any], data: Mapping[str, Any] | None) -> dict[str, Any]:
    allowed = _field_names(cls)
    payload = _mapping_payload(context, data)
    _validate_keys(context, payload, allowed)
    return payload


def _validate_actor_pool_mode(context: str, data: Mapping[str, Any]) -> None:
    mode = data.get("actor_pool_mode")
    if mode is not None and mode not in ("fixed", "elastic"):
        _invalid_config(context, f"actor_pool_mode must be 'fixed' or 'elastic', got {mode!r}")


def build_ingest_request(resolved: dict[str, Any], dataset_path: Path, artifact_dir: Path) -> IngestPlanRequest:
    ingest = _mapping_payload("ingest", deepcopy(resolved["ingest"]))
    _validate_keys("ingest", ingest, _override_child_keys("ingest", INGEST_OVERRIDE_PATHS))
    storage_data = _mapping_payload("ingest.storage", ingest.get("storage"))
    if not storage_data.get("lancedb_uri") or storage_data.get("lancedb_uri") == "lancedb":
        storage_data["lancedb_uri"] = str((artifact_dir / "lancedb").resolve())
    if not storage_data.get("table_name"):
        storage_data["table_name"] = "nemo-retriever"
    ingest["storage"] = storage_data
    resolved["ingest"] = ingest

    extract_data = _mapping_payload("ingest.extract", ingest.get("extract"))
    extract_batch_data = _mapping_payload("ingest.extract.batch", extract_data.pop("batch", {}))
    _validate_actor_pool_mode("ingest.extract.batch", extract_batch_data)
    extract_batch = IngestExtractBatchOptions(
        **_dataclass_kwargs("ingest.extract.batch", IngestExtractBatchOptions, extract_batch_data)
    )
    embed_data = _mapping_payload("ingest.embed", ingest.get("embed"))
    embed_batch_data = _mapping_payload("ingest.embed.batch", embed_data.pop("batch", {}))
    _validate_actor_pool_mode("ingest.embed.batch", embed_batch_data)
    embed_batch = IngestEmbedBatchOptions(
        **_dataclass_kwargs("ingest.embed.batch", IngestEmbedBatchOptions, embed_batch_data)
    )
    return IngestPlanRequest(
        source=IngestSourceOptions(
            documents=[str(dataset_path)],
            profile=ingest.get("profile", "auto"),
            input_type=ingest.get("input_type") or resolved["dataset"].get("input_type") or "pdf",
        ),
        runtime=IngestRuntimeOptions(
            run_mode=ingest.get("run_mode", "inprocess"),
            ray_address=ingest.get("ray_address"),
            ray_log_to_driver=ingest.get("ray_log_to_driver"),
        ),
        extract=IngestExtractOptions(
            **_dataclass_kwargs("ingest.extract", IngestExtractOptions, extract_data),
            batch=extract_batch,
        ),
        media=IngestMediaOptions(**_dataclass_kwargs("ingest.media", IngestMediaOptions, ingest.get("media"))),
        caption=IngestCaptionOptions(
            **_dataclass_kwargs("ingest.caption", IngestCaptionOptions, ingest.get("caption"))
        ),
        dedup=IngestDedupOptions(**_dataclass_kwargs("ingest.dedup", IngestDedupOptions, ingest.get("dedup"))),
        chunk=IngestChunkOptions(**_dataclass_kwargs("ingest.chunk", IngestChunkOptions, ingest.get("chunk"))),
        embed=IngestEmbedOptions(
            **_dataclass_kwargs("ingest.embed", IngestEmbedOptions, embed_data),
            batch=embed_batch,
        ),
        image_store=IngestImageStoreOptions(
            **_dataclass_kwargs("ingest.image_store", IngestImageStoreOptions, ingest.get("image_store"))
        ),
        storage=IngestStorageOptions(**_dataclass_kwargs("ingest.storage", IngestStorageOptions, storage_data)),
    )


def build_query_request(resolved: dict[str, Any], query_text: str) -> QueryRequest:
    query = _mapping_payload("query", resolved.get("query"))
    _validate_keys("query", query, _override_child_keys("query", QUERY_OVERRIDE_PATHS))
    ingest_storage = dict((resolved.get("ingest") or {}).get("storage") or {})
    lancedb_uri = query.get("lancedb_uri") or ingest_storage.get("lancedb_uri") or "lancedb"
    table_name = query.get("table_name") or ingest_storage.get("table_name") or "nemo-retriever"
    query["lancedb_uri"] = lancedb_uri
    query["table_name"] = table_name
    resolved["query"] = query
    return QueryRequest(
        query=query_text,
        retrieval=QueryRetrievalOptions(
            top_k=int(query.get("top_k") or 10),
            candidate_k=query.get("candidate_k"),
            page_dedup=bool(query.get("page_dedup", False)),
            content_types=query.get("content_types"),
            retrieval_mode=str(query.get("retrieval_mode", "auto")),
        ),
        embed=QueryEmbedOptions(
            embed_invoke_url=query.get("embed_invoke_url"),
            embed_model_name=query.get("embed_model_name"),
        ),
        rerank=QueryRerankOptions(
            enabled=bool(query.get("rerank", False)),
            reranker_invoke_url=query.get("reranker_invoke_url"),
            reranker_model_name=query.get("reranker_model_name"),
            reranker_backend=query.get("reranker_backend"),
            reranker_api_key=query.get("reranker_api_key"),
        ),
        storage=QueryStorageOptions(
            lancedb_uri=str(lancedb_uri),
            table_name=str(table_name),
        ),
        agentic=QueryAgenticOptions(
            enabled=bool(query.get("agentic", False)),
            llm_model=query.get("agentic_llm_model"),
            invoke_url=query.get("agentic_invoke_url"),
            local_gpu_memory_utilization=float(query.get("agentic_local_gpu_memory_utilization") or 0.8),
            local_tensor_parallel_size=int(query.get("agentic_local_tensor_parallel_size") or 1),
            local_max_model_len=query.get("agentic_local_max_model_len"),
            local_max_num_seqs=query.get("agentic_local_max_num_seqs"),
            reasoning_effort=query.get("agentic_reasoning_effort"),
            react_max_steps=int(query.get("agentic_react_max_steps") or 50),
            text_truncation=int(query.get("agentic_text_truncation") or 0),
            num_concurrent=int(query.get("agentic_num_concurrent") or 1),
            temperature=(float(query["agentic_temperature"]) if query.get("agentic_temperature") is not None else None),
            llm_client=query.get("agentic_llm_client"),
        ),
    )


def query_plan_payload(plan: ResolvedQueryPlan) -> dict[str, Any]:
    return redact(
        {
            "top_k": plan.top_k,
            "candidate_k": plan.candidate_k,
            "page_dedup": plan.page_dedup,
            "content_types": plan.content_types,
            "lancedb_uri": plan.lancedb_uri,
            "table_name": plan.table_name,
            "embed_kwargs": plan.embed_kwargs,
            "retrieval_mode": plan.retrieval_mode,
            "rerank": plan.rerank,
            "rerank_kwargs": plan.rerank_kwargs,
        }
    )


def _service_kwargs(cls: type[Any], data: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(data or {})
    return {key: value for key, value in payload.items() if key in _field_names(cls)}


def build_service_ingest_plan_request(
    resolved: dict[str, Any],
    dataset_path: Path,
    *,
    service_url: str,
    service_concurrency: int,
    service_api_token: str | None,
) -> ServiceIngestPlanRequest:
    ingest = _mapping_payload("ingest", deepcopy(resolved["ingest"]))
    extract = _mapping_payload("ingest.extract", ingest.get("extract"))
    extract.pop("batch", None)
    embed = _mapping_payload("ingest.embed", ingest.get("embed"))
    embed.pop("batch", None)
    # The deployed chart owns model selection; service requests only carry
    # per-request modality/granularity controls.
    embed.pop("embed_model_name", None)
    return ServiceIngestPlanRequest(
        source=ServiceIngestSourceOptions(
            documents=[str(dataset_path)],
            profile=ingest.get("profile", "auto"),
            input_type=ingest.get("input_type") or resolved["dataset"].get("input_type") or "pdf",
        ),
        connection=ServiceIngestConnectionOptions(
            service_url=service_url,
            service_concurrency=int(service_concurrency),
            service_api_token=service_api_token,
        ),
        extract=ServiceIngestExtractOptions(**_service_kwargs(ServiceIngestExtractOptions, extract)),
        caption=ServiceIngestCaptionOptions(**_service_kwargs(ServiceIngestCaptionOptions, ingest.get("caption"))),
        dedup=ServiceIngestDedupOptions(**_service_kwargs(ServiceIngestDedupOptions, ingest.get("dedup"))),
        chunk=ServiceIngestChunkOptions(**_service_kwargs(ServiceIngestChunkOptions, ingest.get("chunk"))),
        embed=ServiceIngestEmbedOptions(**_service_kwargs(ServiceIngestEmbedOptions, embed)),
        image_store=ServiceIngestImageStoreOptions(
            **_service_kwargs(ServiceIngestImageStoreOptions, ingest.get("image_store"))
        ),
    )


def build_service_query_request(
    resolved: dict[str, Any],
    query_text: str,
    *,
    service_url: str,
    service_api_token: str | None,
) -> ServiceQueryRequest:
    query = _mapping_payload("query", resolved.get("query"))
    _validate_keys("query", query, _override_child_keys("query", QUERY_OVERRIDE_PATHS))
    if query.get("agentic"):
        _invalid_config("query.agentic", "agentic retrieval is not supported for service-mode runs")
    return ServiceQueryRequest(
        query=query_text,
        retrieval=QueryRetrievalOptions(
            top_k=int(query.get("top_k") or 10),
            candidate_k=query.get("candidate_k"),
            page_dedup=bool(query.get("page_dedup", False)),
            content_types=query.get("content_types"),
            retrieval_mode=str(query.get("retrieval_mode", "auto")),
        ),
        service=QueryServiceOptions(
            service_url=service_url,
            service_api_token=service_api_token,
        ),
    )


def service_plan_payload(ingest_request: Any, query_request: ServiceQueryRequest) -> dict[str, Any]:
    return redact(
        {
            "service_url": ingest_request.connection.service_url,
            "service_concurrency": ingest_request.connection.service_concurrency,
            "documents": list(ingest_request.documents),
            "input_type": ingest_request.input_type,
            "query": {
                "top_k": query_request.retrieval.top_k,
                "candidate_k": query_request.retrieval.candidate_k,
                "page_dedup": query_request.retrieval.page_dedup,
                "content_types": query_request.retrieval.content_types,
            },
        }
    )


def show_benchmark_payload(benchmark: str) -> dict[str, Any]:
    try:
        return benchmark_payload(get_benchmark(benchmark))
    except KeyError as exc:
        suggestion = get_close_matches(benchmark, benchmark_names(), n=1)
        suffix = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise HarnessRunError(
            EXIT_INVALID,
            FailurePayload(
                failed_phase="resolve",
                failure_reason="invalid_benchmark",
                retryable=False,
                message=f"Unknown benchmark {benchmark!r}.{suffix}",
            ),
        ) from exc
