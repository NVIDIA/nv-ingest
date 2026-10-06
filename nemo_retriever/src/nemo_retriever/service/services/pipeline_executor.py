# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge between the service layer and the nemo-retriever pipeline.

Builds ``ExtractParams`` / ``EmbedParams`` from :class:`ServiceConfig` and
returns async work functions suitable for :class:`_Pool` worker loops.

Each work function:

1. Constructs a fresh :class:`GraphIngestor` per item (cheap — just sets
   Python attributes).
2. Feeds the raw bytes via ``.buffers()`` so no temp files are needed.
3. Runs the synchronous ``InprocessExecutor`` pipeline in a **child
   process** via :class:`concurrent.futures.ProcessPoolExecutor` to
   isolate PDFium's non-thread-safe C library.
4. Returns a lightweight summary of the result rows for status polling.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from nemo_retriever.common.url_fetch import restore_url_source_value

if TYPE_CHECKING:
    from nemo_retriever.service.config import (
        LocalModelsConfig,
        NimEndpointsConfig,
        ServiceConfig,
    )
    from nemo_retriever.service.services.pipeline_pool import DocumentWriteContext, WorkItem

logger = logging.getLogger(__name__)

_MP_CONTEXT = mp.get_context("forkserver")
_MAX_TASKS_PER_CHILD = 100
_DEFAULT_WARM_MAX_TASKS_PER_CHILD = 10_000
# Storage acknowledgement is part of the ingest contract, so the worker waits
# long enough for a busy VectorDB to commit rather than reporting a success
# the storage tier never confirmed.
_DEFAULT_VECTORDB_WRITE_TIMEOUT_S = 300.0

_SENSITIVE_PATTERNS = frozenset(
    {
        "api_key",
        "password",
        "secret",
        "token",
        "credential",
    }
)


def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *d* with sensitive-looking values masked."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if any(pat in k.lower() for pat in _SENSITIVE_PATTERNS):
            out[k] = "***REDACTED***" if v else None
        elif isinstance(v, dict):
            out[k] = _redact_dict(v)
        else:
            out[k] = v
    return out


def _params_to_dict(params: Any) -> dict[str, Any]:
    """Serialize a Pydantic params model to a redacted dict."""
    if params is None:
        return {}
    raw = params.model_dump(mode="json") if hasattr(params, "model_dump") else {}
    return _redact_dict(raw)


_pipeline_configs: dict[str, dict[str, Any]] = {}


def get_pipeline_configs() -> dict[str, dict[str, Any]]:
    """Return the captured pipeline configurations (populated at startup)."""
    return _pipeline_configs


def _sanitize_result_data(
    df: Any,
    *,
    result_schema: str = "legacy",
    return_embeddings: bool = False,
    return_images: bool = False,
    embedding_column: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a pipeline DataFrame to JSON-safe dicts for the status API.

    ``result_schema="legacy"`` preserves the in-process
    ``GraphIngestor.ingest()`` column layout with bulky values stripped.
    ``result_schema="compact"`` emits the opt-in compact public shape.
    """
    from nemo_retriever.ingestor.results import dataframe_to_transport_records

    return dataframe_to_transport_records(
        df,
        result_schema=result_schema,
        return_embeddings=return_embeddings,
        embedding_column=embedding_column,
        return_images=return_images,
    )


def _pipeline_tracing() -> Any | None:
    try:
        from nemo_retriever.service import tracing
    except Exception:
        logger.debug("Service tracing helper unavailable for pipeline execution", exc_info=True)
        return None
    return tracing


def _capture_trace_context_for_pipeline() -> dict[str, str]:
    tracing = _pipeline_tracing()
    if tracing is None:
        return {}
    try:
        return dict(tracing.inject_trace_context())
    except Exception as exc:
        logger.warning("OpenTelemetry trace context capture failed for pipeline execution: %s", exc)
        return {}


def _extract_trace_context_for_pipeline(
    trace_context: dict[str, str] | None,
) -> Any | None:
    if not trace_context:
        return None
    tracing = _pipeline_tracing()
    if tracing is None:
        return None
    try:
        return tracing.extract_trace_context(trace_context)
    except Exception as exc:
        logger.warning(
            "OpenTelemetry trace context extraction failed for pipeline execution: %s",
            exc,
        )
        return None


# ── Process pool registry ────────────────────────────────────────────

_process_executors: list[ProcessPoolExecutor] = []
_executor_warmup_targets: list[tuple[ProcessPoolExecutor, int]] = []
_service_warmup_state: dict[str, Any] = {
    "enabled": False,
    "complete": False,
    "workers_expected": 0,
    "workers_warm": 0,
    "error": None,
}


def _pool_worker_initializer(warmup_spec_json: str) -> None:
    """Process-pool child initializer — loads HF models when warmup is enabled."""
    if not warmup_spec_json:
        return
    from nemo_retriever.models.warmup_registry import warm_local_models

    warm_local_models(json.loads(warmup_spec_json))


def _pool_worker_ping(_: int) -> bool:
    """No-op task used to force worker processes to start (and run initializer)."""
    from nemo_retriever.models.warmup_registry import is_warmup_active

    return is_warmup_active()


def _resolve_max_tasks_per_child(local: "LocalModelsConfig") -> int:
    if local.enabled and local.warmup_on_startup:
        if local.max_tasks_per_child is not None:
            return local.max_tasks_per_child
        return _DEFAULT_WARM_MAX_TASKS_PER_CHILD
    return _MAX_TASKS_PER_CHILD


def _build_pool_warmup_spec_json(
    local: "LocalModelsConfig",
    extract_params_dict: dict[str, Any],
    embed_params_dict: dict[str, Any] | None,
    asr_params_dict: dict[str, Any] | None,
) -> str:
    if not (local.enabled and local.warmup_on_startup):
        return ""
    from nemo_retriever.models.warmup_registry import build_warmup_spec

    spec = build_warmup_spec(extract_params_dict, embed_params_dict, asr_params_dict)
    if spec is None:
        return ""
    return json.dumps(spec)


def _create_process_executor(
    *,
    num_workers: int,
    max_tasks_per_child: int,
    warmup_spec_json: str,
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=_MP_CONTEXT,
        max_tasks_per_child=max_tasks_per_child,
        initializer=_pool_worker_initializer,
        initargs=(warmup_spec_json,),
    )


def get_service_warmup_status() -> dict[str, Any]:
    """Return startup warmup progress for health/readiness probes."""
    return dict(_service_warmup_state)


def warmup_process_pool_workers() -> dict[str, Any]:
    """Force-spawn all pipeline process-pool workers and wait for model warmup."""
    if not _executor_warmup_targets:
        _service_warmup_state["complete"] = True
        return dict(_service_warmup_state)

    workers_expected = sum(n for _, n in _executor_warmup_targets)
    _service_warmup_state["workers_expected"] = workers_expected
    workers_warm = 0
    try:
        for executor, num_workers in _executor_warmup_targets:
            futures = [executor.submit(_pool_worker_ping, i) for i in range(num_workers)]
            for future in futures:
                if future.result(timeout=300):
                    workers_warm += 1
        _service_warmup_state["workers_warm"] = workers_warm
        _service_warmup_state["complete"] = workers_warm == workers_expected
        if not _service_warmup_state["complete"]:
            _service_warmup_state["error"] = f"only {workers_warm}/{workers_expected} workers reported warmed models"
    except Exception as exc:
        _service_warmup_state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Local model warmup failed")
        raise
    return dict(_service_warmup_state)


def shutdown_process_executors() -> None:
    """Shut down all process pool executors created by work-function factories.

    Called during application shutdown (before the asyncio pool is torn down)
    so that child processes are reaped cleanly.  Actively kills running
    child processes so shutdown is not blocked by long-running pipelines.
    """
    import os
    import signal

    for executor in _process_executors:
        # Kill running child processes immediately so blocked
        # run_in_executor() futures unblock.
        pids: list[int] = []
        if hasattr(executor, "_processes"):
            pids = list(executor._processes.keys())
        executor.shutdown(wait=False, cancel_futures=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    _process_executors.clear()
    _executor_warmup_targets.clear()
    _service_warmup_state.update(
        {
            "complete": False,
            "workers_expected": 0,
            "workers_warm": 0,
            "error": None,
        }
    )
    logger.info("All pipeline process executors shut down")


def _post_records_to_vectordb(
    records: list[list[dict[str, Any]]],
    vectordb_url: str,
    filename: str,
    *,
    context: DocumentWriteContext,
    job_id: str | None = None,
    internal_api_token: str | None = None,
    timeout_s: float = _DEFAULT_VECTORDB_WRITE_TIMEOUT_S,
) -> None:
    """Post canonical NRL record batches to VectorDB and fail the job on rejection.

    A write the storage tier never acknowledged cannot be reported as a
    completed ingest: the caller would see a row count for records that are
    not queryable. Both collection-managed and legacy fixed-table writes
    therefore surface the storage failure to the document.
    """
    import json
    import urllib.request
    import urllib.error
    from nemo_retriever.service.auth import internal_auth_headers

    if not records or not any(records):
        if context.collection_name:
            raise ValueError(f"No vector rows were produced for collection document {filename}")
        return

    url = vectordb_url.rstrip("/") + "/internal/vectordb/write"
    body = json.dumps(
        {
            "records": records,
            "scope": context.scope,
            "collection_name": context.collection_name,
            "document_id": context.storage_document_id,
            "job_id": job_id,
            "filename": filename,
            "content_sha256": context.content_sha256,
            "document_version": context.resolved_version,
            "operation": context.operation,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            **internal_auth_headers(internal_api_token),
        },
        method="POST",
    )
    record_count = sum(len(batch) for batch in records)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            logging.getLogger(__name__).info(
                "Posted %d records to vectordb for %s — HTTP %d",
                record_count,
                filename,
                resp.status,
            )
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Failed to POST %d records to vectordb for %s: %s",
            record_count,
            filename,
            exc,
        )
        if context.collection_name:
            raise RuntimeError(f"Collection write failed for {filename}: {exc}") from exc
        raise RuntimeError(
            f"VectorDB write failed for {filename}: {exc}. "
            "The extracted rows are not confirmed durable, so the document is not queryable."
        ) from exc


_TRUST_OWNED_EXTRACT_KEYS: tuple[str, ...] = (
    "invoke_url",
    "api_key",
    "page_elements_invoke_url",
    "page_elements_api_key",
    "ocr_invoke_url",
    "ocr_api_key",
    "table_structure_invoke_url",
    "nemotron_parse_invoke_url",
    "nemotron_parse_model",
)
_TRUST_OWNED_EMBED_KEYS: tuple[str, ...] = (
    "embed_invoke_url",
    "embedding_endpoint",
    "api_key",
    "embed_model_name",
    "embed_model_provider_prefix",
    "model_name",
)
# Trust-owned caption keys. ``endpoint_url`` / ``api_key`` /
# ``model_name`` are all set by the operator via NimEndpointsConfig and
# can never be redirected per-request.
_TRUST_OWNED_CAPTION_KEYS: tuple[str, ...] = (
    "endpoint_url",
    "api_key",
    "model_name",
)


def _merge_server_owned(
    base: dict[str, Any], override: dict[str, Any] | None, owned: tuple[str, ...]
) -> dict[str, Any]:
    """Merge *override* on top of *base* while preserving server-owned keys.

    The denylist enforced by :mod:`nemo_retriever.service.policy` already
    rejects requests with these keys, but we apply a belt-and-suspenders
    overwrite here so a misconfigured policy can never cause a request
    to redirect endpoint URLs or replace API keys.
    """
    merged = dict(base)
    if override:
        merged.update(override)
    for k in owned:
        if k in base:
            merged[k] = base[k]
    return merged


def _resolve_extract_params(
    base_extract: dict[str, Any],
    extract_override: dict[str, Any] | None,
) -> Any:
    """Merge extraction settings while isolating Parse-only server fields."""
    from nemo_retriever.common.params import ExtractParams

    extract_kwargs = _merge_server_owned(
        base_extract,
        extract_override,
        _TRUST_OWNED_EXTRACT_KEYS,
    )
    if extract_kwargs.get("method", "pdfium") != "nemotron_parse":
        extract_kwargs.pop("nemotron_parse_invoke_url", None)
        extract_kwargs.pop("nemotron_parse_model", None)
    return ExtractParams(**extract_kwargs)


def inject_sidecar_attachment(
    spec: dict[str, Any] | None, *, payload: bytes, content_type: str, filename: str
) -> dict[str, Any] | None:
    """Replace an admitted sidecar ID with gateway-bound attachment bytes."""
    if spec is None:
        return None
    vdb = spec.get("vdb_upload_params")
    if not vdb or not vdb.get("meta_dataframe_id"):
        return spec
    resolved = dict(spec)
    vdb_copy = dict(vdb)
    vdb_copy.pop("meta_dataframe_id", None)
    vdb_copy["_meta_dataframe_bytes"] = payload
    vdb_copy["_meta_dataframe_content_type"] = content_type
    vdb_copy["_meta_dataframe_filename"] = filename
    resolved["vdb_upload_params"] = vdb_copy
    return resolved


def _resolve_sidecar_in_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve legacy standalone sidecars; split work is resolved at admission."""
    if spec is None:
        return None
    vdb = spec.get("vdb_upload_params")
    if not vdb or not vdb.get("meta_dataframe_id"):
        return spec
    from nemo_retriever.service.services.sidecar_store import get_sidecar_store

    store = get_sidecar_store()
    if store is None:
        raise RuntimeError("Sidecar metadata must be resolved by ingest admission before worker execution.")
    entry = store.consume(vdb["meta_dataframe_id"])
    if entry is None:
        raise RuntimeError(f"Sidecar id {vdb['meta_dataframe_id']!r} not found")
    return inject_sidecar_attachment(
        spec, payload=entry.payload, content_type=entry.content_type, filename=entry.filename
    )


def _resolve_service_extraction_mode(
    extraction_mode: str,
    filename: str,
) -> str:
    """Pick the worker extraction mode for a single uploaded file.

    When the client leaves ``extraction_mode`` at ``"auto"`` (the service
    default), infer ``"text"`` / ``"html"`` / … from the filename so HTML
    and TXT uploads use the typed splitters instead of falling through a
    mis-routed graph.
    """
    mode = (extraction_mode or "auto").strip().lower()
    if mode != "auto":
        return mode
    from nemo_retriever.service.utils.file_type import (
        infer_extraction_mode_from_filename,
    )

    inferred = infer_extraction_mode_from_filename(filename)
    return inferred or "auto"


def _request_needs_asr_params(extraction_mode: str | None, filename: str) -> bool:
    """True iff the request is audio/video and should carry ``_asr_params``.

    The worker holds a single ``ASRParams`` derived from
    ``serviceConfig.nimEndpoints.audioGrpcEndpoint``. Attaching that to
    every per-request ingestor is what caused the
    ``RuntimeError: MediaChunkActor requires media dependencies; missing:
    ffmpeg, ffprobe`` for PDF uploads — the audio-only graph branch then
    won the routing decision regardless of file type. We restrict the
    attachment to:

    * ``extraction_mode == "audio"`` or ``"video"`` — explicit caller
      intent; the user already opted into media routing.
    * ``extraction_mode == "auto"`` plus an audio/video file extension —
      ``MultiTypeExtractOperator`` dispatches at row level and only
      needs ASR when the row is actually media.

    Anything else (``"pdf"``, ``"image"``, ``"text"``, ``"html"``, or a
    non-media extension under ``"auto"``) must not pin ASR params.
    """
    mode = (extraction_mode or "").strip().lower()
    if mode in {"audio", "video"}:
        return True
    if mode != "auto":
        return False

    from nemo_retriever.service.utils.file_type import (
        FileClassifier,
        category_requires_media_deps,
    )

    dot = filename.rfind(".")
    suffix = filename[dot:].lower() if dot != -1 else ""
    entry = FileClassifier.SUFFIX_MAP.get(suffix)
    if entry is None:
        return False
    category, _ = entry
    return category_requires_media_deps(category)


def _materialize_sidecar_bytes(vdb_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert resolved sidecar bytes into a pandas DataFrame in place.

    Runs *inside* the child process. ``_meta_dataframe_bytes`` is the
    payload uploaded via ``POST /v1/ingest/sidecar``; the content-type
    (or filename suffix as fallback) picks the right pandas reader.
    """
    payload = vdb_kwargs.pop("_meta_dataframe_bytes", None)
    if payload is None:
        return vdb_kwargs
    content_type = vdb_kwargs.pop("_meta_dataframe_content_type", "") or ""
    filename = vdb_kwargs.pop("_meta_dataframe_filename", "") or ""

    from io import BytesIO

    import pandas as pd

    ct_lower = content_type.lower()
    fname_lower = filename.lower()
    if "parquet" in ct_lower or fname_lower.endswith(".parquet") or fname_lower.endswith(".pq"):
        df = pd.read_parquet(BytesIO(payload))
    elif "json" in ct_lower or fname_lower.endswith(".json") or fname_lower.endswith(".jsonl"):
        df = pd.read_json(BytesIO(payload), lines=fname_lower.endswith(".jsonl"))
    else:
        df = pd.read_csv(BytesIO(payload))
    vdb_kwargs["meta_dataframe"] = df
    return vdb_kwargs


def _build_graph_ingestor_from_spec(
    filename: str,
    payload: bytes,
    base_extract: dict[str, Any],
    base_embed: dict[str, Any] | None,
    spec: dict[str, Any] | None,
    base_caption: dict[str, Any] | None = None,
    base_asr: dict[str, Any] | None = None,
) -> "tuple[Any, str, bool]":
    """Construct a :class:`GraphIngestor` reflecting the per-request *spec*.

    Returns ``(ingestor, extraction_mode, has_per_request_vdb)``. The
    last value tells the caller to skip the legacy out-of-graph
    vectordb fan-out — the in-graph ``IngestVdbOperator`` already
    handles persistence when ``vdb_upload_params`` is present.
    """
    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
    from nemo_retriever.operators.graph_ops.multi_type_extract_operator import (
        DEFAULT_AUDIO_SPLIT_INTERVAL,
        DEFAULT_VIDEO_FRAME_FPS,
    )
    from nemo_retriever.common.params import (
        ASRParams,
        AudioChunkParams,
        AudioVisualFuseParams,
        CaptionParams,
        DedupParams,
        StoreParams,
        VdbUploadParams,
        VideoFrameParams,
        VideoFrameTextDedupParams,
        WebhookParams,
    )

    spec = spec or {}
    extraction_mode = _resolve_service_extraction_mode(spec.get("extraction_mode", "auto"), filename)
    split_config = spec.get("split_config")

    extract_params = _resolve_extract_params(base_extract, spec.get("extract_params"))

    embed_override = spec.get("embed_params")
    embed_params = _resolve_embed_params(base_embed, embed_override)

    # Caption baseline + per-request overrides. The base dict carries
    # the server-owned endpoint/API key/model name; the override carries
    # behavioural knobs (prompt, system_prompt, batch_size, …).
    caption_override = spec.get("caption_params")
    if base_caption is None and caption_override is None:
        caption_params = None
    elif base_caption is None and caption_override is not None:
        raise RuntimeError(
            "caption_params provided but no caption endpoint is configured on "
            "this worker. The policy layer should have rejected this earlier."
        )
    else:
        caption_kwargs = _merge_server_owned(base_caption or {}, caption_override, _TRUST_OWNED_CAPTION_KEYS)
        caption_params = CaptionParams(**caption_kwargs) if caption_kwargs.get("endpoint_url") else None

    asr_params = ASRParams(**base_asr) if base_asr else None

    ingestor = GraphIngestor(run_mode="inprocess", show_progress=False)
    ingestor = ingestor.buffers([(filename, BytesIO(payload))])

    if extraction_mode == "video":
        # Service auto-routing resolves supported video extensions before this
        # point. Preserve the canonical video branch defaults instead of
        # passing the MP4 bytes through the generic PDF extraction path. ASR
        # remains optional, but frame extraction, frame OCR, and frame-text
        # deduplication run for every video. Fusion is appended only when the
        # audio branch is enabled.
        ingestor = ingestor.extract(
            extract_params,
            split_config=split_config,
            extraction_mode=extraction_mode,
            audio_chunk_params=AudioChunkParams(
                enabled=asr_params is not None,
                split_type="size",
                split_interval=DEFAULT_AUDIO_SPLIT_INTERVAL,
            ),
            asr_params=asr_params,
            video_frame_params=VideoFrameParams(enabled=True, fps=DEFAULT_VIDEO_FRAME_FPS, dedup=True),
            video_text_dedup_params=VideoFrameTextDedupParams(
                enabled=True,
                max_dropped_frames=2,
            ),
            av_fuse_params=AudioVisualFuseParams(enabled=True),
        )
    elif extraction_mode == "image":
        ingestor = ingestor.extract_image_files(extract_params, split_config=split_config)
    elif extraction_mode == "html" and split_config is None:
        ingestor = ingestor.extract_html()
    else:
        ingestor = ingestor.extract(
            extract_params,
            split_config=split_config,
            extraction_mode=extraction_mode,
        )
        # Only attach the worker-wide ASR params to the per-request ingestor
        # when the request is genuinely audio/video. ``asr_params`` is
        # auto-derived from the cluster's ``audio_grpc_endpoint`` and would
        # otherwise taint every PDF / image / text / HTML upload with audio
        # state — which then mis-routes the request through the audio-only
        # graph in :func:`nemo_retriever.graph.ingestor_runtime.build_graph`
        # and crashes inside ``MediaChunkActor`` when ffmpeg/ffprobe are
        # absent. The graph builder also gates on extraction_mode now, so
        # this is defence in depth.
        if asr_params is not None and _request_needs_asr_params(extraction_mode, filename):
            ingestor._asr_params = asr_params

    stage_order = spec.get("stage_order") or []
    seen_post_extract: set[str] = set()

    def _apply_store_if_requested() -> None:
        nonlocal ingestor
        store_kwargs = spec.get("store_params")
        if store_kwargs is not None:
            ingestor = ingestor.store(StoreParams(**store_kwargs))

    def _apply_webhook_if_requested() -> None:
        nonlocal ingestor
        webhook_kwargs = spec.get("webhook_params")
        if webhook_kwargs is not None:
            ingestor = ingestor.webhook(WebhookParams(**webhook_kwargs))

    def _apply_caption_if_requested() -> None:
        nonlocal ingestor
        if caption_params is not None:
            ingestor = ingestor.caption(caption_params)

    for stage_name in stage_order:
        if stage_name in ("extract",) or stage_name in seen_post_extract:
            continue
        seen_post_extract.add(stage_name)
        if stage_name == "dedup":
            dedup_kwargs = spec.get("dedup_params") or {}
            ingestor = ingestor.dedup(DedupParams(**dedup_kwargs))
        elif stage_name == "embed":
            if embed_params is not None:
                ingestor = ingestor.embed(embed_params)
        elif stage_name == "filter":
            ingestor = ingestor.filter()
        elif stage_name == "store":
            _apply_store_if_requested()
        elif stage_name == "webhook":
            _apply_webhook_if_requested()
        elif stage_name == "caption":
            _apply_caption_if_requested()

    if embed_params is not None and "embed" not in seen_post_extract:
        ingestor = ingestor.embed(embed_params)

    # ``store`` / ``webhook`` / ``caption`` may be present in params
    # without an explicit stage_order entry (matches the GraphIngestor
    # pattern where the params model triggers the stage). Auto-append.
    if "caption" not in seen_post_extract:
        _apply_caption_if_requested()
    if "store" not in seen_post_extract:
        _apply_store_if_requested()
    if "webhook" not in seen_post_extract:
        _apply_webhook_if_requested()

    # vdb_upload is not a stage_order entry in GraphIngestor either — the
    # operator is always appended after embed/store from the params model.
    has_per_request_vdb = False
    vdb_kwargs = spec.get("vdb_upload_params")
    if vdb_kwargs is not None:
        # Sidecar metadata (Phase 6): the parent process placed the
        # uploaded bytes on the spec; turn them into a DataFrame here.
        vdb_kwargs = _materialize_sidecar_bytes(dict(vdb_kwargs))
        ingestor = ingestor.vdb_upload(VdbUploadParams(**vdb_kwargs))
        has_per_request_vdb = True

    return ingestor, extraction_mode, has_per_request_vdb


def _run_pipeline_in_process(
    filename: str,
    payload: bytes,
    extract_params_dict: dict[str, Any],
    embed_params_dict: dict[str, Any] | None,
    vectordb_url: str | None = None,
    pipeline_spec: dict[str, Any] | None = None,
    caption_params_dict: dict[str, Any] | None = None,
    asr_params_dict: dict[str, Any] | None = None,
    trace_context: dict[str, str] | None = None,
    pool_label: str | None = None,
    service_role: str | None = None,
    write_context: DocumentWriteContext | None = None,
    job_id: str | None = None,
    internal_api_token: str | None = None,
    vectordb_write_timeout_s: float = _DEFAULT_VECTORDB_WRITE_TIMEOUT_S,
) -> tuple[int, list[dict[str, Any]], float]:
    """Execute one pipeline run inside a child process.

    This is a **top-level module function** so it can be pickled by
    :class:`ProcessPoolExecutor`.  All heavy imports happen here so
    that the parent process stays lightweight.

    The pipeline shape comes from two layers:

    * ``extract_params_dict`` / ``embed_params_dict`` — server-owned
      defaults derived from :class:`ServiceConfig.nim_endpoints` at
      startup. Carry the endpoint URLs and API keys.
    * ``pipeline_spec`` — optional per-request override validated by
      :func:`nemo_retriever.service.policy.validate_pipeline_spec`.
      Carries "shape" knobs (chunk sizes, output flags, stage order, …).

    When ``pipeline_spec`` is ``None`` (or empty) the behaviour exactly
    matches the original closure-baked pipeline.
    """
    from nemo_retriever.service import tracing

    t0 = time.monotonic()

    tracing.configure_tracing(service_role=service_role or "worker-process")
    span_context = _extract_trace_context_for_pipeline(trace_context)
    span_attributes = {
        "pool": (pool_label or "").lower(),
        "document.filename": filename,
    }
    try:
        with tracing.start_span("pipeline.ingest", context=span_context, attributes=span_attributes):
            ingestor, _extraction_mode, has_per_request_vdb = _build_graph_ingestor_from_spec(
                filename,
                payload,
                extract_params_dict,
                embed_params_dict,
                pipeline_spec,
                caption_params_dict,
                asr_params_dict,
            )
            document_metadata = write_context.document_metadata if write_context is not None else None
            source_url = document_metadata.get("_nrl_source_url") if document_metadata else None
            if isinstance(source_url, str):
                ingestor._set_source_map({filename: source_url})
            effective_embed_params = getattr(ingestor, "_embed_params", None)
            embedding_column = getattr(effective_embed_params, "output_column", None)

            result_df = ingestor.ingest()
            _merge_document_metadata(
                result_df,
                document_metadata,
                source_identifier=filename,
            )
    finally:
        tracing.force_flush(timeout_millis=500)

    elapsed = time.monotonic() - t0

    row_count = len(result_df)

    from nemo_retriever.service.utils.file_type import is_text_like_filename

    if row_count == 0 and is_text_like_filename(filename):
        raise ValueError(
            f"Extraction produced no rows for {filename!r}. "
            "Supported HTML and TXT inputs must yield at least one text chunk. "
            "If you need custom chunking, pass split_config for the matching "
            "source type (see README: split_config for text/html)."
        )

    if vectordb_url and row_count > 0 and not has_per_request_vdb:
        # Skip the out-of-graph fan-out when the client already wired
        # IngestVdbOperator into the spec — that operator handles
        # persistence itself.
        from nemo_retriever.common.vdb.records import to_client_vdb_records
        from nemo_retriever.service.services.pipeline_pool import DocumentWriteContext

        records = to_client_vdb_records(result_df)
        _post_records_to_vectordb(
            records,
            vectordb_url,
            filename,
            context=write_context or DocumentWriteContext(),
            job_id=job_id,
            internal_api_token=internal_api_token,
            timeout_s=vectordb_write_timeout_s,
        )

    result_options = pipeline_spec or {}
    result_schema = result_options.get("result_schema", "legacy")
    result_data = _sanitize_result_data(
        result_df,
        result_schema=result_schema,
        return_embeddings=bool(result_options.get("return_embeddings", False)),
        embedding_column=embedding_column,
        return_images=bool(result_options.get("return_images", False)),
    )
    return row_count, result_data, elapsed


def _merge_document_metadata(
    result: Any,
    document_metadata: dict[str, Any] | None,
    *,
    source_identifier: str | None = None,
) -> None:
    """Merge request metadata into extracted rows without replacing parser fields."""
    if not document_metadata:
        return

    # Validate and copy at the child-process boundary so storage receives no
    # aliases or values that cannot be represented in JSON query hits.
    canonical = json.loads(json.dumps(document_metadata, ensure_ascii=False))
    source_url = canonical.pop("_nrl_source_url", None)
    source_map = (
        {source_identifier: source_url} if isinstance(source_identifier, str) and isinstance(source_url, str) else {}
    )

    def merge_row(row: Any) -> None:
        if not isinstance(row, dict):
            return
        metadata = row.get("metadata")
        if source_map:
            for field in ("path", "source_path", "source_id"):
                if field in row:
                    row[field] = restore_url_source_value(row[field], source_map)
        if not isinstance(metadata, dict):
            metadata = {}
        if source_map:
            metadata = restore_url_source_value(metadata, source_map)
        row["metadata"] = metadata
        content_metadata = metadata.get("content_metadata")
        if not isinstance(content_metadata, dict):
            content_metadata = {}
            metadata["content_metadata"] = content_metadata
        for key, value in canonical.items():
            content_metadata.setdefault(key, copy.deepcopy(value))

    if isinstance(result, list):
        for row in result:
            merge_row(row)
        return
    if hasattr(result, "iterrows"):
        for index, row in result.iterrows():
            fields = ("path", "source_path", "source_id", "metadata")
            holder = {field: row.get(field) for field in fields if field in result.columns}
            merge_row(holder)
            for field, value in holder.items():
                if field in result.columns:
                    result.at[index, field] = value


def _local_model_runtime_kwargs(local: "LocalModelsConfig") -> dict[str, Any]:
    """Shared ``ModelRuntimeParams`` fields for in-pod HF stages."""
    runtime: dict[str, Any] = {}
    if local.device:
        runtime["device"] = local.device
    if local.hf_cache_dir:
        runtime["hf_cache_dir"] = local.hf_cache_dir
    return runtime


def _embed_params_enabled(embed_kwargs: dict[str, Any]) -> bool:
    """Return True when *embed_kwargs* describe a remote or local embed stage."""
    if not embed_kwargs:
        return False
    if embed_kwargs.get("embed_invoke_url") or embed_kwargs.get("embedding_endpoint"):
        return True
    return bool(embed_kwargs.get("model_name") or embed_kwargs.get("embed_model_name"))


def _resolve_embed_params(
    base_embed: dict[str, Any] | None,
    embed_override: dict[str, Any] | None,
) -> Any | None:
    """Merge server-owned and per-request embed kwargs into :class:`EmbedParams`."""
    if base_embed is None and embed_override is None:
        return None
    embed_base = base_embed or {}
    embed_kwargs = _merge_server_owned(embed_base, embed_override, _TRUST_OWNED_EMBED_KEYS)
    if not _embed_params_enabled(embed_kwargs):
        return None
    from nemo_retriever.common.params import EmbedParams

    return EmbedParams(**embed_kwargs)


def build_extract_params(nim: "NimEndpointsConfig", local: "LocalModelsConfig | None" = None) -> Any:
    """Derive :class:`ExtractParams` from service NIM and local-model config.

    The ``ExtractParams`` model validator auto-enables table structure when
    its invoke URL is provided. When ``local_models.enabled`` is true, the
    same flag is set when the table-structure NIM URL is absent.
    """
    from nemo_retriever.common.params import ExtractParams

    local = local or _default_local_models_config()
    kwargs: dict[str, Any] = {}
    if nim.page_elements_invoke_url:
        kwargs["page_elements_invoke_url"] = nim.page_elements_invoke_url
    if nim.ocr_invoke_url:
        kwargs["ocr_invoke_url"] = nim.ocr_invoke_url
    if nim.table_structure_invoke_url:
        kwargs["table_structure_invoke_url"] = nim.table_structure_invoke_url
    if nim.nemotron_parse_invoke_url:
        # ExtractParams validates that Parse-specific configuration and the
        # extraction method are selected together.
        kwargs["method"] = "nemotron_parse"
        kwargs["nemotron_parse_invoke_url"] = nim.nemotron_parse_invoke_url
        if nim.nemotron_parse_model:
            kwargs["nemotron_parse_model"] = nim.nemotron_parse_model
    if nim.api_key:
        kwargs["api_key"] = nim.api_key

    if local.enabled and local.extract.enabled:
        if local.extract.use_table_structure and not nim.table_structure_invoke_url:
            kwargs["use_table_structure"] = True
        if not nim.ocr_invoke_url:
            kwargs["ocr_version"] = local.extract.ocr_version
            if local.extract.ocr_lang is not None:
                kwargs["ocr_lang"] = local.extract.ocr_lang

    return ExtractParams(**kwargs)


def _default_local_models_config() -> "LocalModelsConfig":
    from nemo_retriever.service.config import LocalModelsConfig

    return LocalModelsConfig()


def build_caption_params(nim: "NimEndpointsConfig") -> Any | None:
    """Derive :class:`CaptionParams` from service NIM endpoint config.

    Returns ``None`` when no caption endpoint is configured — clients
    that request the ``caption`` stage will hit the policy's
    ``caption_enabled`` guard before reaching this point.
    """
    from nemo_retriever.common.params import CaptionParams

    if not nim.caption_invoke_url:
        return None

    kwargs: dict[str, Any] = {"endpoint_url": nim.caption_invoke_url}
    if nim.caption_model_name:
        kwargs["model_name"] = nim.caption_model_name
    if nim.api_key:
        kwargs["api_key"] = nim.api_key
    return CaptionParams(**kwargs)


def build_asr_params(nim: "NimEndpointsConfig", local: "LocalModelsConfig | None" = None) -> Any | None:
    """Derive :class:`ASRParams` from service NIM and local-model config.

    Returns ``None`` when neither a Parakeet NIM gRPC endpoint nor local
    in-pod ASR is configured.
    """
    local = local or _default_local_models_config()
    if nim.audio_grpc_endpoint:
        from nemo_retriever.common.params import ASRParams

        function_id = (os.environ.get("AUDIO_FUNCTION_ID") or "").strip() or None
        auth_token = nim.api_key or (os.environ.get("NVIDIA_API_KEY") or "").strip() or None
        return ASRParams(
            audio_endpoints=(nim.audio_grpc_endpoint, None),
            audio_infer_protocol="grpc",
            auth_token=auth_token,
            function_id=function_id,
        )
    if local.enabled and local.asr.enabled:
        from nemo_retriever.common.params import ASRParams

        return ASRParams()
    return None


def build_embed_params(nim: "NimEndpointsConfig", local: "LocalModelsConfig | None" = None) -> Any | None:
    """Derive :class:`EmbedParams` from service NIM and local-model config.

    Remote ``embed_invoke_url`` wins when set. Otherwise, when
    ``local_models.enabled`` and ``local_models.embed.enabled``, returns
    in-pod embed params (no HTTP endpoint).
    """
    local = local or _default_local_models_config()
    from nemo_retriever.common.params import EmbedParams, ModelRuntimeParams

    if nim.embed_invoke_url:
        kwargs: dict[str, Any] = {"embed_invoke_url": nim.embed_invoke_url}
        if nim.embed_model_name:
            kwargs["model_name"] = nim.embed_model_name
            kwargs["embed_model_name"] = nim.embed_model_name
        if nim.embed_model_provider_prefix:
            kwargs["embed_model_provider_prefix"] = nim.embed_model_provider_prefix
        if nim.api_key:
            kwargs["api_key"] = nim.api_key
        return EmbedParams(**kwargs)

    if local.enabled and local.embed.enabled:
        kwargs = {
            "model_name": local.embed.model_name,
            "embed_model_name": local.embed.model_name,
            "local_ingest_embed_backend": local.embed.local_ingest_embed_backend,
        }
        runtime_kwargs = _local_model_runtime_kwargs(local)
        if local.embed.gpu_memory_utilization != 0.45:
            runtime_kwargs["gpu_memory_utilization"] = local.embed.gpu_memory_utilization
        if runtime_kwargs:
            kwargs["runtime"] = ModelRuntimeParams(**runtime_kwargs)
        return EmbedParams(**kwargs)

    return None


def _make_work_fn(
    config: ServiceConfig,
    *,
    label: str,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Factory that captures pipeline params once and returns an async worker.

    Each invocation creates a :class:`ProcessPoolExecutor` so that every
    pipeline run is isolated in its own child process — this eliminates
    PDFium thread-safety issues (the C library has global mutable state
    that corrupts under concurrent thread access).
    """
    extract_params = build_extract_params(config.nim_endpoints, config.local_models)
    embed_params = build_embed_params(config.nim_endpoints, config.local_models)
    caption_params = build_caption_params(config.nim_endpoints)
    asr_params = build_asr_params(config.nim_endpoints, config.local_models)

    vectordb_url: str | None = None
    if config.vectordb.enabled:
        vectordb_url = config.vectordb.vectordb_url
        logger.info("VectorDB write enabled for %s workers → %s", label, vectordb_url)

    num_workers = config.pipeline.realtime_workers if label.lower() == "realtime" else config.pipeline.batch_workers
    max_tasks_per_child = _resolve_max_tasks_per_child(config.local_models)
    warmup_spec_json = _build_pool_warmup_spec_json(
        config.local_models,
        extract_params.model_dump(mode="json"),
        embed_params.model_dump(mode="json") if embed_params else None,
        asr_params.model_dump(mode="json") if asr_params else None,
    )

    if config.local_models.enabled and config.local_models.warmup_on_startup:
        _service_warmup_state["enabled"] = True

    executor = _create_process_executor(
        num_workers=num_workers,
        max_tasks_per_child=max_tasks_per_child,
        warmup_spec_json=warmup_spec_json,
    )
    _process_executors.append(executor)
    if warmup_spec_json:
        _executor_warmup_targets.append((executor, num_workers))

    extract_params_dict = extract_params.model_dump(mode="json")
    embed_params_dict = embed_params.model_dump(mode="json") if embed_params else None
    caption_params_dict = caption_params.model_dump(mode="json") if caption_params else None
    asr_params_dict = asr_params.model_dump(mode="json") if asr_params else None

    _pipeline_configs[label.lower()] = {
        "label": label,
        "run_mode": "inprocess",
        "execution": "process-isolated",
        "show_progress": False,
        "extract_params": _params_to_dict(extract_params),
        "embed_params": _params_to_dict(embed_params) if embed_params else None,
        "embed_enabled": embed_params is not None,
        "caption_params": (_redact_dict(_params_to_dict(caption_params)) if caption_params else None),
        "caption_enabled": caption_params is not None,
        "asr_params": _redact_dict(_params_to_dict(asr_params)) if asr_params else None,
        "asr_enabled": asr_params is not None,
        "pool": {
            "workers": num_workers,
            "queue_size": (
                config.pipeline.realtime_queue_size if label.lower() == "realtime" else config.pipeline.batch_queue_size
            ),
            "max_tasks_per_child": max_tasks_per_child,
        },
        "nim_endpoints": _redact_dict(config.nim_endpoints.model_dump(mode="json")),
        "local_models": _redact_dict(config.local_models.model_dump(mode="json")),
    }

    logger.info(
        "Pipeline work function created (%s): extract=%s, embed=%s, local_models=%s, "
        "process_pool_workers=%d, max_tasks_per_child=%d, warmup_on_startup=%s",
        label,
        type(extract_params).__name__,
        type(embed_params).__name__ if embed_params else "disabled",
        config.local_models.enabled,
        num_workers,
        max_tasks_per_child,
        config.local_models.warmup_on_startup,
    )

    # Mutable holder so the BrokenProcessPool handler can replace the
    # executor while the closure keeps a stable reference.
    executor_ref: list[ProcessPoolExecutor] = [executor]

    async def _work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        filename = item.filename or item.id
        loop = asyncio.get_running_loop()

        resolved_spec = _resolve_sidecar_in_spec(item.pipeline_spec)
        write_context = item.write.resolved(fallback_document_id=item.id)

        try:
            trace_context = _capture_trace_context_for_pipeline()
            row_count, result_data, elapsed = await loop.run_in_executor(
                executor_ref[0],
                _run_pipeline_in_process,
                filename,
                item.payload,
                extract_params_dict,
                embed_params_dict,
                vectordb_url,
                resolved_spec,
                caption_params_dict,
                asr_params_dict,
                trace_context,
                label,
                config.mode,
                write_context,
                item.job_id,
                config.vectordb.internal_api_token,
                config.vectordb.write_timeout_s,
            )
        except BrokenProcessPool:
            logger.error(
                "%s process pool broken (worker crash) while processing " "id=%s file=%s — recreating pool",
                label,
                item.id,
                filename,
            )
            old = executor_ref[0]
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            if old in _process_executors:
                _process_executors.remove(old)
            new_executor = _create_process_executor(
                num_workers=num_workers,
                max_tasks_per_child=max_tasks_per_child,
                warmup_spec_json=warmup_spec_json,
            )
            executor_ref[0] = new_executor
            _process_executors.append(new_executor)
            if warmup_spec_json:
                _executor_warmup_targets.clear()
                _executor_warmup_targets.append((new_executor, num_workers))
                try:
                    warmup_process_pool_workers()
                except Exception:
                    logger.warning("%s pool recreated but model warmup failed", label)
            raise

        logger.info(
            "%s pipeline completed: id=%s file=%s rows=%d elapsed=%.2fs",
            label,
            item.id,
            filename,
            row_count,
            elapsed,
        )
        return row_count, result_data

    return _work


def create_realtime_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Build the async work function for the **realtime** pool.

    Processes single pages — the extract operator finds one page and the
    pipeline runs with minimal latency.
    """
    return _make_work_fn(config, label="Realtime")


def create_batch_work_fn(
    config: ServiceConfig,
) -> Callable[[WorkItem], Awaitable[tuple[int, list[dict[str, Any]]]]]:
    """Build the async work function for the **batch** pool.

    Processes full documents — the extract operator splits internally
    into N pages and processes them in one pass for better throughput.
    """
    return _make_work_fn(config, label="Batch")
