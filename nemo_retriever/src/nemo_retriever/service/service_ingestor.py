# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestor that submits work to a running ``retriever service`` HTTP server.

This is the third ``run_mode`` exposed via :func:`nemo_retriever.ingestor.create_ingestor`.
Where ``inprocess`` and ``batch`` execute the operator graph in the caller's
process / Ray cluster, ``service`` mode delegates execution to a separate
FastAPI server (see :mod:`nemo_retriever.service`) that runs its own pool of
worker processes and remote NIM endpoints.

Three execution surfaces are exposed:

1. :meth:`ServiceIngestor.ingest` — sync, blocks until every document has
   finished, returns a :class:`ServiceIngestResult` (a ``list`` subclass
   holding per-document completion events, plus ``job_id`` / ``failures``
   / ``document_ids`` / ``document_filenames`` / ``elapsed_s`` /
   ``job_status`` / ``trace_id`` / ``dataframe``
   attributes). By default ``return_results=True`` fetches each
   completed document's rows from the status endpoint into
   ``result.dataframe``. Each
   call implicitly opens one server-side job aggregate sized to
   ``len(documents)`` via ``POST /v1/ingest/job``, then submits every
   document under that ``job_id``.

2. :meth:`ServiceIngestor.ingest_stream` — sync generator yielding one
   ``dict`` per event (``job_created``, ``upload_complete``,
   ``document_complete``, ``upload_failed``, plus the job-lifecycle
   stream: ``job_started``, ``job_progress``, ``job_finalized`` /
   ``job_partial`` / ``job_failed``).

3. :meth:`ServiceIngestor.aingest_stream` — true async generator for
   callers already inside an event loop.

Pipeline configuration in service run_mode goes through a
:class:`~nemo_retriever.service.models.pipeline_spec.PipelineSpec`
that travels alongside each upload. The server validates the spec
against :class:`~nemo_retriever.service.policy.PipelineOverridesPolicy`
(an audited allow-list keyed on parameter name + sink URL) before any
worker sees it — the trust boundary lives on the server, never on the
client.

Fluent methods that *do* take effect by writing to the spec:

* ``.extract(...)`` — per-request extraction knobs (DPI, OCR enable, …)
* ``.texts(...)`` — inline text payloads
* ``.embed(...)`` — embedding model/dim overrides bounded by the
  operator's allow-list
* ``.dedup(...)``, ``.split(...)``, ``.filter()`` — shape knobs
* ``.store(StorageUri="s3://...")`` — remote object storage sink
* ``.webhook(endpoint_url="https://hooks.example.com/...")`` — HTTP sink
* ``.vdb_upload(...)`` — vector-DB sink, including sidecar metadata
  via the dedicated ``POST /v1/ingest/sidecar`` upload endpoint
* ``.caption(...)`` — remote VLM captioning when the operator has wired
  ``nim_endpoints.caption_invoke_url``; trust-sensitive fields like
  endpoint_url / api_key / model_name stay server-owned
* ``.save_to_disk(output_directory="...")`` — client-side persistence:
  fetches ``result_data`` from ``/v1/ingest/status/{id}`` for each
  completed document and writes JSON or gzipped JSON locally

Methods that intentionally remain unsupported in service run_mode:

* ``.udf(...)`` — named UDFs deferred to a follow-up phase;
  arbitrary-code execution is the canonical trust-boundary violation
  and needs a server-side registry first
* ``.store_embed()``, ``.save_intermediate_results()`` — by design,
  in-process-only debugging helpers; use ``run_mode='inprocess'`` for
  stage-by-stage introspection
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
import warnings
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List, Optional, Self, Sequence, Tuple, Union

import httpx

from nemo_retriever.ingestor.results import ResultSchema, concat_ingest_results
from nemo_retriever.ingestor import _merge_params, ingestor
from nemo_retriever.common.inline_text import inline_text_source_id, is_blank_inline_corpus, normalize_inline_texts
from nemo_retriever.common.params import (
    CaptionParams,
    IngestExecuteParams,
    PdfSplitParams,
    StoreParams,
    UrlFetchParams,
    VdbUploadParams,
    WebhookParams,
)
from nemo_retriever.common.url_fetch import (
    FetchedUrl,
    UrlFetchFailure,
    cleanup_fetched_urls,
    fetch_urls,
    normalize_urls,
    restore_url_source_value,
)
from nemo_retriever.service.client import FileUpload, InMemoryUpload, RetrieverServiceClient, UploadInput

logger = logging.getLogger(__name__)

_LEGACY_RESULT_SCHEMA_DEPRECATION = (
    "ServiceIngestor legacy result rows are deprecated and will switch to the compact "
    "schema in a future release. Pass result_schema='compact' to opt in now, or "
    "result_schema='legacy' to keep the current GraphIngestor.ingest() column layout "
    "with bulky image/embedding values stripped during the deprecation window."
)

_RESULT_FETCH_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


class ServiceIngestResult(list):
    """Materialized result returned by :meth:`ServiceIngestor.ingest`.

    Subclasses ``list`` so it satisfies the existing
    ``ingestor.ingest()`` return-type annotation (``List[Any]``); callers
    can iterate it just like a normal list.  Each entry is a per-document
    completion event dict.

    Attributes
    ----------
    job_id
        The server-assigned job aggregate id for this ``ingest()`` call.
        Every :meth:`ServiceIngestor.ingest` invocation opens exactly one
        job, sized to ``len(documents)``; this is the handle to drive
        ``GET /v1/ingest/job/{job_id}`` follow-ups.
    failures
        ``(document_id_or_filename, error_message)`` pairs for documents
        that failed during upload or pipeline processing.
    document_ids
        Document identifiers returned by the server, in upload order.
    document_filenames
        Mapping from server document id to the source filename submitted for
        that document.
    elapsed_s
        Wall-clock seconds from first upload to last result.
    job_status
        Final aggregate status reported by the server
        (``completed`` / ``failed`` / ``partial_success``) when a job
        lifecycle event was observed during the run. ``None`` if the
        stream closed without a terminal job event (e.g. SSE fallback
        only delivered per-document completions).
    trace_id
        Trace id returned by the server on the ``job_created`` event.
        ``None`` when tracing is disabled or the service does not include
        a trace id in the job creation event.
    dataframe
        When :meth:`ServiceIngestor.ingest` is called with
        ``return_results=True`` (the default), a ``pandas.DataFrame``
        of all successfully ingested rows fetched from the service via
        ``GET /v1/ingest/status/{document_id}``, concatenated in upload
        order. The current default ``result_schema="legacy"`` preserves
        the same column layout as ``GraphIngestor.ingest()`` in
        ``inprocess`` / ``batch`` run modes. It preserves full text and strips
        bulky raw image and embedding values from cells before transport. Pass
        ``result_schema="compact"`` to opt into the future compact schema.
        ``None`` when ``return_results=False``.
    """

    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        super().__init__(items or [])
        self.job_id: str | None = None
        self.failures: list[tuple[str, str]] = []
        self.document_ids: list[str] = []
        self.document_filenames: dict[str, str] = {}
        self.elapsed_s: float = 0.0
        self.job_status: str | None = None
        self.trace_id: str | None = None
        self.dataframe: Any = None

    def __repr__(self) -> str:
        return (
            f"ServiceIngestResult(job_id={self.job_id!r}, "
            f"documents={len(self)}, "
            f"failures={len(self.failures)}, "
            f"elapsed_s={self.elapsed_s:.2f})"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

_FLUENT_NOT_SUPPORTED_TEMPLATE = (
    "ServiceIngestor.{method}() is not yet supported in run_mode='service'. "
    "{phase_hint}"
    "See retriever-service.yaml or `retriever service describe` for the "
    "current per-request override policy."
)


def _raise_unsupported(method: str, *, phase_hint: str = "") -> None:
    raise NotImplementedError(
        _FLUENT_NOT_SUPPORTED_TEMPLATE.format(method=method, phase_hint=phase_hint + " " if phase_hint else "")
    )


def _normalize_files(files: Union[str, List[str], List[Path]]) -> list[Path]:
    if isinstance(files, (str, Path)):
        return [Path(files)]
    return [Path(f) for f in files]


def _empty_service_result_dataframe(result_schema: ResultSchema) -> Any:
    """Return an empty DataFrame matching the selected public result schema."""
    if result_schema == "legacy":
        from nemo_retriever.common.modality.txt.split import empty_text_chunks_df

        return empty_text_chunks_df()

    import pandas as pd

    return pd.DataFrame(columns=["text", "source_id", "element_type", "page_number"]).astype({"page_number": "int64"})


# ----------------------------------------------------------------------
# Client-side mirror of service.models.pipeline_spec.PipelineSpec
# ----------------------------------------------------------------------

# Keys this client treats as server-owned. Stripped from any params dict
# before it goes on the wire so users get a clear error if they try.
_SERVER_OWNED_KEYS: frozenset[str] = frozenset(
    {
        "invoke_url",
        "api_key",
        "page_elements_invoke_url",
        "page_elements_api_key",
        "ocr_invoke_url",
        "ocr_api_key",
        "table_structure_invoke_url",
        "nemotron_parse_invoke_url",
        "nemotron_parse_model",
        "embed_invoke_url",
        "embedding_endpoint",
        "embed_model_provider_prefix",
        "endpoint_url",
        "api_base",
        "auth_token",
        "lancedb_uri",
        "storage_uri",
    }
)


def _filter_policy_allowed(params_dict: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    """Keep only keys the default service policy allowlist admits per stage."""
    return {key: value for key, value in params_dict.items() if key in allowed}


def _wire_client_stage_params(
    spec: dict[str, Any],
    spec_key: str,
    merged: Any,
    *,
    method: str,
    allowed: frozenset[str],
) -> None:
    """Serialize client overrides for one pipeline stage onto ``spec``."""
    params_dict = _filter_policy_allowed(
        _strip_server_owned(_params_to_dict(merged), method),
        allowed,
    )
    _set_stage_params(spec, spec_key, params_dict)


def _strip_server_owned(params_dict: dict[str, Any], method: str) -> dict[str, Any]:
    """Raise if the caller set a server-owned key; otherwise return as-is.

    We fail fast on the client so users see a useful message instead of
    a generic 403 from the server.
    """
    rejected = [k for k in params_dict if k in _SERVER_OWNED_KEYS]
    if rejected:
        raise ValueError(
            f"ServiceIngestor.{method}(): keys {rejected!r} are server-owned in "
            "run_mode='service'. Endpoint URLs and API keys are configured by "
            "the retriever service via retriever-service.yaml; they cannot be "
            "set per-request."
        )
    return params_dict


def _require_remote_uri(uri: str, method: str, field: str) -> None:
    """Reject local paths early on the client side.

    The worker pod cannot see the caller's filesystem. A friendly error
    here saves the user a round-trip to receive HTTP 403 from the sink
    allowlist with the same message.
    """
    if "://" not in uri or uri.startswith("file://"):
        raise ValueError(
            f"ServiceIngestor.{method}(): {field}={uri!r} is a local path. "
            "In service run_mode the worker writes from inside the cluster; "
            "use a remote URI such as 's3://bucket/prefix/' instead."
        )


def _params_to_dict(value: Any) -> dict[str, Any]:
    """Normalise a fluent-method argument (model | dict | None) to a dict.

    Serialises only fields the caller explicitly set on a Pydantic params
    model (``exclude_unset=True``) so service-mode overrides do not include
    model defaults or validator-populated server fields (API keys, timeouts,
    ``batch_tuning``, etc.) that the worker policy allowlist would reject.

    Drops ``None`` values so the server's defaults can fill them in.
    """
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        d = value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    elif isinstance(value, dict):
        d = {k: v for k, v in value.items() if v is not None}
    else:
        raise TypeError(f"Cannot serialise {type(value).__name__!r} to a params dict")
    return d


def _set_stage_params(spec: dict[str, Any], key: str, params_dict: dict[str, Any]) -> None:
    """Attach a stage-params block only when the client supplied overrides."""
    if params_dict:
        spec[key] = params_dict


# ----------------------------------------------------------------------
# Async-to-sync queue bridge
# ----------------------------------------------------------------------


_SENTINEL = object()


class _AsyncToSyncBridge:
    """Run an async generator on a background thread and surface it as a sync iterator.

    The generator's items are funneled through a :class:`queue.Queue`; the
    sync side calls ``.get()`` blocking until the next item is ready.  The
    bridge owns its own asyncio event loop so the caller does not need
    one.
    """

    def __init__(self, agen_factory) -> None:
        self._agen_factory = agen_factory
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=64)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exc: BaseException | None = None
        self._stop_event = threading.Event()

    def __iter__(self) -> Iterator[Any]:
        self._thread = threading.Thread(target=self._run, name="ServiceIngestorBridge", daemon=True)
        self._thread.start()
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    if self._exc is not None:
                        raise self._exc
                    return
                yield item
        finally:
            self._stop_event.set()
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._drain())
        except BaseException as exc:  # noqa: BLE001 — we re-raise on the consumer side
            self._exc = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._queue.put(_SENTINEL)

    async def _drain(self) -> None:
        agen = self._agen_factory()
        try:
            async for item in agen:
                while True:
                    if self._stop_event.is_set():
                        return
                    try:
                        self._queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            try:
                await agen.aclose()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass


# ----------------------------------------------------------------------
# ServiceIngestor
# ----------------------------------------------------------------------


class ServiceIngestor(ingestor):
    """Ingestor that submits work to a running ``retriever service``.

    Parameters
    ----------
    base_url
        Base URL of the retriever service (default ``http://localhost:7670``).
    documents
        Initial list of file paths to ingest; may also be set/extended via
        :meth:`files` and :meth:`buffers`.
    max_concurrency
        Maximum concurrent document uploads (default 8).
    request_timeout_s
        Per-request HTTP timeout (default 600s for large documents).
    api_token
        Optional bearer token for service authentication.
    """

    RUN_MODE = "service"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:7670",
        documents: Optional[List[str]] = None,
        max_concurrency: int = 8,
        request_timeout_s: float = 600.0,
        api_token: str | None = None,
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._request_timeout_s = request_timeout_s
        self._url_failures: list[UrlFetchFailure] = []
        self._fetched_urls: list[FetchedUrl] = []
        self._url_source_map: dict[str, str] = {}
        self._api_token = (api_token or "").strip() or None
        self._inline_texts: list[str] | None = None
        self._document_ids: list[str] = []
        self._last_run_elapsed_s: float = 0.0
        self._last_job_id: str | None = None
        self._pipeline_spec: dict[str, Any] = {
            "extraction_mode": "auto",
            "stage_order": [],
        }
        # save_to_disk state (populated by .save_to_disk(...); None when disabled)
        self._save_to_disk_dir: Path | None = None
        self._save_to_disk_compression: str | None = None
        self._save_to_disk_cleanup: bool = True

    # ------------------------------------------------------------------
    # Pipeline-spec helpers
    # ------------------------------------------------------------------

    def _record_stage(self, name: str) -> None:
        order = self._pipeline_spec["stage_order"]
        if name not in order:
            order.append(name)

    def _new_result_fetch_client(self) -> httpx.Client:
        """Create a client for retained-result status requests."""
        return httpx.Client(timeout=self._request_timeout_s, headers=self._auth_headers)

    def _fetch_document_result_data(
        self,
        document_id: str,
        *,
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch ``result_data`` for *document_id* from the status endpoint.

        The status endpoint retains ``result_data`` through the job retention
        window, so retrying this read is safe. When the caller supplies a
        client it is reused for the first attempt. A transient failure receives
        one retry through a fresh client so a stale pooled connection cannot be
        selected again.
        """
        if not document_id:
            raise ValueError("_fetch_document_result_data(): empty document_id")

        if client is None:
            with self._new_result_fetch_client() as scoped_client:
                return self._fetch_document_result_data(document_id, client=scoped_client)

        url = f"{self._base_url}/v1/ingest/status/{document_id}"
        try:
            resp = client.get(url)
        except _RESULT_FETCH_TRANSIENT_ERRORS as exc:
            logger.debug(
                "Transient %s fetching retained result for %s; retrying on a fresh connection",
                type(exc).__name__,
                document_id,
            )
            with self._new_result_fetch_client() as retry_client:
                resp = retry_client.get(url)
        resp.raise_for_status()
        body = resp.json()
        return list(body.get("result_data") or [])

    def _write_result_data_to_disk(self, document_id: str, result_data: list[dict[str, Any]]) -> Path:
        """Write *result_data* for *document_id* to the configured output directory."""
        import gzip
        import json as _json

        if self._save_to_disk_dir is None:
            raise RuntimeError("_write_result_data_to_disk(): save_to_disk was never enabled")

        suffix = ".json.gz" if self._save_to_disk_compression == "gzip" else ".json"
        out_path = self._save_to_disk_dir / f"{document_id}{suffix}"
        payload = _json.dumps(
            {"document_id": document_id, "rows": result_data},
            ensure_ascii=False,
        ).encode("utf-8")
        if self._save_to_disk_compression == "gzip":
            with gzip.open(out_path, "wb") as fh:
                fh.write(payload)
        else:
            out_path.write_bytes(payload)
        return out_path

    def _save_document_to_disk(
        self,
        document_id: str,
        *,
        client: httpx.Client | None = None,
    ) -> Path:
        """Fetch ``result_data`` for *document_id* and write a JSON artifact.

        Returns the path that was written. Raises if the document_id is
        missing or the fetch fails.
        """
        if self._save_to_disk_dir is None:
            raise RuntimeError("_save_document_to_disk(): save_to_disk was never enabled")
        result_data = self._fetch_document_result_data(document_id, client=client)
        return self._write_result_data_to_disk(document_id, result_data)

    def _materialize_completed_document(
        self,
        document_id: str,
        *,
        return_results: bool,
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch (once) and optionally persist rows for a completed document."""
        if not return_results and self._save_to_disk_dir is None:
            return None
        result_data = self._fetch_document_result_data(document_id, client=client)
        if self._save_to_disk_dir is not None:
            self._write_result_data_to_disk(document_id, result_data)
        return result_data if return_results else None

    def _pipeline_payload(
        self,
        *,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> dict[str, Any] | None:
        """Return the spec dict to send on the wire, or ``None`` when empty.

        The "empty" check mirrors :meth:`PipelineSpec.is_empty` server-side
        so the worker can short-circuit identically.
        """
        spec = dict(self._pipeline_spec)
        if self._has_mixed_inline_sources():
            spec["extraction_mode"] = "auto"
        spec["result_schema"] = result_schema
        spec["return_embeddings"] = bool(return_embeddings or spec.get("return_embeddings", False))
        spec["return_images"] = bool(return_images or spec.get("return_images", False))
        is_empty = (
            spec.get("extraction_mode", "auto") in ("pdf", "auto")
            and not spec.get("stage_order")
            and not any(
                spec.get(k)
                for k in (
                    "extract_params",
                    "embed_params",
                    "dedup_params",
                    "caption_params",
                    "store_params",
                    "vdb_upload_params",
                    "webhook_params",
                    "split_config",
                    "pdf_split",
                )
            )
            and spec.get("result_schema", "legacy") == "legacy"
            and not spec.get("return_embeddings", False)
            and not spec.get("return_images", False)
        )
        return None if is_empty else spec

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}

    # ------------------------------------------------------------------
    # Input configuration (these ARE meaningful client-side)
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "ServiceIngestor":
        """Add document paths/URIs for processing."""
        if isinstance(documents, str):
            self._documents.append(documents)
        else:
            self._documents.extend(documents)
        return self

    def urls(
        self,
        urls: Union[str, Sequence[str]],
        params: UrlFetchParams | None = None,
        **kwargs: Any,
    ) -> "ServiceIngestor":
        """Add HTTP(S) sources for client-side fetch and service upload.

        Parameters
        ----------
        urls
            One absolute HTTP(S) URL or a sequence of URLs to append.
        params
            Shared fetch settings. Existing settings are retained when this
            append call supplies neither params nor keyword overrides.
        **kwargs
            Field overrides for UrlFetchParams.

        Returns
        -------
        ServiceIngestor
            This ingestor for fluent chaining.

        Raises
        ------
        ValueError
            If a URL is empty, relative, or uses a non-HTTP(S) scheme.
        """
        self._urls.extend(normalize_urls(urls))
        if params is not None or kwargs:
            merged = _merge_params(params, kwargs)
            self._url_fetch_params = merged if isinstance(merged, UrlFetchParams) else UrlFetchParams(**merged)
        return self

    def texts(self, texts: Union[str, Sequence[str]]) -> Self:
        """Set raw inline text documents, optionally alongside file or buffer uploads."""
        self._inline_texts = normalize_inline_texts(texts)
        return self

    def buffers(
        self,
        buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]],
    ) -> "ServiceIngestor":
        """Add in-memory buffers for processing.

        Each buffer must be ``(filename, BytesIO)`` so the server can record
        a meaningful source filename.
        """
        if isinstance(buffers, tuple):
            buffers = [buffers]
        for name, buf in buffers:
            self._buffers.append((name, buf))
        return self

    def load(self) -> "ServiceIngestor":
        """No-op for service mode."""
        return self

    # ------------------------------------------------------------------
    # Phase 1: pipeline-shape stages — sent via PipelineSpec
    # ------------------------------------------------------------------

    def all_tasks(self) -> "ServiceIngestor":
        """Record the default chain: extract → dedup → embed.

        Concrete params come from server config; ``all_tasks()`` only
        controls *stage order* and is the closest in-process equivalent
        of "run everything the server is configured to do".
        """
        self._record_stage("extract")
        self._record_stage("dedup")
        self._record_stage("embed")
        return self

    def dedup(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record a dedup stage with optional :class:`DedupParams` overrides.

        Setting both mechanisms to ``False`` transmits an explicit opt-out
        that suppresses caption-triggered automatic deduplication.

        Parameters
        ----------
        params
            Optional :class:`DedupParams` instance or parameter mapping to
            transmit to the service.
        **kwargs
            Field values used directly when ``params`` is omitted or applied
            as overrides when ``params`` is a parameter model.

        Returns
        -------
        ServiceIngestor
            This ingestor instance for fluent chaining.

        Raises
        ------
        TypeError
            If ``params`` cannot be serialized as a parameter mapping.
        ValueError
            If the request attempts to set a server-owned field.
        """
        if params is not None or kwargs:
            from nemo_retriever.common.policy import _DEFAULT_ALLOWED_DEDUP_KEYS

            merged = _merge_params(params, kwargs)
            _wire_client_stage_params(
                self._pipeline_spec,
                "dedup_params",
                merged,
                method="dedup",
                allowed=_DEFAULT_ALLOWED_DEDUP_KEYS,
            )
        self._record_stage("dedup")
        return self

    def embed(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record an embed stage with optional :class:`EmbedParams` overrides.

        Embedding endpoint URL and API key are server-owned and will be
        rejected if set here.
        """
        if params is not None or kwargs:
            from nemo_retriever.common.policy import _DEFAULT_ALLOWED_EMBED_KEYS

            merged = _merge_params(params, kwargs)
            _wire_client_stage_params(
                self._pipeline_spec,
                "embed_params",
                merged,
                method="embed",
                allowed=_DEFAULT_ALLOWED_EMBED_KEYS,
            )
        self._record_stage("embed")
        return self

    def extract(
        self,
        params: Any = None,
        *,
        split_config: Optional[dict[str, Any]] = None,
        extraction_mode: str = "auto",
        **kwargs: Any,
    ) -> "ServiceIngestor":
        """Record a generic extraction stage.

        ``extraction_mode`` selects the worker's extraction path
        (``'auto'`` default — dispatches by file extension; ``'pdf'``
        forces the PDF path for all inputs, etc.).

        When no ``ExtractParams`` overrides are supplied, ``extract_params``
        is omitted from the wire payload so the worker applies the
        service's server-owned defaults (and the allow-list is not tripped
        by client-side model defaults).
        """
        if params is not None or kwargs:
            from nemo_retriever.common.policy import _DEFAULT_ALLOWED_EXTRACT_KEYS

            merged = _merge_params(params, kwargs)
            _wire_client_stage_params(
                self._pipeline_spec,
                "extract_params",
                merged,
                method="extract",
                allowed=_DEFAULT_ALLOWED_EXTRACT_KEYS,
            )
        self._pipeline_spec["extraction_mode"] = extraction_mode
        if split_config is not None:
            self._pipeline_spec["split_config"] = split_config
        self._record_stage("extract")
        return self

    def extract_image_files(
        self, params: Any = None, *, split_config: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> "ServiceIngestor":
        """Record image-file extraction (``extraction_mode='image'``)."""
        if params is not None or kwargs:
            from nemo_retriever.common.policy import _DEFAULT_ALLOWED_EXTRACT_KEYS

            merged = _merge_params(params, kwargs)
            _wire_client_stage_params(
                self._pipeline_spec,
                "extract_params",
                merged,
                method="extract_image_files",
                allowed=_DEFAULT_ALLOWED_EXTRACT_KEYS,
            )
        self._pipeline_spec["extraction_mode"] = "image"
        if split_config is not None:
            self._pipeline_spec["split_config"] = split_config
        self._record_stage("extract")
        return self

    def filter(self) -> "ServiceIngestor":
        """Record a filter stage."""
        self._record_stage("filter")
        return self

    def split(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record post-extract split / chunking configuration.

        Accepts the same dict shape as :meth:`GraphIngestor.extract`'s
        ``split_config`` keyword (``{"<source_type>": {"max_tokens": …}}``).
        """
        merged: dict[str, Any]
        if isinstance(params, dict):
            merged = dict(params)
        elif params is None:
            merged = {}
        else:
            merged = _params_to_dict(params)
        merged.update(kwargs)
        self._pipeline_spec["split_config"] = merged
        return self

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "ServiceIngestor":
        """Record PDF page-chunking config (per-request).

        The gateway uses this to decide realtime-vs-batch routing
        (chunked docs always go to batch).
        """
        PdfSplitParams.model_validate({})  # cheap sanity touch
        self._pipeline_spec["pdf_split"] = {"pages_per_chunk": int(pages_per_chunk)}
        return self

    # ------------------------------------------------------------------
    # Phase 2: remote sinks — sent via PipelineSpec, gated by SinkUrlAllowlist
    # ------------------------------------------------------------------

    def store(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record an image-asset store stage targeting a remote URI.

        ``storage_uri`` must be a non-local URI (``s3://``, ``gs://``,
        ``azure://``, …) — the worker pod has no view into the caller's
        filesystem. The server's ``sinks.storage_uri_schemes`` allowlist
        gates which schemes are admissible.
        """
        merged = _merge_params(params, kwargs) if (params or kwargs) else StoreParams()
        params_dict = _params_to_dict(merged)
        uri = params_dict.get("storage_uri")
        if uri is not None:
            _require_remote_uri(uri, "store", "storage_uri")
        # ``storage_uri`` is the legitimate sink destination, so we let
        # it through the local denylist check.
        for k in list(params_dict):
            if k != "storage_uri" and k in _SERVER_OWNED_KEYS:
                raise ValueError(f"ServiceIngestor.store(): key {k!r} is server-owned in " "run_mode='service'.")
        from nemo_retriever.common.policy import _DEFAULT_ALLOWED_STORE_KEYS

        params_dict = _filter_policy_allowed(params_dict, _DEFAULT_ALLOWED_STORE_KEYS)
        _set_stage_params(self._pipeline_spec, "store_params", params_dict)
        self._record_stage("store")
        return self

    def store_embed(self) -> "ServiceIngestor":
        _raise_unsupported(
            "store_embed",
            phase_hint=(
                "By design — service run_mode persists embeddings via "
                ".store(...) / .vdb_upload(...) sinks, not the in-process "
                "store_embed helper. Wire a remote storage_uri instead."
            ),
        )

    def udf(
        self,
        udf_function: str,
        udf_function_name: Optional[str] = None,
        phase: Optional[Union[int, str]] = None,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> "ServiceIngestor":
        _raise_unsupported(
            "udf",
            phase_hint=(
                "Phase 5 deferred to a follow-up. Service run_mode requires "
                "the operator to register Python callables in "
                "retriever-service.yaml under 'udfs:' (clients reference "
                "them by name; arbitrary code never crosses the trust "
                "boundary). Until that ships, run UDFs locally via "
                "run_mode='inprocess'."
            ),
        )

    def vdb_upload(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record a vector-DB upload sink targeting a remote LanceDB URI.

        ``vdb_kwargs.lancedb_uri`` must be a non-local URI matching the
        server's ``sinks.vdb_uri_schemes`` allowlist.

        Sidecar metadata (``meta_dataframe`` + ``meta_source_field`` +
        ``meta_fields``) is uploaded eagerly via ``POST /v1/ingest/sidecar``
        and the returned id is shipped on the spec as ``meta_dataframe_id``.
        The original ``meta_dataframe`` (path or in-memory DataFrame) is
        never sent on the wire — the worker pod cannot read it.
        """
        merged = _merge_params(params, kwargs) if (params or kwargs) else VdbUploadParams()
        params_dict = _params_to_dict(merged)

        # Resolve sidecar metadata: if the caller supplied a path or an
        # in-memory DataFrame, upload it now and substitute the returned id.
        meta_df = params_dict.pop("meta_dataframe", None)
        meta_source = params_dict.pop("meta_source_field", None)
        meta_fields = params_dict.pop("meta_fields", None)
        meta_join = params_dict.pop("meta_join_key", "auto")
        if meta_df is not None or meta_source is not None or meta_fields is not None:
            if meta_df is None or meta_source is None or not meta_fields:
                raise ValueError(
                    "ServiceIngestor.vdb_upload(): sidecar metadata requires all "
                    "three of meta_dataframe / meta_source_field / meta_fields."
                )
            sidecar_id = self._upload_sidecar(meta_df)
            params_dict["meta_dataframe_id"] = sidecar_id
            params_dict["meta_source_field"] = str(meta_source)
            params_dict["meta_fields"] = [str(x) for x in meta_fields]
            params_dict["meta_join_key"] = meta_join

        vdb_kwargs = params_dict.get("vdb_kwargs") or {}
        if vdb_kwargs:
            uri = vdb_kwargs.get("lancedb_uri") or vdb_kwargs.get("uri")
            if uri is not None:
                _require_remote_uri(uri, "vdb_upload", "vdb_kwargs.lancedb_uri")
        self._pipeline_spec["vdb_upload_params"] = params_dict
        return self

    def _upload_sidecar(self, meta_df: Any) -> str:
        """POST sidecar metadata to ``/v1/ingest/sidecar`` and return the id.

        Accepts a path (string / PathLike) or an in-memory ``pandas.DataFrame``.
        DataFrames are serialised as parquet to keep the payload compact;
        local paths are streamed as their on-disk bytes with content-type
        inferred from the suffix.
        """
        import io
        import json as _json
        import urllib.request

        from pathlib import Path as _Path

        filename: str
        content_type: str
        payload: bytes

        # In-memory DataFrame (or duck-typed pandas-like) → parquet bytes.
        if hasattr(meta_df, "to_parquet"):
            buf = io.BytesIO()
            try:
                meta_df.to_parquet(buf, index=False)
            except Exception as exc:
                raise ValueError(
                    f"ServiceIngestor.vdb_upload(): failed to serialise sidecar " f"DataFrame to parquet: {exc}"
                ) from exc
            payload = buf.getvalue()
            filename = "sidecar.parquet"
            content_type = "application/x-parquet"
        else:
            # Treat as filesystem path.
            path = _Path(str(meta_df))
            if not path.is_file():
                raise FileNotFoundError(f"ServiceIngestor.vdb_upload(): sidecar metadata file not found: {path}")
            payload = path.read_bytes()
            filename = path.name
            suf = path.suffix.lower()
            if suf == ".parquet" or suf == ".pq":
                content_type = "application/x-parquet"
            elif suf in (".json", ".jsonl"):
                content_type = "application/json"
            else:
                content_type = "text/csv"

        # Build a minimal multipart/form-data request — avoids dragging in
        # an httpx dependency where urllib already works.
        import re
        import secrets

        safe_filename = re.sub(r"[^\w.\-]", "_", filename) or "upload"
        boundary = f"----nrlib-sidecar-{secrets.token_hex(16)}"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'.encode())
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(payload)
        body.write(f"\r\n--{boundary}--\r\n".encode())

        url = f"{self._base_url}/v1/ingest/sidecar"
        req = urllib.request.Request(url, data=body.getvalue(), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        if self._api_token:
            req.add_header("Authorization", f"Bearer {self._api_token}")
        try:
            with urllib.request.urlopen(req, timeout=self._request_timeout_s) as resp:
                body_json = _json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"ServiceIngestor.vdb_upload(): failed to upload sidecar to {url}: {exc}") from exc

        sidecar_id = body_json.get("sidecar_id")
        if not sidecar_id:
            raise RuntimeError(
                f"ServiceIngestor.vdb_upload(): sidecar upload response missing sidecar_id: {body_json!r}"
            )
        logger.debug("Uploaded sidecar %s (%d bytes)", sidecar_id, len(payload))
        return sidecar_id

    def save_intermediate_results(self, output_dir: str) -> "ServiceIngestor":
        _raise_unsupported(
            "save_intermediate_results",
            phase_hint=(
                "By design — service workers don't expose stage-by-stage "
                "outputs; they run the whole pipeline to completion before "
                "returning a result. For per-stage debugging use "
                "run_mode='inprocess'. To capture final outputs use "
                ".save_to_disk(output_directory=...) instead."
            ),
        )

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "ServiceIngestor":
        """Stream per-document results to ``output_directory`` as they finish.

        Each completed document produces one JSON file (or ``.json.gz`` when
        ``compression='gzip'``) named ``<document_id>.json[.gz]`` whose
        contents are the worker's transport-serialized pipeline rows
        (see :mod:`nemo_retriever.ingest_results`) — the same column
        layout as ``GraphIngestor.ingest()`` in local run modes.

        Important differences from graph mode:

        * Large binary columns (``bytes``, ``page_image``, ``images``,
          ``charts``, ``tables``) are stripped server-side before the
          rows leave the worker. Use :meth:`store` to persist image
          assets to a remote URI; the local-disk artifact only carries
          the structured metadata.
        * The client does the writing — the server has no view into the
          caller's filesystem. ``cleanup`` is accepted for API parity
          with graph mode but has no server-side effect today.
        """
        if output_directory is None:
            raise ValueError("ServiceIngestor.save_to_disk(): output_directory is required.")
        if compression not in (None, "gzip"):
            raise ValueError(
                f"save_to_disk(compression={compression!r}): only None or 'gzip' " "are supported in service run_mode."
            )
        target = Path(output_directory)
        target.mkdir(parents=True, exist_ok=True)
        self._save_to_disk_dir = target
        self._save_to_disk_compression = compression
        self._save_to_disk_cleanup = cleanup
        return self

    def caption(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record a caption stage backed by the server's remote VLM endpoint.

        Captioning non-image documents automatically uses default image
        deduplication unless :meth:`dedup` explicitly disables both
        mechanisms. Standalone image documents remain exempt.

        Behavioural knobs — ``prompt``, ``system_prompt``, ``batch_size``,
        ``context_text_max_chars``, ``caption_infographics``, and generic
        sampling params (``temperature``, ``max_tokens``, ``top_p``,
        ``top_k``) — are honored. Trust-sensitive fields
        (``endpoint_url``, ``api_key``, ``model_name``) and
        local-execution fields (``device``, ``hf_cache_dir``,
        ``tensor_parallel_size``, ``gpu_memory_utilization``) are never
        transmitted. Prohibited keyword overrides fail fast. Non-default
        prohibited values on a ``CaptionParams`` model also fail fast, except
        ``api_key`` because environment auto-fill makes caller intent
        ambiguous. Prohibited mapping values and model defaults are stripped
        instead. The operator-configured remote endpoint is the only path to a
        caption NIM.

        Parameters
        ----------
        params
            Optional :class:`CaptionParams` instance or parameter mapping to
            transmit to the service.
        **kwargs
            Field values used directly when ``params`` is omitted or applied
            as overrides when ``params`` is a parameter model.

        Returns
        -------
        ServiceIngestor
            This ingestor instance for fluent chaining.

        Raises
        ------
        TypeError
            If ``params`` cannot be serialized as a parameter mapping.
        ValueError
            If a prohibited keyword override is supplied, or ``CaptionParams``
            sets a non-default prohibited field other than ``api_key``.
        """
        trust_sensitive = {"endpoint_url", "api_key", "model_name"}
        local_only = {
            "device",
            "hf_cache_dir",
            "tensor_parallel_size",
            "gpu_memory_utilization",
        }

        # Identify which keys the caller actually meant to pass. The
        # signal for kwargs is unambiguous (any key in **kwargs is by
        # definition caller-provided); for a passed-in CaptionParams
        # instance we compare against class defaults, with one wrinkle:
        # ``api_key`` is auto-populated by the model validator from the
        # NVIDIA_API_KEY env var, so we cannot distinguish "caller set
        # this" from "validator set this" — we conservatively strip the
        # value either way and only raise when the caller used kwargs.
        explicit_keys: set[str] = set(kwargs.keys())
        if isinstance(params, CaptionParams):
            class_defaults = {name: field.default for name, field in CaptionParams.model_fields.items()}
            for k in trust_sensitive | local_only:
                if k == "api_key":
                    continue  # see comment above; the env-var auto-fill is ambiguous.
                val = getattr(params, k, None)
                if val is not None and val != class_defaults.get(k):
                    explicit_keys.add(k)

        bad_trust = sorted(explicit_keys & trust_sensitive)
        if bad_trust:
            raise ValueError(
                f"ServiceIngestor.caption(): keys {bad_trust!r} are server-owned in "
                "run_mode='service'. The operator configures the caption "
                "endpoint via retriever-service.yaml (nim_endpoints.caption_invoke_url)."
            )
        bad_local = sorted(explicit_keys & local_only)
        if bad_local:
            raise ValueError(
                f"ServiceIngestor.caption(): keys {bad_local!r} configure local "
                "in-process GPU execution and have no effect against a remote "
                "caption endpoint. Remove them and rely on the server-owned "
                "model / endpoint."
            )

        merged = _merge_params(params, kwargs) if (params or kwargs) else CaptionParams()
        params_dict = _params_to_dict(merged)
        # Drop both classes of keys before the spec leaves the client —
        # the server's allowlist rejects them anyway, but failing fast
        # at the boundary keeps the network message small and the policy
        # error rare in practice.
        scrubbed = {k: v for k, v in params_dict.items() if k not in trust_sensitive | local_only}
        self._pipeline_spec["caption_params"] = scrubbed
        self._record_stage("caption")
        return self

    def webhook(self, params: Any = None, **kwargs: Any) -> "ServiceIngestor":
        """Record a webhook-notification stage targeting a remote URL.

        ``endpoint_url`` must match one of the server's
        ``sinks.webhook_url_prefixes``. Without that allowlist the
        service rejects webhook requests entirely so worker egress
        cannot be steered by clients.
        """
        merged = _merge_params(params, kwargs) if (params or kwargs) else WebhookParams()
        params_dict = _params_to_dict(merged)
        endpoint = params_dict.get("endpoint_url")
        if endpoint is None:
            raise ValueError(
                "ServiceIngestor.webhook(): endpoint_url is required "
                "(unlike inprocess run_mode, an empty endpoint_url is treated "
                "as misconfiguration in service mode)."
            )
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(
                "ServiceIngestor.webhook(): endpoint_url must be a fully-qualified "
                f"http(s):// URL; got {endpoint!r}."
            )
        self._pipeline_spec["webhook_params"] = params_dict
        self._record_stage("webhook")
        return self

    def _ingest_events_with_result_client(
        self,
        *,
        retain_results: bool,
        result_schema: ResultSchema,
        return_embeddings: bool,
        return_images: bool,
    ) -> Iterator[tuple[dict[str, Any], httpx.Client | None]]:
        """Yield ingest events while owning the optional shared result client."""
        client_context = self._new_result_fetch_client() if retain_results else nullcontext(None)
        with client_context as result_client:
            for evt in self.ingest_stream(
                retain_results=retain_results,
                result_schema=result_schema,
                return_embeddings=return_embeddings,
                return_images=return_images,
            ):
                yield evt, result_client

    # ------------------------------------------------------------------
    # Execution — sync materialized
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> Any:
        """Block until every document has finished processing on the server.

        Internally opens exactly one server-side job aggregate for the
        full input set (sized to ``len(documents)``). The aggregate
        ``job_id`` is captured from the first ``job_created`` event and
        exposed on :class:`ServiceIngestResult` so the caller can call
        ``GET /v1/ingest/job/{job_id}`` for follow-up status.

        Parameters
        ----------
        params
            Optional :class:`IngestExecuteParams` (or plain ``dict``)
            carrying execute-time flags.  In service run_mode only
            ``return_failures`` / ``return_traces`` / ``return_results`` /
            ``result_schema`` / ``return_embeddings`` / ``return_images``
            are honored — every other field is recorded on the server-side
            pipeline spec.
        **kwargs
            Same execute-time flags may be passed individually.  Anything
            not recognised is silently ignored (server-side execution
            in service mode is driven by the pipeline spec, not by
            execute-time knobs).
        return_results
            When ``True`` (default), fetch each completed document's
            ``result_data`` from ``GET /v1/ingest/status/{id}`` and
            expose the combined rows on ``result.dataframe`` as a
            ``pandas.DataFrame``. Set to ``False`` to skip those HTTP
            round-trips when only job metadata is needed.
        result_schema
            ``"legacy"`` (default) preserves the existing service
            DataFrame column layout with bulky values stripped and emits
            a deprecation warning when result rows are retained.
            ``"compact"`` opts into the future compact row schema.
        return_embeddings
            Include embedding vectors in legacy or compact result rows.
            Defaults to ``False`` to avoid large responses.
        return_images
            When using legacy result rows, include raw image payloads instead
            of stripping them from transport cells. Defaults to ``False`` to
            avoid large responses.

        Returns
        -------
        ServiceIngestResult
            When neither ``return_failures`` nor ``return_traces`` is
            set — a list subclass of per-document completion events with
            extra ``job_id`` / ``failures`` / ``document_ids`` /
            ``elapsed_s`` / ``job_status`` / ``dataframe`` attributes.
        tuple
            With ``return_failures=True`` only — ``(result, failures)``.
            With ``return_traces=True`` only — ``(result, traces)``.
            With both — ``(result, failures, traces)``.  ``failures``
            mirrors ``result.failures``; ``traces`` is the ordered list
            of raw SSE event dicts observed during the run, useful for
            debugging pipeline behaviour without re-running the job.
        """
        return_failures, return_traces, return_results, result_schema, return_embeddings, return_images = (
            self._resolve_execute_flags(params, kwargs)
        )
        del params, kwargs
        self._validate_input_sources(self._inline_texts)
        if not self._documents and not self._buffers and not self._urls and is_blank_inline_corpus(self._inline_texts):
            self._document_ids.clear()
            self._last_run_elapsed_s = 0.0
            self._last_job_id = None
            result = ServiceIngestResult()
            if return_results:
                result.dataframe = _empty_service_result_dataframe(result_schema)
            if return_failures and return_traces:
                return result, [], []
            if return_failures or return_traces:
                return result, []
            return result

        retain_results = return_results or self._save_to_disk_dir is not None
        if retain_results and result_schema == "legacy":
            warnings.warn(_LEGACY_RESULT_SCHEMA_DEPRECATION, DeprecationWarning, stacklevel=2)
        result = ServiceIngestResult()
        traces: list[dict[str, Any]] = []
        rows_by_document: dict[str, list[dict[str, Any]]] = {}
        t0 = time.monotonic()

        documents_completed = 0
        documents_failed = 0
        total_uploaded = 0

        for evt, result_client in self._ingest_events_with_result_client(
            retain_results=retain_results,
            result_schema=result_schema,
            return_embeddings=return_embeddings,
            return_images=return_images,
        ):
            if return_traces:
                traces.append(evt)
            event_type = evt.get("event")

            if event_type == "job_created":
                result.job_id = evt.get("job_id") or result.job_id
                result.trace_id = evt.get("trace_id") or result.trace_id
                continue

            if event_type in ("job_finalized", "job_partial", "job_failed"):
                if event_type == "job_finalized":
                    result.job_status = "completed"
                elif event_type == "job_partial":
                    result.job_status = "partial_success"
                else:
                    result.job_status = "failed"
                continue

            if event_type == "job_progress" or event_type == "job_started":
                continue

            if event_type == "upload_complete":
                total_uploaded += 1
                document_id = evt.get("document_id")
                filename = evt.get("filename")
                if document_id and filename:
                    result.document_filenames[str(document_id)] = str(filename)
                if result.job_id is None:
                    # Race: SSE delivered an upload_complete before the
                    # generator yielded job_created. Fall back to the
                    # job_id stamped on the per-doc event by the client.
                    result.job_id = evt.get("job_id") or result.job_id
                print(
                    f"\r  Job {result.job_id or '?'}  |  "
                    f"Uploaded: {total_uploaded}  |  "
                    f"Completed: {documents_completed}  |  "
                    f"Failed: {documents_failed}",
                    end="",
                    flush=True,
                )

            elif event_type == "document_complete":
                status = evt.get("status", "completed")
                if status not in ("completed", "failed"):
                    continue
                if status == "failed":
                    documents_failed += 1
                    error = evt.get("error", "unknown error")
                    doc_id = evt.get("document_id", "?")
                    result.failures.append((doc_id, error))
                else:
                    documents_completed += 1
                    doc_id = evt.get("document_id", "")
                    if return_results or self._save_to_disk_dir is not None:
                        try:
                            rows = self._materialize_completed_document(
                                doc_id,
                                return_results=return_results,
                                client=result_client,
                            )
                            if rows is not None and return_results:
                                rows_by_document[doc_id] = rows
                        except Exception as exc:
                            label = "return_results" if return_results else "save_to_disk"
                            logger.warning("%s: failed to fetch/persist %s: %s", label, doc_id, exc)
                            result.failures.append((doc_id, f"{label}: {exc}"))
                result.append(evt)
                print(
                    f"\r  Job {result.job_id or '?'}  |  "
                    f"Uploaded: {total_uploaded}  |  "
                    f"Completed: {documents_completed}  |  "
                    f"Failed: {documents_failed}",
                    end="",
                    flush=True,
                )

            elif event_type == "upload_failed":
                fname = evt.get("filename", "?")
                error = evt.get("error", "unknown")
                result.failures.append((fname, f"upload failed: {error}"))

        if total_uploaded > 0:
            print()

        result.document_ids = list(self._document_ids)
        result.elapsed_s = time.monotonic() - t0
        if return_results:
            doc_order = [d for d in self._document_ids if d in rows_by_document] or list(rows_by_document)
            result.dataframe = concat_ingest_results(rows_by_document, doc_order)
        self._last_run_elapsed_s = result.elapsed_s
        # Cache the job_id on the ingestor for the get_status() /
        # remaining_jobs() accessors so they can target the job
        # aggregate endpoints once J6 wiring is opted in (kept
        # backwards compatible — get_status() still uses document_ids).
        self._last_job_id = result.job_id

        if return_failures and return_traces:
            return result, list(result.failures), traces
        if return_failures:
            return result, list(result.failures)
        if return_traces:
            return result, traces
        return result

    @staticmethod
    def _normalize_result_schema(value: Any) -> ResultSchema:
        schema = str(value or "legacy").strip().lower()
        if schema not in ("legacy", "compact"):
            raise ValueError("result_schema must be 'legacy' or 'compact'")
        return schema  # type: ignore[return-value]

    @classmethod
    def _resolve_execute_flags(
        cls, params: Any, kwargs: dict[str, Any]
    ) -> tuple[bool, bool, bool, ResultSchema, bool, bool]:
        """Read execute-time flags from ``params`` and/or ``kwargs``.

        kwargs take precedence over fields on ``params`` when both supply
        the same flag, mirroring the precedence used by
        :func:`nemo_retriever.ingestor._merge_params`.
        """

        def _from_params(name: str, *, default: bool) -> bool:
            if isinstance(params, IngestExecuteParams):
                return bool(getattr(params, name, default))
            if isinstance(params, dict):
                if name in params:
                    return bool(params[name])
                return default
            return default

        def _from_params_value(name: str, *, default: Any) -> Any:
            if isinstance(params, IngestExecuteParams):
                return getattr(params, name, default)
            if isinstance(params, dict):
                return params.get(name, default)
            return default

        return_failures = (
            bool(kwargs["return_failures"])
            if "return_failures" in kwargs
            else _from_params("return_failures", default=False)
        )
        return_traces = (
            bool(kwargs["return_traces"]) if "return_traces" in kwargs else _from_params("return_traces", default=False)
        )
        return_results = (
            bool(kwargs["return_results"])
            if "return_results" in kwargs
            else _from_params("return_results", default=True)
        )
        result_schema = cls._normalize_result_schema(
            kwargs["result_schema"]
            if "result_schema" in kwargs
            else _from_params_value("result_schema", default="legacy")
        )
        return_embeddings = (
            bool(kwargs["return_embeddings"])
            if "return_embeddings" in kwargs
            else _from_params("return_embeddings", default=False)
        )
        return_images = (
            bool(kwargs["return_images"]) if "return_images" in kwargs else _from_params("return_images", default=False)
        )
        return return_failures, return_traces, return_results, result_schema, return_embeddings, return_images

    # ------------------------------------------------------------------
    # Execution — sync streaming
    # ------------------------------------------------------------------

    def ingest_stream(
        self,
        *,
        retain_results: bool = False,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Sync generator yielding events as documents are processed.

        Yields dicts with:

        * ``{"event": "job_created", "job_id": ..., "expected_documents": ...}``
        * ``{"event": "upload_complete", "filename": ..., "document_id": ..., "job_id": ...}``
        * ``{"event": "document_complete", "document_id": ..., "status": ..., "job_id": ..., ...}``
        * ``{"event": "upload_failed", "filename": ..., "error": ..., "job_id": ...}``
        * ``{"event": "job_progress", "job_id": ..., "completed": ..., "failed": ..., ...}``
        * ``{"event": "job_finalized"|"job_partial"|"job_failed", "job_id": ..., ...}``
        """
        result_schema = self._normalize_result_schema(result_schema)
        self._validate_input_sources(self._inline_texts)
        return self._ingest_stream_with_retain(
            retain_results,
            result_schema=result_schema,
            return_embeddings=return_embeddings,
            return_images=return_images,
        )

    # ------------------------------------------------------------------
    # Execution — async streaming
    # ------------------------------------------------------------------

    async def aingest_stream(
        self,
        *,
        retain_results: bool = False,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async generator yielding events as documents are processed."""
        result_schema = self._normalize_result_schema(result_schema)
        self._reset_run_state()
        collection_task = asyncio.create_task(asyncio.to_thread(self._collect_inputs))
        try:
            try:
                files = await asyncio.shield(collection_task)
            except asyncio.CancelledError:
                await self._finish_cancelled_input_collection(collection_task)
                raise
            for event in self._url_failure_events():
                yield event
            if not files:
                return

            async for evt in self._aingest_stream_impl(
                files,
                retain_results=retain_results,
                result_schema=result_schema,
                return_embeddings=return_embeddings,
                return_images=return_images,
            ):
                if evt.get("event") == "upload_complete":
                    did = evt.get("document_id")
                    if did:
                        self._document_ids.append(did)
                yield evt
        finally:
            self._cleanup_url_inputs()

    # ------------------------------------------------------------------
    # Async helper used by both sync and async streaming entry points
    # ------------------------------------------------------------------

    def _ingest_stream_with_retain(
        self,
        retain_results: bool,
        *,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Like :meth:`ingest_stream` but passes server-side retention to the HTTP client."""
        self._reset_run_state()
        files = self._collect_inputs()
        failure_events = self._url_failure_events()
        if not files:
            try:
                yield from failure_events
            finally:
                self._cleanup_url_inputs()
            return

        def _record_doc_id(evt: dict[str, Any]) -> None:
            if evt.get("event") == "upload_complete":
                did = evt.get("document_id")
                if did:
                    self._document_ids.append(did)

        def _factory():
            return self._wrap_for_capture(
                self._aingest_stream_impl(
                    files,
                    retain_results=retain_results,
                    result_schema=result_schema,
                    return_embeddings=return_embeddings,
                    return_images=return_images,
                ),
                _record_doc_id,
            )

        bridge = _AsyncToSyncBridge(_factory)

        def _events() -> Iterator[dict[str, Any]]:
            try:
                yield from failure_events
                yield from bridge
            finally:
                self._cleanup_url_inputs()

        yield from _events()

    async def _aingest_stream_impl(
        self,
        files: list[UploadInput],
        *,
        retain_results: bool = False,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        client = RetrieverServiceClient(
            base_url=self._base_url,
            max_concurrency=self._max_concurrency,
            api_token=self._api_token,
        )
        pipeline_payload = self._pipeline_payload(
            result_schema=result_schema,
            return_embeddings=return_embeddings,
            return_images=return_images,
        )
        async for evt in client.aingest_documents_stream(
            files=files,
            pipeline_spec=pipeline_payload,
            retain_results=retain_results,
        ):
            yield self._restore_url_event(evt)

    @staticmethod
    async def _wrap_for_capture(
        agen: AsyncIterator[dict[str, Any]],
        on_event,
    ) -> AsyncIterator[dict[str, Any]]:
        """Pass-through wrapper that lets the sync bridge capture document_ids."""
        async for evt in agen:
            on_event(evt)
            yield evt

    # ------------------------------------------------------------------
    # Async-future API
    # ------------------------------------------------------------------

    def ingest_async(
        self,
        *,
        return_failures: bool = False,
        return_traces: bool = False,
        return_results: bool = True,
        result_schema: ResultSchema = "legacy",
        return_embeddings: bool = False,
        return_images: bool = False,
    ) -> Any:
        """Run :meth:`ingest` on a background thread; return a ``Future``.

        The flags are forwarded to :meth:`ingest`, so calling
        ``future.result()`` produces the same tuple/list shape that a
        direct synchronous call with the same flags would return.
        """
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ServiceIngestorAsync")
        return executor.submit(
            self.ingest,
            return_failures=return_failures,
            return_traces=return_traces,
            return_results=return_results,
            result_schema=result_schema,
            return_embeddings=return_embeddings,
            return_images=return_images,
        )

    # ------------------------------------------------------------------
    # Status & document-counter accessors
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, str]:
        """Return ``{document_id: status}`` for every document submitted so far."""
        if not self._document_ids:
            return {}
        url = f"{self._base_url}/v1/ingest/status/batch"
        with httpx.Client(timeout=30.0, headers=self._auth_headers) as client:
            try:
                resp = client.post(url, json={"ids": self._document_ids})
                resp.raise_for_status()
                items = resp.json().get("items", {})
                return {did: info.get("status", "unknown") for did, info in items.items()}
            except Exception as exc:
                logger.warning("Could not fetch bulk status: %s", exc)
                return {did: "unknown" for did in self._document_ids}

    def completed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "completed")

    def failed_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s == "failed")

    def cancelled_jobs(self) -> int:
        return 0

    def remaining_jobs(self) -> int:
        return sum(1 for s in self.get_status().values() if s in ("processing", "unknown"))

    # ------------------------------------------------------------------
    # Cancel — not supported (no server endpoint)
    # ------------------------------------------------------------------

    def cancel(self, job_id: str | None = None) -> dict[str, Any]:
        """Not supported — the server does not expose a cancel endpoint."""
        raise NotImplementedError(
            "Cancel is not supported in service mode. " "The server does not currently expose a cancel endpoint."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    async def _finish_cancelled_input_collection(collection_task: asyncio.Task[list[UploadInput]]) -> None:
        """Wait for a shielded preparation thread before its owner cleans up."""

        while not collection_task.done():
            try:
                await asyncio.shield(collection_task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            collection_task.result()
        except Exception:
            logger.exception("Service input collection failed while cancellation was being handled")

    def _reset_run_state(self) -> None:
        self._document_ids.clear()
        self._last_job_id = None
        self._last_run_elapsed_s = 0.0

    def _cleanup_url_inputs(self) -> None:
        cleanup_fetched_urls(self._fetched_urls)
        self._fetched_urls = []
        self._url_source_map = {}

    def _restore_url_event(self, event: dict[str, Any]) -> dict[str, Any]:
        filename = event.get("filename")
        restored = restore_url_source_value(filename, self._url_source_map)
        if restored == filename:
            return event
        translated = dict(event)
        translated["filename"] = restored
        return translated

    def _url_failure_events(self) -> list[dict[str, Any]]:
        return [
            {"event": "upload_failed", "filename": failure.url, "error": f"URL fetch failed: {failure.message}"}
            for failure in self._url_failures
        ]

    def _has_mixed_inline_sources(self) -> bool:
        return bool(self._inline_texts) and bool(self._documents or self._buffers or self._urls)

    def _collect_inputs(self) -> list[UploadInput]:
        """Gather filesystem and in-memory inputs for the service client."""
        self._validate_input_sources(self._inline_texts)
        self._cleanup_url_inputs()
        if not self._documents and not self._buffers and not self._urls and is_blank_inline_corpus(self._inline_texts):
            return []

        files: list[UploadInput] = [Path(p) for p in self._documents]

        self._fetched_urls, self._url_failures = fetch_urls(self._urls, self._url_fetch_params)
        self._url_source_map = {item.transport_path: item.url for item in self._fetched_urls}
        for item in self._fetched_urls:
            files.append(
                FileUpload(
                    path=item.local_path,
                    filename=item.transport_path,
                    content_type=item.content_type,
                    classification_filename=item.classification_filename,
                    metadata={"_nrl_source_url": item.url},
                )
            )

        if self._buffers:
            import tempfile

            tmp_dir = Path(tempfile.mkdtemp(prefix="service_ingestor_"))
            for name, buf in self._buffers:
                target = tmp_dir / name
                target.write_bytes(buf.getvalue())
                files.append(target)

        for index, text in enumerate(self._inline_texts or []):
            source_id = inline_text_source_id(index)
            files.append(
                InMemoryUpload(
                    filename=source_id,
                    content=text.encode("utf-8"),
                    content_type="text/plain; charset=utf-8",
                    classification_filename=f"inline-{index:08d}.txt",
                )
            )

        return files
