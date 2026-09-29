# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GraphIngestor: builds operator graphs directly and runs them via an executor.

Unlike the high-level :func:`create_ingestor` factory this class constructs
the :class:`~nemo_retriever.graph.Graph` itself—using
:func:`~nemo_retriever.graph.ingestor_runtime.build_graph`—and
passes it to a :class:`~nemo_retriever.graph.RayDataExecutor` or
:class:`~nemo_retriever.graph.InprocessExecutor` for execution.

Usage::

    from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
    from nemo_retriever.common.params import ExtractParams, EmbedParams

    result_ds = (
        GraphIngestor(run_mode="inprocess")
        .files(["/data/*.pdf"])
        .extract(ExtractParams(method="pdfium"))
        .embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"))
        .ingest()
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Self, Sequence, Tuple, Union

from nemo_retriever.graph import InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.executor import arrow_table_to_pandas, call_pandas_function_on_arrow
from nemo_retriever.ingestor.branch_extraction import ExtractionBranchExecutor, merge_node_overrides
from nemo_retriever.graph.ingestor_runtime import (
    batch_tuning_to_node_overrides,
    build_graph,
    default_concurrency_node_names,
)
from nemo_retriever.ingestor.manifest import (
    DEFAULT_AUDIO_SPLIT_INTERVAL,
    DEFAULT_VIDEO_FRAME_FPS,
    ExtractionBranchPlan,
    ResolvedExtractionInputs,
    build_input_manifest,
    default_asr_params,
    format_branch_summary,
    plan_extraction_branches,
    resolve_branch_extraction_inputs,
)
from nemo_retriever.ingestor.plans import dedup_params_enabled, resolve_effective_dedup_params
from nemo_retriever.ingestor import ingestor
from nemo_retriever.common.inline_text import (
    inline_text_source_id,
    is_blank_inline_corpus,
    is_inline_text_source,
    normalize_inline_texts,
)
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    IngestExecuteParams,
    NO_API_KEY,
    StoreParams,
    TextChunkParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
    VdbUploadParams,
    WebhookParams,
    SPLIT_CONFIG_VALID_KEYS,
    resolve_split_params,
)
from nemo_retriever.common.input_files import (
    PDF_DOCUMENT_INPUT_TYPES,
    _is_explicit_glob_path,
    expand_input_directories,
    expand_input_file_patterns,
    input_type_for_path,
)
from nemo_retriever.common.remote_auth import resolve_remote_api_key
from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime
from nemo_retriever.common.ray_resource_hueristics import gather_cluster_resources
from nemo_retriever.common.stage_errors import (
    ERROR_FIELD_KEYS,
    is_populated_error_field,
    iter_stage_errors_from_value,
)
from nemo_retriever.common.modality.txt.split import empty_text_chunks_df

_ERROR_FIELD_KEYS = ERROR_FIELD_KEYS
_REMOTE_EMBED_ENDPOINT_FIELDS = ("embedding_endpoint", "embed_invoke_url")
_DEFAULT_PAGE_ELEMENTS_COLUMN = "page_elements_v3"
_DEFAULT_EMBED_COLUMN = "text_embeddings_1b_v2"
_ERROR_MESSAGE_LIMIT = 256
_SOURCE_IDENTIFIER_COLUMNS = ("document_id", "path", "source_path", "metadata", "source_id", "source_name")
logger = logging.getLogger(__name__)
_HTTP_STATUS_FIELDS: tuple[str, ...] = ("status_code", "http_status", "status", "code")
_EXPLICIT_MODE_INPUT_TYPES: dict[str, frozenset[str]] = {
    "pdf": PDF_DOCUMENT_INPUT_TYPES,
    "image": frozenset({"image"}),
    "text": frozenset({"txt"}),
    "html": frozenset({"html"}),
    "audio": frozenset({"audio"}),
    "video": frozenset({"video"}),
}


@dataclass(frozen=True)
class _StageDiagnostic:
    """Resolved diagnostic info for one stage error column.

    The :class:`GraphIngestor` builds one of these per remote-NIM column
    at error-raising time so the formatter can attribute each row-level
    error to a concrete stage, NIM URL, and (when present in the payload)
    HTTP status code. ``display_name`` and ``invoke_url`` are best-effort:
    when the caller raises :class:`GraphIngestionError` directly (without
    the resolver), they fall back to ``None`` and the formatter renders
    the legacy ``row N, column X`` shape.
    """

    column: str
    display_name: str
    invoke_url: str | None
    model_name: str | None = None
    role: str | None = None


class GraphIngestionError(RuntimeError):
    """Raised when graph ingestion stages report structured row-level errors.

    The exception message is built to be self-diagnosing: when the
    caller provides ``stage_diagnostics`` (a mapping from the dataframe
    column the error landed in to a :class:`_StageDiagnostic` describing
    the originating NIM), each row in the rendered message names the
    stage and the configured invoke URL, and the message gains a
    ``Troubleshooting:`` footer with concrete next steps for the
    observed (stage, HTTP status) tuples.

    Backwards compatible signature: ``GraphIngestionError(records)``
    still works and produces the legacy message shape.
    """

    def __init__(
        self,
        records: Any,
        stage_diagnostics: dict[str, _StageDiagnostic] | None = None,
    ) -> None:
        # Exception pickling reconstructs an instance from ``BaseException.args``.
        # Treat a preformatted message as one record rather than an iterable of
        # characters so older serialized instances also fail safely.
        if isinstance(records, str):
            self.records = [records]
        elif isinstance(records, (list, tuple)):
            self.records = list(records)
        else:
            self.records = [records]
        self.stage_diagnostics = dict(stage_diagnostics) if stage_diagnostics else {}
        super().__init__(_format_stage_error_message(self.records, self.stage_diagnostics))

    def __reduce__(self) -> tuple[Any, tuple[list[Any], dict[str, _StageDiagnostic]]]:
        """Reconstruct the exception from structured data across processes."""
        return type(self), (self.records, self.stage_diagnostics)


def _normalize_stage_error_record(record: Any) -> dict[str, Any] | None:
    """Coerce a stage-error record to the dict shape expected by formatting."""
    if isinstance(record, str):
        text = record.strip()
        if not text:
            return None
        return {"row_index": None, "column": None, "path": "error", "error": text}
    if not isinstance(record, dict):
        return {"row_index": None, "column": None, "path": "error", "error": record}
    return record


def _format_stage_error_message(
    records: list[Any],
    stage_diagnostics: dict[str, _StageDiagnostic] | None = None,
) -> str:
    limit = 5
    diagnostics = stage_diagnostics or {}
    details: list[str] = []
    observed_status_codes: dict[str, set[int | None]] = {}

    for raw in records[:limit]:
        record = _normalize_stage_error_record(raw)
        if record is None:
            continue
        column = record.get("column")
        diag = diagnostics.get(column) if isinstance(column, str) else None
        status_code = _extract_http_status_code(record.get("error"))
        if isinstance(column, str):
            observed_status_codes.setdefault(column, set()).add(status_code)
        stage_prefix = _render_stage_prefix(diag, status_code)
        details.append(
            "row {row_index}, column {column}{stage}, path {path}: {summary}".format(
                row_index=record.get("row_index"),
                column=column,
                stage=stage_prefix,
                path=record.get("path"),
                summary=_summarize_error_payload(record.get("error")),
            )
        )

    more = "" if len(records) <= limit else f" ({len(records) - limit} more)"
    troubleshooting = _format_troubleshooting_footer(
        records=records,
        diagnostics=diagnostics,
        observed_status_codes=observed_status_codes,
    )
    body = (
        "Graph ingestion detected row-level errors from an explicitly "
        "configured remote NIM endpoint"
        f"{more}. " + "; ".join(details)
    )
    if troubleshooting:
        body = body + " " + troubleshooting
    return body


def _render_stage_prefix(diag: _StageDiagnostic | None, status_code: int | None) -> str:
    """Build the bracketed ``[stage=… url=… http=…]`` suffix per row."""
    parts: list[str] = []
    if diag is not None:
        parts.append(f"stage={diag.display_name}")
        if diag.invoke_url:
            parts.append(f"url={diag.invoke_url}")
    if status_code is not None:
        parts.append(f"http={status_code}")
    if not parts:
        return ""
    return " [" + " ".join(parts) + "]"


def _format_troubleshooting_footer(
    *,
    records: list[Any],
    diagnostics: dict[str, _StageDiagnostic],
    observed_status_codes: dict[str, set[int | None]],
) -> str:
    """Build a ``Troubleshooting:`` footer keyed off (stage, status) pairs.

    Surfaces actionable next steps for the most common remote-NIM
    failure modes (network unreachable, auth, 4xx vs 5xx). When no
    diagnostics are available the footer is omitted to avoid printing
    generic advice next to the legacy message shape.
    """
    if not diagnostics:
        return ""

    hints: list[str] = []
    seen_columns: set[str] = set()
    for raw in records:
        record = _normalize_stage_error_record(raw)
        if record is None:
            continue
        column = record.get("column")
        if not isinstance(column, str) or column in seen_columns:
            continue
        seen_columns.add(column)
        diag = diagnostics.get(column)
        if diag is None:
            continue
        statuses = observed_status_codes.get(column, set())
        hint = _hint_for_stage(diag, statuses)
        if hint:
            hints.append(hint)

    if not hints:
        return ""
    return "Troubleshooting: " + " ".join(hints)


def _hint_for_stage(diag: _StageDiagnostic, statuses: set[int | None]) -> str:
    """Return a one-line, actionable hint for *diag* given observed *statuses*."""
    bucket = _classify_status_codes(statuses)
    url_clause = f" at {diag.invoke_url}" if diag.invoke_url else ""
    name = diag.display_name

    if bucket == "auth":
        return (
            f"{name}{url_clause} returned an auth error \u2014 verify "
            "NGC_API_KEY / NVIDIA_API_KEY is set on the service pod and "
            "that the NIM accepts the same credentials."
        )
    if bucket == "client":
        return (
            f"{name}{url_clause} returned a 4xx client error \u2014 "
            "check the request payload shape (file format, page size, "
            "model name) against the NIM's expected input schema."
        )
    if bucket == "server":
        return (
            f"{name}{url_clause} returned a 5xx server error \u2014 "
            "inspect the NIM pod logs, GPU memory, and readiness "
            "probes; the upstream model may be saturated or crashed."
        )
    return (
        f"{name}{url_clause} reported a row-level error \u2014 verify "
        "the NIM is reachable from the retriever service pod "
        f"(e.g. `kubectl exec ... -- curl -sS {diag.invoke_url or '<invoke_url>'}` "
        "should return a non-empty response) and that its readiness "
        "endpoint is healthy."
    )


def _classify_status_codes(statuses: set[int | None]) -> str:
    """Bucket a set of observed HTTP statuses into one diagnostic class."""
    concrete = {s for s in statuses if isinstance(s, int)}
    if any(s in (401, 403) for s in concrete):
        return "auth"
    if any(500 <= s < 600 for s in concrete):
        return "server"
    if any(400 <= s < 500 for s in concrete):
        return "client"
    return "generic"


def _extract_http_status_code(error: Any) -> int | None:
    """Return the first HTTP status integer found in common error payloads."""
    if isinstance(error, dict):
        for field in _HTTP_STATUS_FIELDS:
            value = error.get(field)
            coerced = _coerce_status_int(value)
            if coerced is not None:
                return coerced
        for nested in error.values():
            if isinstance(nested, dict):
                code = _extract_http_status_code(nested)
                if code is not None:
                    return code
    return None


def _coerce_status_int(value: Any) -> int | None:
    if isinstance(value, bool):
        # ``True``/``False`` would otherwise coerce to 1/0 via int().
        return None
    if isinstance(value, int):
        return value if 100 <= value < 1000 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            code = int(text)
            return code if 100 <= code < 1000 else None
    return None


def _summarize_error_payload(error: Any) -> str:
    if isinstance(error, dict):
        parts = []
        stage = error.get("stage")
        err_type = error.get("type") or error.get("error_type")
        message = _sanitize_error_text(error.get("message") or error.get("detail"))
        if stage:
            parts.append(str(stage))
        if err_type:
            parts.append(str(err_type))
        if message:
            parts.append(str(message))
        if parts:
            return ": ".join(parts)
    return _sanitize_error_text(error) or type(error).__name__


def _format_public_failure_message(record: dict[str, Any]) -> str:
    return "row {row_index}, column {column}, path {path}: {summary}".format(
        row_index=record.get("row_index"),
        column=record.get("column"),
        path=record.get("path"),
        summary=_summarize_error_payload(record.get("error")),
    )


def _coerce_source_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitize_error_text(value: Any, *, limit: int = _ERROR_MESSAGE_LIMIT) -> str | None:
    if value is None:
        return None
    text = str(value).encode("ascii", errors="ignore").decode("ascii")
    text = "".join(ch if ch.isprintable() else " " for ch in text).split()
    text = " ".join(text)
    if not text:
        return None
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _resolve_api_key(params: Any) -> Any:
    """Auto-resolve api_key from NVIDIA_API_KEY / NGC_API_KEY if not explicitly set."""
    if params is None:
        return params
    uses_no_api_key = getattr(params, "_uses_no_api_key", None)
    if callable(uses_no_api_key) and uses_no_api_key("api_key"):
        return params
    if not getattr(params, "api_key", None) and hasattr(params, "model_copy"):
        key = resolve_remote_api_key()
        if key:
            return params.model_copy(update={"api_key": key})
    return params


def _coerce(params: Any, kwargs: dict[str, Any], *, default_factory: Callable[[], Any] | None = None) -> Any:
    """Merge keyword overrides into a params object and materialize defaults when requested."""
    if params is None:
        if default_factory is None:
            return kwargs or None
        params = default_factory()
        if not kwargs:
            return params
    if not kwargs:
        return params
    if hasattr(params, "model_copy"):
        values = params.model_dump()
        uses_no_api_key = getattr(params, "_uses_no_api_key", None)
        api_key_env_reference = getattr(params, "_api_key_env_reference", None)
        for field_name in type(params).model_fields:
            if field_name in kwargs:
                continue
            if callable(uses_no_api_key) and uses_no_api_key(field_name):
                values[field_name] = NO_API_KEY
                continue
            if callable(api_key_env_reference):
                reference = api_key_env_reference(field_name)
                if reference is not None:
                    values[field_name] = reference
        return type(params).model_validate(values | kwargs)
    return params


def _coerce_audio_chunk_params(params: Any, kwargs: dict[str, Any]) -> Any:
    """Build audio chunk params; size splits default to ``DEFAULT_AUDIO_SPLIT_INTERVAL``."""
    if kwargs.get("split_type", "size") != "size":
        return _coerce(params, kwargs, default_factory=AudioChunkParams)
    return _coerce(
        params, kwargs, default_factory=lambda: AudioChunkParams(split_interval=DEFAULT_AUDIO_SPLIT_INTERVAL)
    )


class GraphIngestor(ingestor):
    """Ingestor that constructs and executes operator graphs directly.

    The fluent builder methods record pipeline stages. When :meth:`ingest` is
    called it builds a :class:`~nemo_retriever.graph.Graph` and feeds it to
    the appropriate executor.

    Parameters
    ----------
    run_mode
        ``"inprocess"`` (single-process pandas, default) or ``"batch"`` (Ray
        Data).
    ray_address
        Ray cluster address. ``None`` starts a local cluster.
    batch_size
        Default ``map_batches`` batch size for ``RayDataExecutor``.
    num_cpus
        Default CPU resources per operator node (batch mode).
    num_gpus
        Default GPU resources per operator node (batch mode).
    node_overrides
        Per-node resource/batching overrides forwarded to
        :class:`~nemo_retriever.graph.RayDataExecutor`.  Keys are node names
        (e.g. ``"OCRActor"``); values are dicts accepted by
        ``RayDataExecutor.__init__`` (``num_gpus``, ``batch_size``, etc.).
    show_progress
        Show a tqdm progress bar when running in inprocess mode.
    error_policy
        ``"raise"`` raises when explicitly configured remote NIM stages report
        row-level errors. ``"collect"`` returns partial results with the stage
        error payloads preserved.
    """

    RUN_MODE = "graph"

    def __init__(
        self,
        *,
        run_mode: str = "inprocess",
        documents: Optional[List[str]] = None,
        ray_address: Optional[str] = None,
        ray_log_to_driver: bool = True,
        debug: bool = False,
        allow_no_gpu: bool = False,
        batch_size: int = 1,
        num_cpus: float = 1,
        num_gpus: float = 0,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        show_progress: bool = True,
        error_policy: str = "raise",
    ) -> None:
        super().__init__()
        self.files(documents or [])
        if run_mode not in {"batch", "inprocess"}:
            raise ValueError(f"run_mode must be 'batch' or 'inprocess', got {run_mode!r}")
        if error_policy not in {"raise", "collect"}:
            raise ValueError(f"error_policy must be 'raise' or 'collect', got {error_policy!r}")
        self._run_mode = run_mode
        self._ray_address = ray_address
        self._ray_log_to_driver = ray_log_to_driver
        self._debug = debug
        self._allow_no_gpu = allow_no_gpu
        self._batch_size = batch_size
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._node_overrides: Dict[str, Dict[str, Any]] = node_overrides or {}
        self._show_progress = show_progress
        self._error_policy = error_policy
        self._rd_dataset: Any = None
        self._buffers: list[tuple[str, BytesIO]] = []
        self._inline_texts: list[str] | None = None

        # Pipeline configuration accumulated by fluent methods
        self._extraction_mode: str | None = None
        self._extract_params: Any = None
        self._text_params: Any = None
        self._html_params: Any = None
        self._audio_chunk_params: Any = None
        self._asr_params: Any = None
        self._video_frame_params: Any = None
        self._video_text_dedup_params: Any = None
        self._av_fuse_params: Any = None
        self._embed_params: Any = None
        self._split_config: dict[str, Any] = dict.fromkeys(SPLIT_CONFIG_VALID_KEYS, None)
        self._caption_params: Any = None
        self._dedup_params: Any = None
        self._store_params: Any = None
        self._vdb_upload_params: Any = None
        self._webhook_params: Any = None
        # Ordered list of stage names; "extract" is tracked but excluded from
        # the post-extraction stage_order passed to graph builders.
        self._stage_order: List[str] = []

    # ------------------------------------------------------------------
    # Input configuration
    # ------------------------------------------------------------------

    def files(self, documents: Union[str, List[str]]) -> "GraphIngestor":
        """Set the input file paths, glob patterns, or directories (expanded to their supported files)."""
        self._documents = expand_input_directories([documents] if isinstance(documents, str) else list(documents))
        return self

    def texts(self, texts: Union[str, Sequence[str]]) -> Self:
        """Set raw inline text documents as the graph input.

        Each string is one logical source document. It receives a deterministic
        ``inline://`` identifier and flows through the normal text splitter,
        embedding, and sink stages without being written to a temporary file.
        Inline text may be combined with file or buffer inputs; the manifest
        planner routes each source through its matching extraction branch.
        """
        self._inline_texts = normalize_inline_texts(texts)
        return self

    def buffers(
        self,
        buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]],
    ) -> "GraphIngestor":
        """Set in-memory buffers for processing.

        Each buffer is a ``(name, BytesIO)`` pair where *name* carries the
        original filename (including extension) so downstream operators can
        detect file type.  Accepts a single tuple or a list of tuples.

        Only supported for ``run_mode='inprocess'``.
        """
        if isinstance(buffers, tuple) and len(buffers) == 2 and isinstance(buffers[0], str):
            self._buffers = [buffers]
        else:
            self._buffers = list(buffers)
        return self

    # ------------------------------------------------------------------
    # Extraction stage (sets extraction_mode and primary params)
    # ------------------------------------------------------------------

    def extract(
        self,
        params: Optional[ExtractParams] = None,
        *,
        split_config: dict[str, Any] | None = None,
        extraction_mode: str | None = None,
        text_params: Optional[TextChunkParams] = None,
        html_params: Optional[HtmlChunkParams] = None,
        audio_chunk_params: Optional[AudioChunkParams] = None,
        asr_params: Optional[ASRParams] = None,
        video_frame_params: Optional[VideoFrameParams] = None,
        video_text_dedup_params: Optional[VideoFrameTextDedupParams] = None,
        av_fuse_params: Optional[AudioVisualFuseParams] = None,
        **kwargs: Any,
    ) -> "GraphIngestor":
        """Configure extraction.

        By default, the effective extraction mode is inferred from the input
        file extensions immediately before graph construction. Pass
        ``extraction_mode='pdf'`` to force the dedicated PDF/document graph, or
        ``extraction_mode='auto'`` to dispatch a mixed folder through
        :class:`MultiTypeExtractOperator`.
        Chunking is opt-in: pass ``split_config={"<key>": {...}}`` to enable
        post-extract token chunking for that source type.

        Unknown ``**kwargs`` raise :class:`TypeError`. Only fields declared
        on :class:`ExtractParams` are accepted as extra kwargs; ASR / audio
        configuration belongs on :class:`ASRParams` (pass ``asr_params=``
        or use :meth:`extract_audio`).
        """
        unknown = set(kwargs) - set(ExtractParams.model_fields)
        if unknown:
            raise TypeError(
                f"extract() got unexpected keyword argument(s) {sorted(unknown)!r}. "
                f"Allowed extra kwargs must be fields of ExtractParams. "
                f"For ASR / audio configuration, pass asr_params=ASRParams(...) "
                f"or use .extract_audio(asr_params=ASRParams(...)) "
                f"(see docs/extraction/audio-video.md)."
            )
        self._extraction_mode = extraction_mode
        self._extract_params = _resolve_api_key(_coerce(params, kwargs, default_factory=ExtractParams))
        if text_params is not None:
            self._text_params = text_params
        if html_params is not None:
            self._html_params = html_params
        if audio_chunk_params is not None:
            self._audio_chunk_params = audio_chunk_params
        if asr_params is not None:
            self._asr_params = asr_params
        if video_frame_params is not None:
            self._video_frame_params = video_frame_params
        if video_text_dedup_params is not None:
            self._video_text_dedup_params = video_text_dedup_params
        if av_fuse_params is not None:
            self._av_fuse_params = av_fuse_params
        self._apply_split_config(split_config)
        self._record_stage("extract")
        return self

    def extract_image_files(
        self,
        params: Optional[ExtractParams] = None,
        *,
        split_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "GraphIngestor":
        """Configure image extraction (extraction_mode='image')."""
        self._extraction_mode = "image"
        self._extract_params = _resolve_api_key(_coerce(params, kwargs, default_factory=ExtractParams))
        self._apply_split_config(split_config)
        self._record_stage("extract")
        return self

    def extract_html(self, params: Optional[HtmlChunkParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Configure HTML extraction (extraction_mode='html')."""
        self._extraction_mode = "html"
        self._html_params = _coerce(params, kwargs, default_factory=HtmlChunkParams)
        self._record_stage("extract")
        return self

    def extract_audio(
        self,
        params: Optional[AudioChunkParams] = None,
        *,
        asr_params: Optional[ASRParams] = None,
        split_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "GraphIngestor":
        """Configure audio extraction (extraction_mode='audio')."""
        self._extraction_mode = "audio"
        self._audio_chunk_params = _coerce_audio_chunk_params(params, kwargs)
        self._asr_params = asr_params or default_asr_params()
        self._apply_split_config(split_config)
        self._record_stage("extract")
        return self

    def extract_video(
        self,
        params: Optional[AudioChunkParams] = None,
        *,
        asr_params: Optional[ASRParams] = None,
        video_frame_params: Optional[VideoFrameParams] = None,
        video_text_dedup_params: Optional[VideoFrameTextDedupParams] = None,
        av_fuse_params: Optional[AudioVisualFuseParams] = None,
        extract_params: Optional[ExtractParams] = None,
        split_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "GraphIngestor":
        """Configure video extraction.

        Sets ``extraction_mode='auto'`` so :class:`MultiTypeExtractOperator`
        dispatches by file extension; ``.mp4``/``.mov``/``.mkv``
        files are routed to a combined audio-from-video ASR + frame OCR +
        scene fusion pipeline.

        Frame OCR config (``ocr_invoke_url``, ``ocr_api_key``,
        ``inference_batch_size``, ``ocr_request_timeout_s``) is read from
        :class:`ExtractParams` — the same object the PDF/image pipelines
        use — so the user only configures OCR once.

        The ``split_config`` keyword honors the ``"video"`` key (chunking the
        fused audio+visual transcript). The ``"audio"`` key is ignored on the
        video pipeline — for audio-only chunking, use :meth:`extract_audio`
        directly with that file.
        """
        self._extraction_mode = "auto"
        self._audio_chunk_params = _coerce_audio_chunk_params(params, kwargs)
        self._asr_params = asr_params or default_asr_params()
        self._video_frame_params = video_frame_params or VideoFrameParams(fps=DEFAULT_VIDEO_FRAME_FPS)
        self._video_text_dedup_params = video_text_dedup_params or VideoFrameTextDedupParams()
        self._av_fuse_params = av_fuse_params or AudioVisualFuseParams()
        if extract_params is not None:
            self._extract_params = _resolve_api_key(extract_params)
        elif self._extract_params is None:
            self._extract_params = ExtractParams()
        self._apply_split_config(split_config)
        self._record_stage("extract")
        return self

    # ------------------------------------------------------------------
    # Post-extraction transform stages
    # ------------------------------------------------------------------

    def dedup(self, params: Optional[DedupParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record image-deduplication settings.

        Configure both mechanisms as ``False`` to suppress the automatic
        deduplication otherwise enabled by captioning non-image documents.

        Parameters
        ----------
        params
            Optional image-deduplication parameters. When neither ``params``
            nor keyword overrides are supplied, both default passes are
            enabled.
        **kwargs
            Field overrides applied after values from ``params``.

        Returns
        -------
        GraphIngestor
            This ingestor instance for fluent chaining.

        Raises
        ------
        ValueError
            If the supplied values do not form valid deduplication parameters.
        """
        self._dedup_params = _coerce(params, kwargs, default_factory=DedupParams)
        self._record_stage("dedup")
        return self

    def caption(self, params: Optional[CaptionParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a caption stage with automatic document-image deduplication.

        Inputs resolved to image extraction are exempt. Call :meth:`dedup`
        with both mechanisms disabled to preserve every extracted image crop.

        Parameters
        ----------
        params
            Optional image-captioning parameters. When neither ``params`` nor
            keyword overrides are supplied, default parameters are used.
        **kwargs
            Field overrides applied after values from ``params``.

        Returns
        -------
        GraphIngestor
            This ingestor instance for fluent chaining.

        Raises
        ------
        ValueError
            If the supplied values do not form valid caption parameters.
        """
        self._caption_params = _resolve_api_key(_coerce(params, kwargs, default_factory=CaptionParams))
        self._record_stage("caption")
        return self

    def store(self, params: Optional[StoreParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a store stage for persisting extracted image assets to storage."""
        self._store_params = _coerce(params, kwargs, default_factory=StoreParams)
        self._record_stage("store")
        return self

    def embed(self, params: Optional[EmbedParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record an embedding stage."""
        self._embed_params = _resolve_api_key(_coerce(params, kwargs, default_factory=EmbedParams))
        self._record_stage("embed")
        return self

    def vdb_upload(self, params: Optional[VdbUploadParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a vector DB upload **sink** (in-graph after embed/store, before webhook).

        Does not call :meth:`_record_stage`: ``stage_order`` only lists
        ``dedup`` / ``caption`` / ``store`` / ``embed`` for reordering; VDB is
        always appended from ``_vdb_upload_params`` in
        :func:`~nemo_retriever.graph.ingestor_runtime._append_ordered_transform_stages`.
        Plan builders that round-trip sinks use :meth:`~nemo_retriever.ingest_plans.BaseIngestPlan.record_sink`.
        """
        self._vdb_upload_params = _coerce(params, kwargs, default_factory=VdbUploadParams)
        return self

    def webhook(self, params: Optional[WebhookParams] = None, **kwargs: Any) -> "GraphIngestor":
        """Record a webhook notification stage (always runs last).

        When ``endpoint_url`` is set, processed results are HTTP-POSTed to
        that URL.  If ``endpoint_url`` is ``None`` the stage is a no-op.
        """
        self._webhook_params = _coerce(params, kwargs, default_factory=WebhookParams)
        self._record_stage("webhook")
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def ingest(self, params: Any = None, **kwargs: Any) -> Any:
        """Build the operator graph and run it through the configured executor.

        Captioning automatically applies default image deduplication to
        non-image inputs unless an explicit dedup configuration disables both
        deduplication passes.

        Parameters
        ----------
        params
            Optional :class:`IngestExecuteParams` (or plain ``dict``) carrying
            execute-time flags. Graph run modes honor ``return_failures``.
        **kwargs
            Execute-time flags passed directly. ``return_failures`` may be
            passed here and takes precedence over the value in ``params``.
        return_failures
            When ``True`` (default ``False``), return ``(result, failures)``
            instead of raising collected row-level stage errors. If no explicit
            remote-stage diagnostics are configured, all output columns are
            scanned for populated error fields so local collected failures can
            still be returned; the default raise path remains scoped to
            explicitly configured remote stages.

        Returns
        -------
        ``run_mode='batch'`` or ``run_mode='inprocess'``
            A ``pandas.DataFrame``.
        ``return_failures=True``
            ``(result, failures)`` where ``failures`` is a list of
            service-style ``(source, error)`` tuples.
        """
        return_failures = self._resolve_return_failures(params, kwargs)
        self._validate_input_sources(self._inline_texts)
        if not self._documents and not self._buffers and is_blank_inline_corpus(self._inline_texts):
            result = empty_text_chunks_df()
            if self._run_mode == "batch":
                self._rd_dataset = result
            else:
                self._rd_dataset = None
            return self._finalize_ingest_result(result, return_failures=return_failures)

        default_branches = self._plan_default_extraction_branches()
        execute_branches = default_branches is not None and (
            len(default_branches) > 1 or self._has_mixed_inline_sources()
        )
        if default_branches is None:
            single_effective = self._resolve_effective_extraction_inputs()
        elif not execute_branches:
            single_effective = self._resolve_branch_extraction_inputs(default_branches[0])
        else:
            single_effective = None

        image_only = single_effective is not None and single_effective.extraction_mode == "image"
        effective_dedup_params = resolve_effective_dedup_params(
            self._dedup_params,
            caption_enabled=self._caption_params is not None,
            image_only=image_only,
        )
        effective_stage_order = list(self._stage_order)
        if dedup_params_enabled(effective_dedup_params) and "dedup" not in effective_stage_order:
            try:
                idx = effective_stage_order.index("caption")
            except ValueError:
                idx = len(effective_stage_order)
            effective_stage_order.insert(idx, "dedup")

        post_extract_order = tuple(s for s in effective_stage_order if s != "extract")

        if execute_branches:
            result = self._execute_extraction_branches(
                default_branches,
                dedup_params=effective_dedup_params,
                post_extract_order=post_extract_order,
            )
        else:
            if single_effective is None:
                raise RuntimeError("Internal error: extraction inputs were not resolved.")
            result = self._execute_single_graph(
                single_effective,
                dedup_params=effective_dedup_params,
                post_extract_order=post_extract_order,
            )

        return self._finalize_ingest_result(result, return_failures=return_failures)

    def _execute_single_graph(
        self,
        effective_extraction: ResolvedExtractionInputs,
        *,
        dedup_params: DedupParams | None,
        post_extract_order: tuple[str, ...],
    ) -> Any:
        if self._run_mode == "batch":
            return self._execute_single_graph_batch(
                effective_extraction,
                dedup_params=dedup_params,
                post_extract_order=post_extract_order,
            )
        return self._execute_single_graph_inprocess(
            effective_extraction,
            dedup_params=dedup_params,
            post_extract_order=post_extract_order,
        )

    def _execute_single_graph_batch(
        self,
        effective_extraction: ResolvedExtractionInputs,
        *,
        dedup_params: DedupParams | None,
        post_extract_order: tuple[str, ...],
    ) -> Any:
        ray, cluster_resources = self._ensure_batch_runtime()
        graph = build_graph(
            extraction_mode=effective_extraction.extraction_mode,
            extract_params=effective_extraction.extract_params,
            text_params=effective_extraction.text_params,
            html_params=effective_extraction.html_params,
            audio_chunk_params=effective_extraction.audio_chunk_params,
            asr_params=effective_extraction.asr_params,
            video_frame_params=effective_extraction.video_frame_params,
            video_text_dedup_params=effective_extraction.video_text_dedup_params,
            av_fuse_params=effective_extraction.av_fuse_params,
            embed_params=self._embed_params,
            split_config=self._split_config,
            caption_params=self._caption_params,
            dedup_params=dedup_params,
            store_params=self._store_params,
            vdb_upload_params=self._vdb_upload_params,
            webhook_params=self._webhook_params,
            stage_order=post_extract_order,
        )
        effective_allow_no_gpu = self._allow_no_gpu or cluster_resources.total_gpu_count() == 0
        derived_overrides = batch_tuning_to_node_overrides(
            effective_extraction.extract_params,
            self._embed_params,
            store_params=self._store_params,
            cluster_resources=cluster_resources,
            allow_no_gpu=effective_allow_no_gpu,
            caption_params=self._caption_params,
            video_frame_params=effective_extraction.video_frame_params,
            extraction_mode=effective_extraction.extraction_mode,
        )
        executor = RayDataExecutor(
            graph,
            ray_address=self._ray_address,
            batch_size=self._batch_size,
            num_cpus=self._num_cpus,
            num_gpus=self._num_gpus,
            node_overrides=merge_node_overrides(derived_overrides, self._node_overrides),
            auto_concurrency_nodes=(
                default_concurrency_node_names(
                    effective_extraction.extract_params,
                    self._embed_params,
                    self._store_params,
                    self._caption_params,
                )
                - {name for name, override in self._node_overrides.items() if "concurrency" in override}
            ),
        )
        executor_input = self._inline_text_dataset(ray.data) if self._inline_texts else self._documents
        result = executor.ingest(executor_input)
        self._rd_dataset = result
        return result

    def _execute_single_graph_inprocess(
        self,
        effective_extraction: ResolvedExtractionInputs,
        *,
        dedup_params: DedupParams | None,
        post_extract_order: tuple[str, ...],
    ) -> Any:
        graph = build_graph(
            extraction_mode=effective_extraction.extraction_mode,
            extract_params=effective_extraction.extract_params,
            text_params=effective_extraction.text_params,
            html_params=effective_extraction.html_params,
            audio_chunk_params=effective_extraction.audio_chunk_params,
            asr_params=effective_extraction.asr_params,
            video_frame_params=effective_extraction.video_frame_params,
            video_text_dedup_params=effective_extraction.video_text_dedup_params,
            av_fuse_params=effective_extraction.av_fuse_params,
            embed_params=self._embed_params,
            split_config=self._split_config,
            caption_params=self._caption_params,
            dedup_params=dedup_params,
            store_params=self._store_params,
            vdb_upload_params=self._vdb_upload_params,
            webhook_params=self._webhook_params,
            stage_order=post_extract_order,
        )
        executor = InprocessExecutor(graph, show_progress=self._show_progress)
        self._rd_dataset = None
        if self._inline_texts:
            return executor.ingest(self._inline_text_dataframe())
        if self._buffers:
            import pandas as pd

            df = pd.DataFrame([{"bytes": buf.getvalue(), "path": name} for name, buf in self._buffers])
            return executor.ingest(df)
        return executor.ingest(self._documents)

    def _execute_extraction_branches(
        self,
        branches: tuple[ExtractionBranchPlan, ...],
        *,
        dedup_params: DedupParams | None,
        post_extract_order: tuple[str, ...],
    ) -> Any:
        result = ExtractionBranchExecutor(
            run_mode=self._run_mode,
            branches=branches,
            documents=self._documents,
            buffers=self._buffers,
            inline_rows=self._inline_text_rows(),
            split_config=self._split_config,
            extract_params=self._extract_params,
            text_params=self._text_params,
            html_params=self._html_params,
            audio_chunk_params=self._audio_chunk_params,
            asr_params=self._asr_params,
            video_frame_params=self._video_frame_params,
            video_text_dedup_params=self._video_text_dedup_params,
            av_fuse_params=self._av_fuse_params,
            embed_params=self._embed_params,
            caption_params=self._caption_params,
            dedup_params=dedup_params,
            store_params=self._store_params,
            vdb_upload_params=self._vdb_upload_params,
            webhook_params=self._webhook_params,
            post_extract_order=post_extract_order,
            ray_address=self._ray_address,
            batch_size=self._batch_size,
            num_cpus=self._num_cpus,
            num_gpus=self._num_gpus,
            node_overrides=self._node_overrides,
            show_progress=self._show_progress,
            allow_no_gpu=self._allow_no_gpu,
            ensure_batch_runtime=self._ensure_batch_runtime,
        ).execute()
        self._rd_dataset = result if self._run_mode == "batch" else None
        return result

    def _ensure_batch_runtime(self) -> tuple[Any, Any]:
        ray = ensure_local_ray_runtime(self._ray_address, log_to_driver=self._ray_log_to_driver)
        return ray, gather_cluster_resources(ray)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_mixed_inline_sources(self) -> bool:
        return bool(self._inline_texts) and bool(self._documents or self._buffers)

    def _inline_text_rows(self) -> list[dict[str, str]]:
        return [
            {"text": text, "path": inline_text_source_id(index)} for index, text in enumerate(self._inline_texts or [])
        ]

    def _inline_text_dataframe(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self._inline_text_rows(), columns=["text", "path"])

    def _inline_text_dataset(self, ray_data: Any) -> Any:
        return ray_data.from_items(self._inline_text_rows())

    def _configured_input_paths(self) -> list[str]:
        paths: list[str] = []
        for document in self._documents:
            try:
                paths.extend(expand_input_file_patterns([document]))
            except FileNotFoundError:
                paths.append(os.fspath(document))
        paths.extend(name for name, _ in self._buffers)
        paths.extend(inline_text_source_id(index) for index, _ in enumerate(self._inline_texts or []))
        return paths

    def _classified_input_paths(self) -> list[tuple[str, str | None]]:
        # Service workers receive inline text as a named byte buffer. Keep the
        # logical URI as its source path while classifying it as decoded text.
        return [
            (path, "txt" if is_inline_text_source(path) else input_type_for_path(path))
            for path in self._configured_input_paths()
        ]

    @staticmethod
    def _input_type_examples(paths: Iterable[str], *, limit: int = 3) -> str:
        examples = list(paths)[:limit]
        return ", ".join(examples)

    def _validate_explicit_extraction_mode_inputs(
        self,
        extraction_mode: str,
        classified: list[tuple[str, str | None]],
    ) -> None:
        allowed_types = _EXPLICIT_MODE_INPUT_TYPES.get(extraction_mode)
        if allowed_types is None:
            return

        mismatched = [
            path
            for path, input_type in classified
            if not _is_explicit_glob_path(path) and (input_type is None or input_type not in allowed_types)
        ]
        if mismatched:
            examples = self._input_type_examples(mismatched)
            raise ValueError(f"Input file type(s) do not match extraction_mode={extraction_mode!r}: {examples}")

    def _plan_default_extraction_branches(self) -> tuple[ExtractionBranchPlan, ...] | None:
        if self._extraction_mode is not None and not self._has_mixed_inline_sources():
            return None
        manifest = build_input_manifest(self._configured_input_paths())
        branches = plan_extraction_branches(manifest)
        if self._debug:
            logger.info(
                "Retriever ingest manifest planned %d extraction branches: %s",
                len(branches),
                format_branch_summary(branches),
            )
        return branches

    def _resolve_branch_extraction_inputs(self, branch: ExtractionBranchPlan) -> ResolvedExtractionInputs:
        return resolve_branch_extraction_inputs(
            branch,
            extract_params=self._extract_params,
            text_params=self._text_params,
            html_params=self._html_params,
            audio_chunk_params=self._audio_chunk_params,
            asr_params=self._asr_params,
            video_frame_params=self._video_frame_params,
            video_text_dedup_params=self._video_text_dedup_params,
            av_fuse_params=self._av_fuse_params,
        )

    def _resolve_effective_extraction_inputs(self) -> ResolvedExtractionInputs:
        extraction_mode = self._extraction_mode
        classified = self._classified_input_paths()
        if extraction_mode is not None:
            self._validate_explicit_extraction_mode_inputs(extraction_mode, classified)
            text_params = self._text_params
            html_params = self._html_params
            if extraction_mode == "auto":
                observed_input_types = {input_type for _, input_type in classified if input_type is not None}
                if "txt" in observed_input_types:
                    text_params = text_params or TextChunkParams()
                if "html" in observed_input_types:
                    html_params = html_params or HtmlChunkParams()
            return ResolvedExtractionInputs(
                extraction_mode=extraction_mode,
                extract_params=self._extract_params,
                text_params=text_params,
                html_params=html_params,
                audio_chunk_params=self._audio_chunk_params,
                asr_params=self._asr_params,
                video_frame_params=self._video_frame_params,
                video_text_dedup_params=self._video_text_dedup_params,
                av_fuse_params=self._av_fuse_params,
            )

        branches = self._plan_default_extraction_branches()
        if branches is None:
            raise RuntimeError("Internal error: default extraction planning did not return branches.")
        if len(branches) == 1:
            return self._resolve_branch_extraction_inputs(branches[0])

        # Compatibility fallback for private callers that still ask for a
        # scalar effective mode directly. The public ingest path executes the
        # branches instead of using this MultiType fallback.
        return ResolvedExtractionInputs(
            extraction_mode="auto",
            extract_params=self._extract_params or ExtractParams(),
            text_params=self._text_params or TextChunkParams(),
            html_params=self._html_params or HtmlChunkParams(),
            audio_chunk_params=self._audio_chunk_params,
            asr_params=self._asr_params,
            video_frame_params=self._video_frame_params,
            video_text_dedup_params=self._video_text_dedup_params,
            av_fuse_params=self._av_fuse_params,
        )

    @staticmethod
    def _is_populated_error_field(key: str, value: Any) -> bool:
        return is_populated_error_field(key, value)

    @classmethod
    def _iter_stage_errors_from_value(cls, value: Any, *, path: str = "") -> Iterator[dict[str, Any]]:
        yield from iter_stage_errors_from_value(value, path=path)

    @staticmethod
    def _row_value(row: Any, key: str) -> Any:
        if isinstance(row, dict):
            return row.get(key)
        getter = getattr(row, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except Exception as exc:  # noqa: BLE001 - row metadata lookup is best-effort diagnostic context.
                logger.debug(
                    "Failed to read source identifier field %r from row type %s: %s",
                    key,
                    type(row).__name__,
                    exc,
                    exc_info=True,
                )
                return None
        return None

    @staticmethod
    def _nested_mapping_value(value: Any, path: tuple[str, ...]) -> Any:
        current = value
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @classmethod
    def _source_identifier_from_row(cls, row: Any, row_index: Any) -> str:
        for field in ("document_id", "path", "source_path"):
            identifier = _coerce_source_identifier(cls._row_value(row, field))
            if identifier is not None:
                return identifier

        metadata = cls._row_value(row, "metadata")
        for nested_path in (
            ("source_path",),
            ("source_metadata", "source_id"),
            ("source_metadata", "source_name"),
        ):
            identifier = _coerce_source_identifier(cls._nested_mapping_value(metadata, nested_path))
            if identifier is not None:
                return identifier

        for field in ("source_id", "source_name"):
            identifier = _coerce_source_identifier(cls._row_value(row, field))
            if identifier is not None:
                return identifier

        return f"row {row_index}" if row_index is not None else "row ?"

    @staticmethod
    def _public_failure_tuple(record: dict[str, Any]) -> tuple[str, str]:
        identifier = _coerce_source_identifier(record.get("source_identifier"))
        if identifier is None:
            row_index = record.get("row_index")
            identifier = f"row {row_index}" if row_index is not None else "row ?"
        return identifier, _format_public_failure_message(record)

    @classmethod
    def _stage_error_records(cls, batch: Any, *, columns: Iterable[str] | None = None) -> list[dict[str, Any]]:
        iter_batches = getattr(batch, "iter_batches", None)
        if getattr(batch, "columns", None) is None and not callable(iter_batches):
            return []
        requested_columns = list(columns) if columns is not None else None

        batches = iter_batches(batch_format="pyarrow") if callable(iter_batches) else (batch,)

        records: list[dict[str, Any]] = []
        for raw_batch in batches:
            available_columns = (
                getattr(raw_batch, "column_names", None)
                if callable(iter_batches)
                else getattr(raw_batch, "columns", None)
            )
            if available_columns is None:
                continue
            target_columns = (
                list(available_columns)
                if requested_columns is None
                else [c for c in requested_columns if c in available_columns]
            )
            if not target_columns:
                continue
            # Finalization only needs diagnostic payloads and source context.
            # Reading the full frame can iterate unrelated Arrow-backed image
            # columns that pandas cannot safely materialize row by row.
            scan_columns = list(
                dict.fromkeys(
                    [*target_columns, *(column for column in _SOURCE_IDENTIFIER_COLUMNS if column in available_columns)]
                )
            )
            batch_df = (
                arrow_table_to_pandas(raw_batch.select(scan_columns))
                if callable(iter_batches)
                else raw_batch.loc[:, scan_columns]
            )
            column_values = {column: batch_df[column].array for column in scan_columns}
            for row_position, row_index in enumerate(batch_df.index):
                row = {column: values[row_position] for column, values in column_values.items()}
                source_identifier = cls._source_identifier_from_row(row, row_index)
                for column in target_columns:
                    for record in cls._iter_stage_errors_from_value(row[column]):
                        records.append(
                            {
                                "row_index": row_index,
                                "source_identifier": source_identifier,
                                "column": column,
                                **record,
                            }
                        )
        return records

    @staticmethod
    def _has_error(v: Any) -> bool:
        return any(GraphIngestor._iter_stage_errors_from_value(v))

    @staticmethod
    def _param_value(params: Any, field: str) -> Any:
        if params is None:
            return None
        if isinstance(params, dict):
            return params.get(field)
        return getattr(params, field, None)

    @classmethod
    def _is_configured(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return any(cls._is_configured(v) for v in value)
        return bool(value)

    @classmethod
    def _params_has_configured_field(cls, params: Any, fields: tuple[str, ...]) -> bool:
        return any(cls._is_configured(cls._param_value(params, field)) for field in fields)

    def _remote_stage_error_columns(self) -> set[str]:
        """Backwards-compatible thin shim over :meth:`_remote_stage_diagnostics`.

        Older callers (and existing tests) consume the set of columns
        the strict-error-policy will gate on. The richer
        :meth:`_remote_stage_diagnostics` mapping carries the same set
        of keys plus per-stage NIM URL / display-name diagnostics that
        :class:`GraphIngestionError` uses to format actionable messages.
        """
        return set(self._remote_stage_diagnostics().keys())

    def _remote_stage_diagnostics(self) -> dict[str, _StageDiagnostic]:
        """Build a column → :class:`_StageDiagnostic` map for remote-NIM stages.

        Only stages that have an explicitly configured invoke URL appear
        here — the ``"raise"`` error policy is scoped to remote endpoints
        the operator opted into. The map's keys are the dataframe column
        names emitted by each stage; the values carry the resolved
        display name and URL so :class:`GraphIngestionError` can render
        ``stage=… url=…`` per row and a ``Troubleshooting:`` footer.
        """
        diagnostics: dict[str, _StageDiagnostic] = {}

        extract = self._extract_params
        if self._params_has_configured_field(extract, ("page_elements_invoke_url",)):
            column = self._param_value(extract, "output_column") or _DEFAULT_PAGE_ELEMENTS_COLUMN
            diagnostics[column] = _StageDiagnostic(
                column=column,
                display_name="Page Elements NIM",
                invoke_url=self._param_value(extract, "page_elements_invoke_url"),
                role="page_elements",
            )
        if self._params_has_configured_field(extract, ("ocr_invoke_url",)):
            diagnostics["ocr"] = _StageDiagnostic(
                column="ocr",
                display_name="OCR NIM",
                invoke_url=self._param_value(extract, "ocr_invoke_url"),
                role="ocr",
            )
        if self._params_has_configured_field(extract, ("table_structure_invoke_url",)):
            diagnostics["table_structure_ocr_v1"] = _StageDiagnostic(
                column="table_structure_ocr_v1",
                display_name="Table Structure NIM",
                invoke_url=self._param_value(extract, "table_structure_invoke_url"),
                role="table_structure",
            )
        if self._params_has_configured_field(extract, ("invoke_url", "nemotron_parse_invoke_url")):
            url = self._param_value(extract, "nemotron_parse_invoke_url") or self._param_value(extract, "invoke_url")
            diagnostics["nemotron_parse_v1_2"] = _StageDiagnostic(
                column="nemotron_parse_v1_2",
                display_name="Nemotron Parse NIM",
                invoke_url=url,
                role="nemotron_parse",
            )
        if self._params_has_configured_field(self._embed_params, _REMOTE_EMBED_ENDPOINT_FIELDS):
            column = self._param_value(self._embed_params, "output_column") or _DEFAULT_EMBED_COLUMN
            url = self._param_value(self._embed_params, "embed_invoke_url") or self._param_value(
                self._embed_params, "embedding_endpoint"
            )
            diagnostics[column] = _StageDiagnostic(
                column=column,
                display_name="Embedding NIM",
                invoke_url=url,
                model_name=self._param_value(self._embed_params, "model_name"),
                role="embed",
            )
        return diagnostics

    def _raise_for_stage_errors(self, result: Any) -> None:
        if self._error_policy == "collect":
            return
        diagnostics = self._remote_stage_diagnostics()
        if not diagnostics:
            return
        records = self._stage_error_records(result, columns=set(diagnostics.keys()))
        if records:
            raise GraphIngestionError(records, stage_diagnostics=diagnostics)

    @staticmethod
    def _resolve_return_failures(params: Any, kwargs: dict[str, Any]) -> bool:
        if "return_failures" in kwargs:
            return bool(kwargs["return_failures"])
        if isinstance(params, IngestExecuteParams):
            return bool(params.return_failures)
        if isinstance(params, dict) and "return_failures" in params:
            return bool(params["return_failures"])
        return False

    def _collect_failure_records(self, result: Any) -> list[dict[str, Any]]:
        diagnostics = self._remote_stage_diagnostics()
        # With explicit remote stages, report only their diagnostic columns.
        # Without them, scan all columns so ``return_failures=True`` can expose
        # local collected failures instead of silently returning an empty list.
        columns = set(diagnostics.keys()) if diagnostics else None
        return self._stage_error_records(result, columns=columns)

    def _collect_failure_tuples(self, result: Any) -> list[tuple[str, str]]:
        return [self._public_failure_tuple(record) for record in self._collect_failure_records(result)]

    def _finalize_ingest_result(self, result: Any, *, return_failures: bool) -> Any:
        if return_failures:
            return result, self._collect_failure_tuples(result)
        self._raise_for_stage_errors(result)
        return result

    @staticmethod
    def extract_error_rows(batch: Any) -> Any:
        if batch is None:
            return batch
        columns = getattr(batch, "columns", None)
        if columns is None:
            return batch
        if len(columns) == 0:
            return batch.iloc[0:0]

        mask = batch[columns[0]].apply(GraphIngestor._has_error).astype(bool)
        for c in columns[1:]:
            mask = mask | batch[c].apply(GraphIngestor._has_error).astype(bool)
        return batch[mask]

    def get_error_rows(self, dataset: Any = None) -> Any:
        import pandas as pd

        target = dataset if dataset is not None else self._rd_dataset
        if target is None:
            raise RuntimeError("No Ray Dataset available to inspect for errors.")
        if isinstance(target, pd.DataFrame):
            return self.extract_error_rows(target)
        return target.map_batches(
            call_pandas_function_on_arrow,
            batch_format="pyarrow",
            fn_kwargs={"fn": self.extract_error_rows},
        )

    def get_dataset(self) -> Any:
        return self._rd_dataset

    def _record_stage(self, name: str) -> None:
        """Append *name* to the stage order list (deduplicated in place)."""
        if name not in self._stage_order:
            self._stage_order.append(name)

    def _apply_split_config(self, split_config: dict[str, Any] | None) -> None:
        """Resolve an explicitly supplied split configuration."""
        if split_config is not None:
            self._split_config = resolve_split_params(split_config)
