# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Batch runmode.

Intended for large-scale batch execution over large inputs on multiple workers.
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import timedelta
from functools import partial

from typing import Union

import ray
import ray.data as rd
from nemo_retriever.utils.convert import DocToPdfConversionActor
from nemo_retriever.chart.chart_detection import GraphicElementsActor
from nemo_retriever.page_elements import PageElementDetectionActor
from nemo_retriever.ocr.ocr import NemotronParseActor, OCRActor
from nemo_retriever.table.table_detection import TableStructureActor
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.utils.hf_cache import resolve_hf_cache_dir
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.utils.ray_resource_hueristics import (
    gather_cluster_resources,
    resolve_requested_plan,
)
from nemo_retriever.ingest_modes.inprocess import collapse_content_to_page_rows, explode_content_to_rows

from ..image.load import SUPPORTED_IMAGE_EXTENSIONS
from ..ingestor import Ingestor
from ..params import ASRParams
from ..params import AudioChunkParams
from ..params import EmbedParams
from ..params import ExtractParams
from ..params import HtmlChunkParams
from ..params import IngestExecuteParams
from ..params import PdfSplitParams
from ..params import TextChunkParams
from ..params import VdbUploadParams

logger = logging.getLogger(__name__)


def _setup_batch_debug_logger(enabled: bool) -> logging.Logger:
    """Return a batch logger configured for optional DEBUG verbosity."""
    logger = logging.getLogger("nemo_retriever.ingest_modes.batch")
    logger.propagate = True
    if enabled:
        logger.setLevel(logging.DEBUG)
        logger.debug("Batch debug logging enabled.")
    return logger


def _debug_log(*, logger: logging.Logger, location: str, message: str, data: dict[str, Any]) -> None:
    """Emit structured debug payloads without interrupting pipeline execution."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        logger.debug("%s | %s | %s", location, message, json.dumps(data, default=str))
    except Exception:
        logger.debug("%s | %s | %r", location, message, data)


from nemo_retriever.params.utils import coerce_params as _coerce_params


def _runtime_env_vars() -> dict[str, str]:
    env_vars = {
        "NEMO_RETRIEVER_HF_CACHE_DIR": resolve_hf_cache_dir(),
        "LOG_LEVEL": "INFO",
    }
    return {key: value for key, value in env_vars.items() if isinstance(value, str)}


def _normalize_requested_plan_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _normalize_requested_plan_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0.0 else None


def _batch_tuning_to_requested_plan_overrides(batch_tuning: dict[str, Any]) -> dict[str, int | float | None]:
    overrides: dict[str, int | float | None] = {}

    if "pdf_extract_workers" in batch_tuning:
        overrides["override_pdf_extract_tasks"] = _normalize_requested_plan_int(batch_tuning.get("pdf_extract_workers"))
    if "pdf_extract_num_cpus" in batch_tuning:
        overrides["override_pdf_extract_cpus_per_task"] = _normalize_requested_plan_float(
            batch_tuning.get("pdf_extract_num_cpus")
        )
    if "pdf_extract_batch_size" in batch_tuning:
        overrides["override_pdf_extract_batch_size"] = _normalize_requested_plan_int(
            batch_tuning.get("pdf_extract_batch_size")
        )
    if "page_elements_batch_size" in batch_tuning:
        overrides["override_page_elements_batch_size"] = _normalize_requested_plan_int(
            batch_tuning.get("page_elements_batch_size")
        )
    if "embed_batch_size" in batch_tuning:
        overrides["override_embed_batch_size"] = _normalize_requested_plan_int(batch_tuning.get("embed_batch_size"))

    ocr_batch_size = batch_tuning.get("ocr_batch_size", batch_tuning.get("detect_batch_size"))
    if "ocr_batch_size" in batch_tuning or "detect_batch_size" in batch_tuning:
        overrides["override_ocr_batch_size"] = _normalize_requested_plan_int(ocr_batch_size)

    actor_count_fields = {
        "page_elements_workers": "page_elements",
        "ocr_workers": "ocr",
        "detect_workers": "ocr",
        "embed_workers": "embed",
    }
    for field_name, stage_name in actor_count_fields.items():
        if field_name not in batch_tuning:
            continue
        actor_count = _normalize_requested_plan_int(batch_tuning.get(field_name))
        overrides[f"override_{stage_name}_initial_actors"] = actor_count
        overrides[f"override_{stage_name}_min_actors"] = actor_count
        overrides[f"override_{stage_name}_max_actors"] = actor_count

    gpu_fields = {
        "gpu_page_elements": "page_elements",
        "gpu_ocr": "ocr",
        "gpu_embed": "embed",
    }
    for field_name, stage_name in gpu_fields.items():
        if field_name not in batch_tuning:
            continue
        overrides[f"override_{stage_name}_gpus_per_actor"] = _normalize_requested_plan_float(
            batch_tuning.get(field_name)
        )

    return overrides


class _LanceDBWriteActor:
    """Ray Data actor that streams batches into LanceDB as they arrive.

    Creates the table on the first batch, then appends subsequent batches.
    Index creation is intentionally deferred until after the full pipeline
    has been consumed (handled by ``BatchIngestor.ingest()``).
    """

    def __init__(self, params: VdbUploadParams | None = None) -> None:
        from nemo_retriever.ingest_modes.lancedb_utils import lancedb_schema

        lancedb_params = (params or VdbUploadParams()).lancedb

        self._lancedb_uri = lancedb_params.lancedb_uri
        self._table_name = lancedb_params.table_name
        self._overwrite = lancedb_params.overwrite
        self._embedding_column = lancedb_params.embedding_column
        self._embedding_key = lancedb_params.embedding_key
        self._include_text = lancedb_params.include_text
        self._text_column = lancedb_params.text_column

        import lancedb  # type: ignore

        self._db = lancedb.connect(uri=self._lancedb_uri)
        self._total_rows = 0

        # Use a default dim for the initial empty table; rows are appended via add().
        self._schema = lancedb_schema(2048)
        mode = "overwrite" if self._overwrite else "create"
        self._table = self._db.create_table(
            self._table_name,
            schema=self._schema,
            mode=mode,
        )

    def _build_rows(self, df: Any) -> list:
        """Build LanceDB rows from a pandas DataFrame batch."""
        from nemo_retriever.ingest_modes.lancedb_utils import build_lancedb_rows

        return build_lancedb_rows(
            df,
            embedding_column=self._embedding_column,
            embedding_key=self._embedding_key,
            text_column=self._text_column,
            include_text=self._include_text,
        )

    def __call__(self, batch_df: Any) -> Any:
        rows = self._build_rows(batch_df)
        if rows:
            # Infer schema from first batch
            if self._table is None:
                self._table = self._db.open_table(self._table_name)
            self._table.add(rows)

            self._total_rows += len(rows)

        return batch_df


class _BatchEmbedActor:
    """Ray Data actor that holds a local text embedder on a single GPU.

    When ``embedding_endpoint`` is provided in kwargs, the actor skips local
    model creation and delegates to a remote NIM endpoint instead.
    """

    def __init__(self, params: EmbedParams) -> None:
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=r".*`input_embeds` is deprecated.*create_bidirectional_mask.*",
            category=FutureWarning,
        )

        self._params = params
        self._kwargs = {
            **params.model_dump(mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True),
            **params.runtime.model_dump(mode="python", exclude_none=True),
        }
        if "embedding_endpoint" not in self._kwargs and self._kwargs.get("embed_invoke_url"):
            self._kwargs["embedding_endpoint"] = self._kwargs.get("embed_invoke_url")

        # If a remote NIM endpoint is configured, skip local model creation.
        endpoint = (self._kwargs.get("embedding_endpoint") or self._kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            self._model = None
            return

        from nemo_retriever.model import create_local_embedder

        self._model = create_local_embedder(
            self._kwargs.get("model_name"),
            device=str(self._kwargs["device"]) if self._kwargs.get("device") else None,
            hf_cache_dir=str(self._kwargs["hf_cache_dir"]) if self._kwargs.get("hf_cache_dir") else None,
            normalize=bool(self._kwargs.get("normalize", True)),
            max_length=int(self._kwargs.get("max_length", 8192)),
        )

    def __call__(self, batch_df: Any) -> Any:
        from nemo_retriever.ingest_modes.inprocess import embed_text_main_text_embed

        return embed_text_main_text_embed(batch_df, model=self._model, **self._kwargs)


class BatchIngestor(Ingestor):
    RUN_MODE = "batch"

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        ray_address: Optional[str] = None,
        ray_log_to_driver: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(documents=documents)

        self._logger = _setup_batch_debug_logger(bool(debug))
        self._debug = bool(debug)
        if self._debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize Ray for distributed execution.
        ray.init(
            address=ray_address or "local",
            ignore_reinit_error=True,
            log_to_driver=bool(ray_log_to_driver),
            runtime_env={"env_vars": _runtime_env_vars()},
        )

        # Use the new Rich progress UI instead of verbose tqdm bars.
        ctx = rd.DataContext.get_current()
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False

        # Scheduling flow:
        # 1. Gather local/cluster resources (batch mode always runs on Ray).
        # 1a. available_resources(ray) -> ClusterResources(BaseModel).
        # 2. Resolve requested resources from user overrides or heuristics defaults.
        # 2a. -> requested_resources = resolve_requested_resources() -> RequestedResources(BaseModel)
        # 3. Compute final values from available + requested resources.
        # 3a. -> final_resources = compute_final_resources(
        #         available_resources, requested_resources
        #       ) -> FinalResources(BaseModel)
        # 4. Probe per-node GPU details (e.g. memory) via remote calls when needed.

        # 1. Gather available resources
        self._cluster_resources = gather_cluster_resources(ray)  # Contains both total and available resources
        self._total_cpu_count = self._cluster_resources.total_cpu_count()
        self._total_gpu_count = self._cluster_resources.total_gpu_count()
        self._available_cpu_count = self._cluster_resources.available_cpu_count()
        self._available_gpu_count = self._cluster_resources.available_gpu_count()
        logger.info(self._cluster_resources)

        # 2. Resolve requested plan for the Ray DAG that will be built
        self._requested_plan = resolve_requested_plan(cluster_resources=self._cluster_resources)
        logger.info(self._requested_plan)
        self._requested_plan_overrides: dict[str, int | float | None] = {}

        # Builder-style task configuration recorded for later execution.
        # Keep backwards-compatibility with code that inspects `Ingestor._documents`
        # (older examples/tests) by ensuring both names refer to the same list.
        self._input_documents: List[str] = self._documents  # List of original input documents.
        self._rd_dataset: rd.Dataset = None  # Ray Data dataset created from input documents.
        self._tasks: List[tuple[str, dict[str, Any]]] = []
        self._intermediate_output_dir: Optional[str] = None
        self._pipeline_type: str = "pdf"  # "pdf" | "txt" | "html"
        self._extract_txt_kwargs: Dict[str, Any] = {}  # noqa: F821
        self._extract_html_kwargs: Dict[str, Any] = {}  # noqa: F821
        self._use_nemotron_parse_only: bool = False

    def _refresh_requested_plan(self, batch_tuning: dict[str, Any]) -> None:
        requested_plan_overrides = _batch_tuning_to_requested_plan_overrides(batch_tuning)
        if not requested_plan_overrides:
            return

        self._requested_plan_overrides.update(requested_plan_overrides)
        self._requested_plan = resolve_requested_plan(
            cluster_resources=self._cluster_resources,
            **self._requested_plan_overrides,
        )
        logger.info(self._requested_plan)

    def files(self, documents: Union[str, List[str]]) -> "BatchIngestor":
        """
        Add local files for batch processing.

        This runmode assumes all inputs are local. Any glob pattern or explicit
        path must resolve to at least one existing file.
        """
        if isinstance(documents, str):
            documents = [documents]

        for pattern in documents:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"Invalid document pattern: {pattern!r}")

            # Expand globs (supports ** when recursive=True).
            matches = glob.glob(pattern, recursive=True)
            if matches:
                files = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not files:
                    raise FileNotFoundError(f"Pattern resolved, but no files found: {pattern!r}")
                self._input_documents.extend(files)
                continue

            # No glob matches: treat as explicit path.
            if os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
                continue

            raise FileNotFoundError(f"No local files found for: {pattern!r}")

        self._rd_dataset = rd.read_binary_files(self._input_documents, include_paths=True)

        return self

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Configure extraction for batch processing (builder only).

        This does not run extraction yet; it records configuration so the batch
        executor can build a concrete pipeline later.

        If all input files have a ``.txt`` extension, the pipeline automatically
        delegates to :meth:`extract_txt` with default :class:`TextChunkParams`.

        Resource-tuning kwargs (auto-detected from available resources if omitted):

        - ``pdf_split_batch_size``: Batch size for PDF split stage (default 1).
        - ``pdf_extract_batch_size``: Batch size for PDF extraction (default 32).
        - ``pdf_extract_workers``: TaskPool size for extraction (default num_cpus // 8).
        - ``page_elements_batch_size``: Batch size for page-element detection (default 16).
        - ``page_elements_workers``: ActorPool size for page elements (default num_gpus // 2).
        - ``detect_batch_size``: Batch size for detection stages (default 16).
        - ``detect_workers``: ActorPool size for detection stages (default num_gpus // 4).
        - ``page_elements_cpus_per_actor``: CPUs reserved per page-elements actor (default 1).
        - ``ocr_cpus_per_actor``: CPUs reserved per OCR actor (default 1).
        """

        if self._input_documents and all(f.lower().endswith(".txt") for f in self._input_documents):
            txt_params = TextChunkParams(
                max_tokens=kwargs.pop("max_tokens", 1024),
                overlap_tokens=kwargs.pop("overlap_tokens", 0),
            )
            return self.extract_txt(params=txt_params)

        if self._input_documents and all(
            os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS for f in self._input_documents
        ):
            return self.extract_image_files(params=params, **kwargs)

        resolved = _coerce_params(params, ExtractParams, kwargs)
        if (
            any(
                (
                    resolved.invoke_url,
                    resolved.page_elements_invoke_url,
                    resolved.ocr_invoke_url,
                    resolved.graphic_elements_invoke_url,
                    resolved.table_structure_invoke_url,
                )
            )
            and not resolved.api_key
        ):
            resolved = resolved.model_copy(update={"api_key": resolve_remote_api_key()})

        kwargs = {
            **resolved.model_dump(mode="python", exclude={"remote_retry", "batch_tuning"}, exclude_none=True),
            **resolved.remote_retry.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
        }
        self._refresh_requested_plan(kwargs)

        # -- Pop resource-tuning kwargs before forwarding to actors --
        def _endpoint_count(raw: Any) -> int:
            s = str(raw or "").strip()
            if not s:
                return 0
            return len([p for p in s.split(",") if p.strip()])

        # Downstream batch stages assume `page_image.image_b64` exists for every page.
        # Ensure PDF extraction emits a page image unless the caller explicitly disables it.
        kwargs.setdefault("extract_page_as_image", True)

        # 200 DPI is sufficient for both detection and OCR.  YOLOX resizes to
        # 1024x1024 internally, and NemotronOCR also resizes crops to 1024x1024,
        # nv-ingest NIM uses 300 DPI for page-element detection; match that
        # default here so local-model recall matches the container path.
        kwargs.setdefault("dpi", 300)
        kwargs.setdefault("image_format", "jpeg")
        kwargs.setdefault("jpeg_quality", 100)
        self._pipeline_type = "pdf"
        self._tasks.append(("extract", dict(kwargs)))

        # Convert DOCX/PPTX to PDF before splitting.  CPU-only, one
        # LibreOffice process per file (batch_size=1).
        self._rd_dataset = self._rd_dataset.map_batches(
            DocToPdfConversionActor,
            batch_size=1,
            num_cpus=1,
            batch_format="pandas",
        )

        # PDF SPLIT - Splits each PDF document into individual pages.
        # To help amortize downstream processing if PDFs have vastly different numbers of pages.
        # This is a CPU-only stage. This "Actor" is technically scheduled as a Task
        self._rd_dataset = self._rd_dataset.map_batches(
            PDFSplitActor(
                split_params=PdfSplitParams(
                    start_page=kwargs.get("start_page"),
                    end_page=kwargs.get("end_page"),
                )
            ),
            batch_size=1,
            num_cpus=1,
            batch_format="pandas",
        )

        # PDF EXTRACTION - Extracts text, tables, charts, infographics, etc. from the PDF pages.
        # This is a CPU-only stage and is the main CPU bottleneck of the entire DAG
        self._rd_dataset = self._rd_dataset.map_batches(
            PDFExtractionActor(**kwargs),
            batch_size=self._requested_plan.get_pdf_extract_batch_size(),
            batch_format="pandas",
            num_cpus=self._requested_plan.get_pdf_extract_cpus_per_task(),
            compute=rd.TaskPoolStrategy(size=self._requested_plan.get_pdf_extract_tasks()),
        )

        self._apply_nemotron_parse_overrides(kwargs)

        self._append_detection_stages(kwargs)

        return self

    def _apply_nemotron_parse_overrides(self, kwargs: dict[str, Any]) -> None:
        """Update ``_requested_plan`` with user-provided Nemotron Parse resource overrides
        and set ``_use_nemotron_parse_only``."""
        nemotron_parse_workers = float(kwargs.get("nemotron_parse_workers", 0.0) or 0.0)
        gpu_nemotron_parse = float(kwargs.get("gpu_nemotron_parse", 0.0) or 0.0)
        nemotron_parse_batch_size = float(kwargs.get("nemotron_parse_batch_size", 0.0) or 0.0)
        self._use_nemotron_parse_only = kwargs.get("method") == "nemotron_parse" or (
            nemotron_parse_workers > 0.0 and gpu_nemotron_parse > 0.0 and nemotron_parse_batch_size > 0.0
        )

        # Forward CLI overrides into the RequestedPlan so that downstream Ray
        # actor pools (batch size, GPU fraction, pool size) honour them.
        overrides: dict[str, Any] = {}
        if nemotron_parse_workers > 0.0:
            workers = int(nemotron_parse_workers)
            overrides["nemotron_parse_initial_actors"] = workers
            overrides["nemotron_parse_min_actors"] = workers
            overrides["nemotron_parse_max_actors"] = workers
        if gpu_nemotron_parse > 0.0:
            overrides["nemotron_parse_gpus_per_actor"] = gpu_nemotron_parse
        if nemotron_parse_batch_size > 0.0:
            overrides["nemotron_parse_batch_size"] = int(nemotron_parse_batch_size)
        if overrides:
            self._requested_plan = self._requested_plan.model_copy(update=overrides)

    def _append_detection_stages(self, kwargs: dict[str, Any]) -> None:
        """Append downstream GPU detection stages (page elements, OCR, table/chart/infographic).

        Shared by ``extract()`` (PDF) and ``extract_image_files()`` (standalone images).
        """
        # Stage-specific kwargs: upstream PDF stages accept many options (dpi, extract_*),
        # but downstream detect_* Ray actors accept only a small set. Passing the whole
        # dict can cause TypeErrors (e.g. unexpected `method=`).
        detect_passthrough_keys = {
            "inference_batch_size",
            "output_column",
            "num_detections_column",
            "counts_by_label_column",
            "api_key",
            "request_timeout_s",
            "remote_max_pool_workers",
            "remote_max_retries",
            "remote_max_429_retries",
        }
        detect_kwargs = {k: kwargs[k] for k in detect_passthrough_keys if k in kwargs}
        page_elements_invoke_url = kwargs.get("page_elements_invoke_url", kwargs.get("invoke_url"))
        if page_elements_invoke_url:
            detect_kwargs["invoke_url"] = page_elements_invoke_url
        if "page_elements_request_timeout_s" in kwargs:
            detect_kwargs["request_timeout_s"] = kwargs["page_elements_request_timeout_s"]
        if "page_elements_api_key" in kwargs:
            detect_kwargs["api_key"] = kwargs["page_elements_api_key"]

        detect_kwargs["inference_batch_size"] = self._requested_plan.get_page_elements_batch_size()

        # In further stages we don't prefer individual rows as batching is more performant.
        # Here we set the target number of rows per block to either
        # nemotron-parse batch size or the page-elements batch size, depending on which is used.
        if self._use_nemotron_parse_only:

            # Set the target number of rows per block to the nemotron-parse batch size
            self._rd_dataset = self._rd_dataset.repartition(
                target_num_rows_per_block=self._requested_plan.get_nemotron_parse_batch_size()
            )

            parse_flags: dict[str, Any] = {}
            if kwargs.get("extract_tables") is True:
                parse_flags["extract_tables"] = True
            if kwargs.get("extract_charts") is True:
                parse_flags["extract_charts"] = True
            if kwargs.get("extract_infographics") is True:
                parse_flags["extract_infographics"] = True
            for k in (
                "api_key",
                "request_timeout_s",
                "remote_max_pool_workers",
                "remote_max_retries",
                "remote_max_429_retries",
            ):
                if k in kwargs:
                    parse_flags[k] = kwargs[k]
            parse_invoke_url = kwargs.get(
                "nemotron_parse_invoke_url", kwargs.get("ocr_invoke_url", kwargs.get("invoke_url"))
            )
            if parse_invoke_url:
                parse_flags["invoke_url"] = parse_invoke_url
            self._rd_dataset = self._rd_dataset.map_batches(
                NemotronParseActor,
                batch_size=self._requested_plan.get_nemotron_parse_batch_size(),
                batch_format="pandas",
                num_gpus=self._requested_plan.get_nemotron_parse_gpus_per_actor(),
                compute=rd.ActorPoolStrategy(
                    initial_size=self._requested_plan.get_nemotron_parse_initial_actors(),
                    min_size=self._requested_plan.get_nemotron_parse_min_actors(),
                    max_size=self._requested_plan.get_nemotron_parse_max_actors(),
                ),
                fn_constructor_kwargs=parse_flags,
            )
        else:
            # Set the target number of rows per block to the page-elements batch size
            self._rd_dataset = self._rd_dataset.repartition(
                target_num_rows_per_block=self._requested_plan.get_page_elements_batch_size()
            )

            # Page-element detection with a GPU actor pool.
            self._rd_dataset = self._rd_dataset.map_batches(
                PageElementDetectionActor,
                batch_size=self._requested_plan.get_page_elements_batch_size(),
                batch_format="pandas",
                num_gpus=self._requested_plan.get_page_elements_gpus_per_actor(),
                compute=rd.ActorPoolStrategy(
                    initial_size=self._requested_plan.get_page_elements_initial_actors(),
                    min_size=self._requested_plan.get_page_elements_min_actors(),
                    max_size=self._requested_plan.get_page_elements_max_actors(),
                ),
                fn_constructor_kwargs=dict(detect_kwargs),
            )

            # Graphic elements detection for charts (runs before OCR).
            use_graphic_elements = bool(kwargs.get("use_graphic_elements", False))
            if use_graphic_elements and kwargs.get("extract_charts") is True:
                ge_kwargs: dict[str, Any] = {}
                ge_invoke_url = kwargs.get("graphic_elements_invoke_url", "")
                if ge_invoke_url:
                    ge_kwargs["graphic_elements_invoke_url"] = ge_invoke_url
                ocr_invoke_url_for_ge = kwargs.get("ocr_invoke_url", kwargs.get("invoke_url"))
                if ocr_invoke_url_for_ge:
                    ge_kwargs["ocr_invoke_url"] = ocr_invoke_url_for_ge
                if "inference_batch_size" in kwargs:
                    ge_kwargs["inference_batch_size"] = kwargs["inference_batch_size"]
                for k in (
                    "api_key",
                    "request_timeout_s",
                    "remote_max_pool_workers",
                    "remote_max_retries",
                    "remote_max_429_retries",
                ):
                    if k in kwargs:
                        ge_kwargs[k] = kwargs[k]
                if "ocr_request_timeout_s" in kwargs:
                    ge_kwargs["request_timeout_s"] = kwargs["ocr_request_timeout_s"]
                if "ocr_api_key" in kwargs:
                    ge_kwargs["api_key"] = kwargs["ocr_api_key"]
                ge_gpu = (
                    0.0
                    if (ge_invoke_url and ocr_invoke_url_for_ge)
                    else self._requested_plan.get_page_elements_gpus_per_actor()
                )
                self._rd_dataset = self._rd_dataset.map_batches(
                    GraphicElementsActor,
                    batch_size=self._requested_plan.get_page_elements_batch_size(),
                    batch_format="pandas",
                    num_gpus=ge_gpu,
                    compute=rd.ActorPoolStrategy(
                        initial_size=self._requested_plan.get_page_elements_initial_actors(),
                        min_size=self._requested_plan.get_page_elements_min_actors(),
                        max_size=self._requested_plan.get_page_elements_max_actors(),
                    ),
                    fn_constructor_kwargs=ge_kwargs,
                )

            use_table_structure = bool(kwargs.get("use_table_structure", False))
            from nemo_retriever.application.pipeline.build_plan import validate_table_structure_flags

            validate_table_structure_flags(
                use_table_structure, str(kwargs.get("table_output_format", "pseudo_markdown"))
            )

            # When use_table_structure is True, tables go through
            # the combined table-structure + OCR stage instead of OCR-only.
            if use_table_structure and kwargs.get("extract_tables") is True:
                ts_ocr_flags: dict[str, Any] = {}
                for k in (
                    "api_key",
                    "request_timeout_s",
                    "remote_max_pool_workers",
                    "remote_max_retries",
                    "remote_max_429_retries",
                ):
                    if k in kwargs:
                        ts_ocr_flags[k] = kwargs[k]
                ts_invoke_url = kwargs.get("table_structure_invoke_url")
                if ts_invoke_url:
                    ts_ocr_flags["table_structure_invoke_url"] = ts_invoke_url
                ocr_invoke_url = kwargs.get("ocr_invoke_url", kwargs.get("invoke_url"))
                if ocr_invoke_url:
                    ts_ocr_flags["ocr_invoke_url"] = ocr_invoke_url
                if "ocr_request_timeout_s" in kwargs:
                    ts_ocr_flags["request_timeout_s"] = kwargs["ocr_request_timeout_s"]
                if "ocr_api_key" in kwargs:
                    ts_ocr_flags["api_key"] = kwargs["ocr_api_key"]

                self._rd_dataset = self._rd_dataset.map_batches(
                    TableStructureActor,
                    batch_size=self._requested_plan.get_ocr_batch_size(),
                    batch_format="pandas",
                    num_gpus=self._requested_plan.get_ocr_gpus_per_actor(),
                    compute=rd.ActorPoolStrategy(
                        initial_size=self._requested_plan.get_ocr_initial_actors(),
                        min_size=self._requested_plan.get_ocr_min_actors(),
                        max_size=self._requested_plan.get_ocr_max_actors(),
                    ),
                    fn_constructor_kwargs=ts_ocr_flags,
                )

            # OCR-based extraction for tables/charts/infographics (single stage).
            # When use_table_structure is True, tables are handled above;
            # charts/infographics still go through OCR.
            ocr_flags = {}
            method = kwargs.get("method", "pdfium")
            if method in ("pdfium_hybrid", "ocr") and kwargs.get("extract_text") is True:
                ocr_flags["extract_text"] = True
            if kwargs.get("extract_tables") is True and not use_table_structure:
                ocr_flags["extract_tables"] = True
            if kwargs.get("extract_charts") is True and not use_graphic_elements:
                ocr_flags["extract_charts"] = True
            if kwargs.get("extract_infographics") is True:
                ocr_flags["extract_infographics"] = True
            for k in (
                "api_key",
                "request_timeout_s",
                "remote_max_pool_workers",
                "remote_max_retries",
                "remote_max_429_retries",
            ):
                if k in kwargs:
                    ocr_flags[k] = kwargs[k]
            ocr_invoke_url = kwargs.get("ocr_invoke_url", kwargs.get("invoke_url"))
            if ocr_invoke_url:
                ocr_flags["invoke_url"] = ocr_invoke_url
            if "ocr_request_timeout_s" in kwargs:
                ocr_flags["request_timeout_s"] = kwargs["ocr_request_timeout_s"]
            if "ocr_api_key" in kwargs:
                ocr_flags["api_key"] = kwargs["ocr_api_key"]

            ocr_flags["inference_batch_size"] = self._requested_plan.get_ocr_batch_size()

            if ocr_flags:
                self._rd_dataset = self._rd_dataset.map_batches(
                    OCRActor,
                    batch_size=self._requested_plan.get_ocr_batch_size(),
                    batch_format="pandas",
                    num_gpus=self._requested_plan.get_ocr_gpus_per_actor(),
                    compute=rd.ActorPoolStrategy(
                        initial_size=self._requested_plan.get_ocr_initial_actors(),
                        min_size=self._requested_plan.get_ocr_min_actors(),
                        max_size=self._requested_plan.get_ocr_max_actors(),
                    ),
                    fn_constructor_kwargs=ocr_flags,
                )

    def extract_image_files(self, params: ExtractParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Configure image-only pipeline: read_binary_files -> ImageLoadActor -> detection stages.

        Use with .files("*.png").extract_image_files(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_image_files().
        """
        from nemo_retriever.image.ray_data import ImageLoadActor

        resolved = _coerce_params(params, ExtractParams, kwargs)
        if (
            any(
                (
                    resolved.invoke_url,
                    resolved.page_elements_invoke_url,
                    resolved.ocr_invoke_url,
                    resolved.graphic_elements_invoke_url,
                    resolved.table_structure_invoke_url,
                )
            )
            and not resolved.api_key
        ):
            resolved = resolved.model_copy(update={"api_key": resolve_remote_api_key()})

        kwargs = {
            **resolved.model_dump(mode="python", exclude={"remote_retry", "batch_tuning"}, exclude_none=True),
            **resolved.remote_retry.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
        }

        self._pipeline_type = "image"
        self._tasks.append(("extract_image_files", dict(kwargs)))

        # Image loading: bytes+path -> page DataFrame (CPU-only, like TxtSplitActor).
        self._rd_dataset = self._rd_dataset.map_batches(
            ImageLoadActor,
            batch_size=4,
            batch_format="pandas",
            num_cpus=1,
        )

        # Downstream detection stages (page elements, OCR, table/chart/infographic).
        self._apply_nemotron_parse_overrides(kwargs)
        self._append_detection_stages(kwargs)

        return self

    def split(self, params: TextChunkParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Re-chunk the ``text`` column by token count (post-extraction transform).

        Adds a ``map_batches(TextChunkActor, ...)`` stage to the Ray Dataset so
        already-extracted text is re-chunked before embedding.
        """
        from nemo_retriever.txt.ray_data import TextChunkActor

        resolved = _coerce_params(params, TextChunkParams, kwargs)
        self._tasks.append(("split", resolved.model_dump(mode="python")))

        self._rd_dataset = self._rd_dataset.map_batches(
            TextChunkActor,
            batch_size=4,
            batch_format="pandas",
            num_cpus=1,
            fn_constructor_kwargs={"params": resolved},
        )
        return self

    def extract_txt(self, params: TextChunkParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Configure txt-only pipeline: read_binary_files -> TxtSplitActor (bytes -> chunk rows).

        Use with .files("*.txt").extract_txt(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_txt().
        """
        from nemo_retriever.txt.ray_data import TxtSplitActor

        self._pipeline_type = "txt"
        resolved = _coerce_params(params, TextChunkParams, kwargs)
        self._extract_txt_kwargs = resolved.model_dump(mode="python")
        self._tasks.append(("extract_txt", dict(self._extract_txt_kwargs)))

        self._rd_dataset = self._rd_dataset.map_batches(
            TxtSplitActor,
            batch_size=4,
            batch_format="pandas",
            num_cpus=1,
            fn_constructor_kwargs={"params": TextChunkParams(**self._extract_txt_kwargs)},
        )
        return self

    def extract_html(self, params: HtmlChunkParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Configure HTML-only pipeline: read_binary_files -> HtmlSplitActor (bytes -> chunk rows).

        Use with .files("*.html").extract_html(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_html().
        """
        from nemo_retriever.html.ray_data import HtmlSplitActor

        self._pipeline_type = "html"
        resolved = _coerce_params(params, HtmlChunkParams, kwargs)
        self._extract_html_kwargs = resolved.model_dump(mode="python")
        self._tasks.append(("extract_html", dict(self._extract_html_kwargs)))

        self._rd_dataset = self._rd_dataset.map_batches(
            HtmlSplitActor,
            batch_size=4,
            batch_format="pandas",
            num_cpus=1,
            fn_constructor_kwargs={"params": HtmlChunkParams(**self._extract_html_kwargs)},
        )
        return self

    def extract_audio(
        self,
        params: AudioChunkParams | None = None,
        asr_params: ASRParams | None = None,
        **kwargs: Any,
    ) -> "BatchIngestor":
        """
        Configure audio pipeline: read_binary_files -> MediaChunkActor -> ASRActor (chunk -> transcript).

        Use with .files("mp3/*.mp3").extract_audio(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_audio().
        ASR requires a remote or self-deployed Parakeet/Riva gRPC endpoint (see ASRParams.audio_endpoints).
        Optional kwargs: audio_chunk_batch_size (default 4), asr_batch_size (default 8),
        asr_num_gpus (default 0.5; GPUs reserved per ASR actor for local Parakeet).
        """
        from nemo_retriever.audio import ASRActor
        from nemo_retriever.audio import MediaChunkActor

        self._pipeline_type = "audio"
        chunk_resolved = _coerce_params(params, AudioChunkParams, kwargs)
        asr_resolved = _coerce_params(asr_params, ASRParams, kwargs)
        self._extract_audio_chunk_kwargs = chunk_resolved.model_dump(mode="python")
        self._extract_audio_asr_kwargs = asr_resolved.model_dump(mode="python")
        self._tasks.append(
            ("extract_audio", {"chunk": self._extract_audio_chunk_kwargs, "asr": self._extract_audio_asr_kwargs})
        )

        audio_chunk_batch_size = kwargs.get("audio_chunk_batch_size", 4)
        asr_batch_size = kwargs.get("asr_batch_size", 8)
        asr_num_gpus = kwargs.get("asr_num_gpus", 0.5)

        self._rd_dataset = self._rd_dataset.map_batches(
            MediaChunkActor,
            batch_size=audio_chunk_batch_size,
            batch_format="pandas",
            num_cpus=1,
            fn_constructor_kwargs={"params": AudioChunkParams(**self._extract_audio_chunk_kwargs)},
        )
        self._rd_dataset = self._rd_dataset.map_batches(
            ASRActor,
            batch_size=asr_batch_size,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=asr_num_gpus,
            fn_constructor_kwargs={"params": ASRParams(**self._extract_audio_asr_kwargs)},
        )
        return self

    def embed(
        self,
        params: EmbedParams | None = None,
        input_dataset: rd.Dataset | None = None,
        **kwargs: Any,
    ) -> "BatchIngestor":
        """
        Add a text-embedding stage to the batch pipeline.

        Uses a GPU actor pool so the HuggingFace model stays resident across
        batches.  Resource-tuning kwargs:

        - ``embed_workers``: ActorPool size (default 1).
        - ``embed_batch_size``: Ray Data batch size (default 256).
        - ``embed_cpus_per_actor``: CPUs reserved per embedding actor (default 1).
        - ``device``, ``hf_cache_dir``, ``normalize``, ``max_length``:
          forwarded to ``LlamaNemotronEmbed1BV2Embedder``.
        - ``embedding_endpoint`` / ``embed_invoke_url``: optional NIM endpoint URL
          (e.g. ``"http://embedding:8000/v1"``).  When set, the actor
          delegates to the remote NIM instead of loading a local model,
          and no GPU is requested for this stage.
        - ``input_dataset``: optional Ray Dataset to embed instead of the
          currently configured internal dataset.
        """
        if input_dataset is not None:
            self._rd_dataset = input_dataset
        if self._rd_dataset is None:
            raise RuntimeError(
                "No Ray Dataset to embed. Provide input_dataset or run .files(...) / .extract(...) first."
            )

        from nemo_retriever.params.utils import build_embed_kwargs

        resolved = _coerce_params(params, EmbedParams, kwargs)
        if any((resolved.embedding_endpoint, resolved.embed_invoke_url)) and not resolved.api_key:
            resolved = resolved.model_copy(update={"api_key": resolve_remote_api_key()})

        kwargs = build_embed_kwargs(resolved, include_batch_tuning=True)
        self._refresh_requested_plan(kwargs)

        # Remaining kwargs are forwarded to the actor constructor.
        embed_modality = resolved.embed_modality
        embed_granularity = resolved.embed_granularity
        self._tasks.append(("embed", dict(kwargs)))

        # We want to create Ray batches that are of the same size as the embed_batch_size.
        self._rd_dataset = self._rd_dataset.repartition(
            target_num_rows_per_block=self._requested_plan.get_embed_batch_size()
        )

        if embed_granularity == "page":
            _row_fn = partial(
                collapse_content_to_page_rows,
                modality=embed_modality,
            )
        else:
            text_elements_modality = resolved.text_elements_modality or embed_modality
            structured_elements_modality = resolved.structured_elements_modality or embed_modality
            _row_fn = partial(
                explode_content_to_rows,
                modality=embed_modality,
                text_elements_modality=text_elements_modality,
                structured_elements_modality=structured_elements_modality,
            )
        self._rd_dataset = self._rd_dataset.map_batches(
            _row_fn,
            batch_size=self._requested_plan.get_embed_batch_size(),
            batch_format="pandas",
            num_cpus=1,
        )

        # When using a remote NIM endpoint, no GPU is needed for embedding.
        endpoint = (kwargs.get("embedding_endpoint") or kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            embed_actor_num_gpus = 0  # We do not need GPU resources if invoking a remote NIM endpoint
        else:
            embed_actor_num_gpus = self._requested_plan.get_embed_gpus_per_actor()

        self._rd_dataset = self._rd_dataset.map_batches(
            _BatchEmbedActor,
            batch_size=self._requested_plan.get_embed_batch_size(),
            batch_format="pandas",
            num_gpus=embed_actor_num_gpus,  # pulled from if statement above
            compute=rd.ActorPoolStrategy(
                initial_size=self._requested_plan.get_embed_initial_actors(),
                min_size=self._requested_plan.get_embed_min_actors(),
                max_size=self._requested_plan.get_embed_max_actors(),
            ),
            fn_constructor_kwargs={"params": resolved},
        )

        return self

    def vdb_upload(self, params: VdbUploadParams | None = None, **kwargs: Any) -> "BatchIngestor":
        """
        Add a streaming LanceDB upload stage to the batch pipeline.

        Instead of buffering the entire dataset into pandas, this adds a
        ``map_batches`` stage with a ``_LanceDBWriteActor`` that writes each
        batch to LanceDB as it streams through.  Index creation is deferred
        to ``ingest()`` (must happen after all writes).

        Accepts the same kwargs as
        ``inprocess.upload_embeddings_to_lancedb_inprocess``.
        """
        p = params or VdbUploadParams()
        if kwargs:
            lancedb_kwargs = {k: v for k, v in kwargs.items() if k != "purge_results_after_upload"}
            if lancedb_kwargs:
                p = p.model_copy(update={"lancedb": p.lancedb.model_copy(update=lancedb_kwargs)})
            if "purge_results_after_upload" in kwargs:
                p = p.model_copy(update={"purge_results_after_upload": bool(kwargs["purge_results_after_upload"])})
        _ = p.purge_results_after_upload
        vdb_kwargs = p.lancedb.model_dump(mode="python")
        self._tasks.append(("vdb_upload", dict(vdb_kwargs)))
        self._vdb_upload_kwargs = dict(vdb_kwargs)

        # Streaming write stage — single actor, CPU-only, no GPU needed.
        self._rd_dataset = self._rd_dataset.map_batches(
            _LanceDBWriteActor,
            batch_format="pandas",
            num_cpus=1,
            compute=rd.ActorPoolStrategy(size=1),
            fn_constructor_kwargs={"params": p},
        )

        return self

    def save_intermediate_results(self, output_dir: str) -> "BatchIngestor":
        """
        Persist the current Ray Dataset to disk under `output_dir`.

        Writes Parquet files (Ray Data's native/efficient on-disk format) so downstream
        steps can reload as a Ray Dataset using `ray.data.read_parquet(...)`.
        """
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"output_dir must be a non-empty string, got {output_dir!r}")
        if self._rd_dataset is None:
            raise RuntimeError("No Ray Dataset to write. Call .files(...) (and optionally .extract(...)) first.")

        base_dir = os.path.abspath(output_dir)
        os.makedirs(base_dir, exist_ok=True)

        # Ray's writers typically expect a directory that does not already contain output.
        # To avoid destructive behavior, if the directory is non-empty we write to a timestamped subdir.
        target_dir = base_dir
        try:
            if os.listdir(base_dir):
                ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                target_dir = os.path.join(base_dir, f"ray_dataset_{ts}")
                os.makedirs(target_dir, exist_ok=False)
        except FileNotFoundError:
            # Rare race: directory disappeared between abspath and listdir.
            os.makedirs(base_dir, exist_ok=True)

        # Trigger execution and write results.
        # Parquet supports nested list/struct columns used by our stages (e.g. detections payloads).
        self._rd_dataset.write_parquet(target_dir)
        self._intermediate_output_dir = target_dir
        return self

    # Backwards-compatibility: some examples call `write_to_disk(...)`.
    def write_to_disk(self, output_dir: str) -> "BatchIngestor":
        return self.save_intermediate_results(output_dir=output_dir)

    @staticmethod
    def _has_error(v: Any) -> bool:
        """Recursively detect whether a value contains error-like payloads."""

        def _is_populated_error_field(key: str, value: Any) -> bool:
            if value is None:
                return False
            if key == "failed" and isinstance(value, bool):
                return value
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, set, dict)):
                return len(value) > 0
            return bool(value)

        if v is None:
            return False
        if isinstance(v, dict):
            for k in ("error", "errors", "exception", "traceback", "failed"):
                if k in v and _is_populated_error_field(k, v.get(k)):
                    return True
            return any(BatchIngestor._has_error(x) for x in v.values())
        if isinstance(v, list):
            return any(BatchIngestor._has_error(x) for x in v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return False
            # Parse JSON-like strings first, then fall back to keyword matching.
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    return BatchIngestor._has_error(json.loads(s))
                except Exception:
                    pass
            low = s.lower()
            return any(tok in low for tok in ("error", "exception", "traceback", "failed"))
        return False

    @staticmethod
    def has_error(v: Any) -> bool:
        """Public helper for checking if a scalar value carries an error."""
        return BatchIngestor._has_error(v)

    @staticmethod
    def extract_error_rows(batch: Any) -> Any:
        """
        Return only rows that contain error-like payloads in known columns.

        Expected to run with ``batch_format="pandas"`` from Ray Data.
        """
        if batch is None:
            return batch
        columns = getattr(batch, "columns", None)
        if columns is None:
            return batch
        error_candidate_columns = (
            "error",
            "errors",
            "exception",
            "traceback",
            "metadata",
            "source",
            "embedding",
        )
        cols = [c for c in error_candidate_columns if c in columns]
        if not cols:
            return batch.iloc[0:0]

        mask = batch[cols[0]].apply(BatchIngestor._has_error).astype(bool)
        for c in cols[1:]:
            mask = mask | batch[c].apply(BatchIngestor._has_error).astype(bool)
        return batch[mask]

    def get_error_rows(self, dataset: rd.Dataset | None = None) -> rd.Dataset:
        """
        Build a dataset containing only error rows from this pipeline.

        If ``dataset`` is omitted, uses the ingestor's current internal dataset.
        """
        target = dataset if dataset is not None else self._rd_dataset
        if target is None:
            raise RuntimeError("No Ray Dataset available to inspect for errors.")
        return target.map_batches(self.extract_error_rows, batch_format="pandas")

    def get_dataset(self) -> rd.Dataset | None:
        """Return the current in-memory Ray Dataset for this ingestor."""
        return self._rd_dataset

    def ingest(self, params: IngestExecuteParams | None = None, **kwargs: Any) -> int:
        """
        Execute the Ray Data pipeline and return the total number of pages.

        If a VDB upload stage was added (via ``vdb_upload()``), data is written
        to LanceDB in a streaming fashion by ``_LanceDBWriteActor``.  After the
        pipeline finishes, we create the LanceDB vector index (which must happen
        after all writes are complete).
        """
        run_params = _coerce_params(params, IngestExecuteParams, kwargs)
        _ = (
            run_params.show_progress,
            run_params.return_failures,
            run_params.save_to_disk,
            run_params.return_traces,
        )

        return self

    def _create_lancedb_index(self) -> None:
        """Create the LanceDB vector index after streaming writes finish."""
        kw = self._vdb_upload_kwargs
        if not kw.get("create_index", True):
            return

        lancedb_uri = str(kw.get("lancedb_uri", "lancedb"))
        table_name = str(kw.get("table_name", "nv-ingest"))
        index_type = str(kw.get("index_type", "IVF_HNSW_SQ"))
        metric = str(kw.get("metric", "l2"))
        num_partitions = int(kw.get("num_partitions", 16))
        num_sub_vectors = int(kw.get("num_sub_vectors", 256))

        try:
            import lancedb  # type: ignore
        except Exception as e:
            print(f"Warning: lancedb not available for index creation: {e}")
            return

        try:
            db = lancedb.connect(uri=lancedb_uri)
            table = db.open_table(table_name)
            n_vecs = table.count_rows()
        except Exception as e:
            print(f"Warning: could not open LanceDB table for indexing: {e}")
            return

        if n_vecs < 2:
            print("Skipping LanceDB index creation (not enough vectors).")
            return

        k = int(num_partitions)
        if k >= n_vecs:
            k = max(1, n_vecs - 1)

        try:
            table.create_index(
                index_type=index_type,
                metric=metric,
                num_partitions=k,
                num_sub_vectors=num_sub_vectors,
                vector_column_name="vector",
            )
        except TypeError:
            table.create_index(vector_column_name="vector")
        except Exception as e:
            print(f"Warning: failed to create LanceDB index (continuing without index): {e}")

        if kw.get("hybrid", False):
            text_column = str(kw.get("text_column", "text"))
            fts_language = str(kw.get("fts_language", "English"))
            try:
                table.create_fts_index(text_column, language=fts_language)
            except Exception as e:
                print(
                    f"Warning: FTS index creation failed on column {text_column!r} (continuing with vector-only): {e}"
                )

        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

        print(f"Wrote {n_vecs} rows to LanceDB uri={lancedb_uri!r} table={table_name!r}")
