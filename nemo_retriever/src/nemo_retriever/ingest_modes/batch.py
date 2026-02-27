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
import time
from pathlib import Path
from typing import Any, List, Optional
from datetime import timedelta

from typing import Union

import ray
import ray.data as rd
from nemo_retriever.utils.convert import DocToPdfConversionActor
from nemo_retriever.page_elements import PageElementDetectionActor
from nemo_retriever.ocr.ocr import OCRActor
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor

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

DEBUG_LOG_PATH = "/home/jeremy/Development/nv-ingest/.cursor/debug-250ae2.log"


def _coerce_params[T](params: T | None, model_cls: type[T], kwargs: dict[str, Any]) -> T:
    if params is None:
        return model_cls(**kwargs)
    if kwargs:
        return params.model_copy(update=kwargs)  # type: ignore[return-value]
    return params


# region agent log
def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": "250ae2",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion


class _LanceDBWriteActor:
    """Ray Data actor that streams batches into LanceDB as they arrive.

    Creates the table on the first batch, then appends subsequent batches.
    Index creation is intentionally deferred until after the full pipeline
    has been consumed (handled by ``BatchIngestor.ingest()``).
    """

    def __init__(self, params: VdbUploadParams | None = None) -> None:
        import json
        from pathlib import Path

        self._json = json
        self._Path = Path
        lancedb_params = (params or VdbUploadParams()).lancedb

        self._lancedb_uri = lancedb_params.lancedb_uri
        self._table_name = lancedb_params.table_name
        self._overwrite = lancedb_params.overwrite
        self._embedding_column = lancedb_params.embedding_column
        self._embedding_key = lancedb_params.embedding_key
        self._include_text = lancedb_params.include_text
        self._text_column = lancedb_params.text_column

        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore

        self._pa = pa
        self._db = lancedb.connect(uri=self._lancedb_uri)
        self._table = None
        self._schema = None
        self._first_batch = True
        self._total_rows = 0
        self._table = None
        mode = "overwrite" if self._overwrite else "create"
        fields = [
            pa.field("vector", pa.list_(pa.float32(), 2048)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
        ]
        self._schema = pa.schema(fields)

        self._table = self._db.create_table(
            self._table_name,
            schema=self._schema,
            mode=mode,
        )

    def _build_rows(self, df: Any) -> list:
        """Build LanceDB rows from a pandas DataFrame batch.

        Mirrors the row-building logic from
        ``upload_embeddings_to_lancedb_inprocess`` in inprocess.py.
        """
        rows: list = []
        for row in df.itertuples(index=False):
            # Extract embedding
            emb = None
            meta = getattr(row, "metadata", None)
            if isinstance(meta, dict):
                emb = meta.get("embedding")
                if not (isinstance(emb, list) and emb):
                    emb = None
            if emb is None:
                payload = getattr(row, self._embedding_column, None)
                if isinstance(payload, dict):
                    emb = payload.get(self._embedding_key)
                    if not (isinstance(emb, list) and emb):
                        emb = None
            if emb is None:
                continue

            # Extract source path and page number
            path = ""
            page = -1
            v = getattr(row, "path", None)
            if isinstance(v, str) and v.strip():
                path = v.strip()
            v = getattr(row, "page_number", None)
            try:
                if v is not None:
                    page = int(v)
            except Exception:
                pass
            if isinstance(meta, dict):
                sp = meta.get("source_path")
                if isinstance(sp, str) and sp.strip():
                    path = sp.strip()

            p = self._Path(path) if path else None
            filename = p.name if p is not None else ""
            pdf_basename = p.stem if p is not None else ""
            pdf_page = f"{pdf_basename}_{page}" if (pdf_basename and page >= 0) else ""
            source_id = path or filename or pdf_basename

            metadata_obj = {"page_number": int(page) if page is not None else -1}
            if pdf_page:
                metadata_obj["pdf_page"] = pdf_page
            # Persist per-page detection counters for end-of-run summaries.
            # These may be duplicated across exploded content rows; downstream
            # summary logic should dedupe by (source_id, page_number).
            pe_num = getattr(row, "page_elements_v3_num_detections", None)
            if pe_num is not None:
                try:
                    metadata_obj["page_elements_v3_num_detections"] = int(pe_num)
                except Exception:
                    pass
            pe_counts = getattr(row, "page_elements_v3_counts_by_label", None)
            if isinstance(pe_counts, dict):
                metadata_obj["page_elements_v3_counts_by_label"] = {
                    str(k): int(v) for k, v in pe_counts.items() if isinstance(k, str) and v is not None
                }
            for ocr_col in ("table", "chart", "infographic"):
                entries = getattr(row, ocr_col, None)
                if isinstance(entries, list):
                    metadata_obj[f"ocr_{ocr_col}_detections"] = int(len(entries))
            source_obj = {"source_id": str(path)}

            row_out = {
                "vector": emb,
                "pdf_page": pdf_page,
                "filename": filename,
                "pdf_basename": pdf_basename,
                "page_number": int(page) if page is not None else -1,
                "source_id": str(source_id),
                "path": str(path),
                "metadata": self._json.dumps(metadata_obj, ensure_ascii=False),
                "source": self._json.dumps(source_obj, ensure_ascii=False),
            }

            if self._include_text:
                t = getattr(row, self._text_column, None)
                row_out["text"] = str(t) if isinstance(t, str) else ""
            else:
                row_out["text"] = ""

            rows.append(row_out)
        return rows

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

        device = self._kwargs.get("device")
        hf_cache_dir = self._kwargs.get("hf_cache_dir")
        normalize = bool(self._kwargs.get("normalize", True))
        max_length = int(self._kwargs.get("max_length", 8192))
        model_name_raw = self._kwargs.get("model_name")

        from nemo_retriever.model import is_vl_embed_model, resolve_embed_model

        model_id = resolve_embed_model(model_name_raw)

        if is_vl_embed_model(model_name_raw):
            from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            self._model = LlamaNemotronEmbedVL1BV2Embedder(
                device=str(device) if device else None,
                hf_cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
                model_id=model_id,
            )
        else:
            from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

            self._model = LlamaNemotronEmbed1BV2Embedder(
                device=str(device) if device else None,
                hf_cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
                normalize=normalize,
                max_length=max_length,
                model_id=model_id,
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
    ) -> None:
        super().__init__(documents=documents)

        logging.basicConfig(level=logging.INFO)

        # Initialize Ray for distributed execution.
        ray.init(address=ray_address or "local", ignore_reinit_error=True, log_to_driver=bool(ray_log_to_driver))

        # Use the new Rich progress UI instead of verbose tqdm bars.
        ctx = rd.DataContext.get_current()
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False

        # Query available resources so extract() can auto-size worker pools.
        resources = ray.available_resources()
        self._num_gpus = int(resources.get("GPU", 0))
        self._num_cpus = int(resources.get("CPU", os.cpu_count() or 4))

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

        resolved = _coerce_params(params, ExtractParams, kwargs)
        kwargs = {
            **resolved.model_dump(mode="python", exclude={"remote_retry", "batch_tuning"}, exclude_none=True),
            **resolved.remote_retry.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
        }

        # -- Pop resource-tuning kwargs before forwarding to actors --
        def _endpoint_count(raw: Any) -> int:
            s = str(raw or "").strip()
            if not s:
                return 0
            return len([p for p in s.split(",") if p.strip()])

        debug_run_id = str(kwargs.pop("debug_run_id", "unknown"))
        pdf_split_batch_size = kwargs.pop("pdf_split_batch_size", 1)
        pdf_extract_batch_size = kwargs.pop("pdf_extract_batch_size", 4)
        pdf_extract_num_cpus = float(kwargs.pop("pdf_extract_num_cpus", 2))
        page_elements_batch_size = kwargs.pop("page_elements_batch_size", 24)
        detect_batch_size = kwargs.pop("detect_batch_size", 24)

        # Count GPU stages that will be created (page_elements is always on).
        # +1 reserves headroom for a downstream embed() stage.
        detect_stage_count = (
            1 if any(kwargs.get(k) is True for k in ("extract_tables", "extract_charts", "extract_infographics")) else 0
        )
        gpu_stage_count = 1 + detect_stage_count + 1  # page_elements + detection + embed

        # Per-stage GPU allocation: give OCR (the bottleneck) a full GPU;
        # page-elements (lightweight YOLOX) and embedding share 0.5 each.
        # Total = 0.5 + 1.0 + 0.5 = 2.0, so all 3 stages run concurrently.
        num_gpus = self._num_gpus
        if num_gpus >= 2 and gpu_stage_count == 3:
            gpu_page_elements = 0.5
            gpu_ocr = 1.0
            gpu_embed = 0.5
        else:
            gpu_per_stage = min(1.0, num_gpus / max(1, gpu_stage_count))
            gpu_page_elements = gpu_per_stage
            gpu_ocr = gpu_per_stage
            gpu_embed = gpu_per_stage

        # Allow explicit per-stage GPU overrides for controlled experiments.
        gpu_page_elements = float(kwargs.pop("gpu_page_elements", gpu_page_elements))
        gpu_ocr = float(kwargs.pop("gpu_ocr", gpu_ocr))
        gpu_embed = float(kwargs.pop("gpu_embed", gpu_embed))

        # Each GPU stage gets 1 worker by default (each worker holds 1 model).
        page_elements_workers = kwargs.pop("page_elements_workers", 1)
        ocr_workers = kwargs.pop("ocr_workers", 1)
        detect_workers = kwargs.pop("detect_workers", ocr_workers)
        page_elements_cpus_per_actor = float(kwargs.pop("page_elements_cpus_per_actor", 1))
        ocr_cpus_per_actor = float(kwargs.pop("ocr_cpus_per_actor", 1))

        # When remote endpoints are provided as a comma-separated list, scale
        # actor count to match endpoint count so load can be distributed.
        pe_endpoints = _endpoint_count(kwargs.get("page_elements_invoke_url", kwargs.get("invoke_url")))
        if pe_endpoints > 0 and int(page_elements_workers) != int(pe_endpoints):
            logging.warning(
                "page_elements invoke URL list has %d endpoint(s); overriding page_elements_workers from %d to %d",
                pe_endpoints,
                int(page_elements_workers),
                int(pe_endpoints),
            )
            page_elements_workers = int(pe_endpoints)

        ocr_endpoints = _endpoint_count(kwargs.get("ocr_invoke_url", kwargs.get("invoke_url")))
        if ocr_endpoints > 0 and int(detect_workers) != int(ocr_endpoints):
            logging.warning(
                "ocr invoke URL list has %d endpoint(s); overriding detect_workers from %d to %d",
                ocr_endpoints,
                int(detect_workers),
                int(ocr_endpoints),
            )
            detect_workers = int(ocr_endpoints)

        # Reserve CPUs for GPU actors, then divide the rest among extract workers.
        total_gpu_cpus = (
            page_elements_workers * page_elements_cpus_per_actor
            + detect_workers * detect_stage_count * ocr_cpus_per_actor
        )
        cpus_for_extract = max(1, self._num_cpus - total_gpu_cpus)
        pdf_extract_workers = kwargs.pop("pdf_extract_workers", max(1, cpus_for_extract // 2))

        # region agent log
        total_gpu_requested = (
            float(gpu_page_elements) * float(page_elements_workers)
            + float(gpu_ocr) * float(detect_workers * detect_stage_count)
            + float(gpu_embed) * 1.0
        )
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H1",
            location="ingest_modes/batch.py:extract",
            message="Resource envelope computed for Ray stages",
            data={
                "num_gpus_available": self._num_gpus,
                "num_cpus_available": self._num_cpus,
                "gpu_page_elements": gpu_page_elements,
                "gpu_ocr": gpu_ocr,
                "gpu_embed": gpu_embed,
                "page_elements_workers": page_elements_workers,
                "detect_workers": detect_workers,
                "detect_stage_count": detect_stage_count,
                "pdf_extract_workers": pdf_extract_workers,
                "pdf_extract_num_cpus": pdf_extract_num_cpus,
                "total_gpu_requested_across_stages": total_gpu_requested,
                "can_fully_overlap_gpu_stages": bool(total_gpu_requested <= float(self._num_gpus)),
            },
        )
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H2",
            location="ingest_modes/batch.py:extract",
            message="CPU reservation breakdown for actors and extraction pool",
            data={
                "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
                "ocr_cpus_per_actor": ocr_cpus_per_actor,
                "total_gpu_actor_cpus_reserved": total_gpu_cpus,
                "cpus_for_extract": cpus_for_extract,
            },
        )
        # endregion

        # Store per-stage GPU allocations for downstream stages (e.g. embed).
        self._gpu_page_elements = gpu_page_elements
        self._gpu_ocr = gpu_ocr
        self._gpu_embed = gpu_embed

        logging.info(
            "Batch extract resources: %d GPUs, %d CPUs | "
            "pdf_extract_workers=%d, page_elements_workers=%d, ocr_workers=%d, "
            "gpu_page_elements=%.2f, gpu_ocr=%.2f, gpu_embed=%.2f",
            self._num_gpus,
            self._num_cpus,
            pdf_extract_workers,
            page_elements_workers,
            detect_workers,
            gpu_page_elements,
            gpu_ocr,
            gpu_embed,
        )

        # Downstream batch stages assume `page_image.image_b64` exists for every page.
        # Ensure PDF extraction emits a page image unless the caller explicitly disables it.
        kwargs.setdefault("extract_page_as_image", True)

        # 200 DPI is sufficient for both detection and OCR.  YOLOX resizes to
        # 1024x1024 internally, and NemotronOCR also resizes crops to 1024x1024,
        # so resolution above ~1200px per side is wasted.  200 DPI (Letter =
        # 1700x2200) gives enough detail while reducing extraction time and
        # memory usage by ~30-40% vs 300 DPI.
        kwargs.setdefault("dpi", 200)

        self._pipeline_type = "pdf"
        self._tasks.append(("extract", dict(kwargs)))

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

        # Convert DOCX/PPTX to PDF before splitting.  CPU-only, one
        # LibreOffice process per file (batch_size=1).
        self._rd_dataset = self._rd_dataset.map_batches(
            DocToPdfConversionActor,
            batch_size=1,
            num_cpus=1,
            num_gpus=0,
            batch_format="pandas",
        )

        # Splitting pdfs is broken into a separate stage to help amortize downstream
        # processing if PDFs have vastly different numbers of pages.
        pdf_split_actor = PDFSplitActor(
            split_params=PdfSplitParams(
                start_page=kwargs.get("start_page"),
                end_page=kwargs.get("end_page"),
            )
        )
        self._rd_dataset = self._rd_dataset.map_batches(
            pdf_split_actor,
            batch_size=pdf_split_batch_size,
            num_cpus=1,
            num_gpus=0,
            batch_format="pandas",
        )

        # Pre-split pdfs are now ready for extraction — the main CPU bottleneck.
        extraction_actor = PDFExtractionActor(**kwargs)
        self._rd_dataset = self._rd_dataset.map_batches(
            extraction_actor,
            batch_size=pdf_extract_batch_size,
            batch_format="pandas",
            num_cpus=pdf_extract_num_cpus,
            num_gpus=0,
            compute=rd.TaskPoolStrategy(size=pdf_extract_workers),
        )
        self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=24)
        # Page-element detection with a GPU actor pool.
        # For ActorPoolStrategy, Ray Data expects a *callable class* (so it can
        # construct one instance per actor). Passing an already-constructed
        # callable object is treated as a "regular function" and will fail.
        self._rd_dataset = self._rd_dataset.map_batches(
            PageElementDetectionActor,
            batch_size=page_elements_batch_size,
            batch_format="pandas",
            num_cpus=page_elements_cpus_per_actor,
            num_gpus=gpu_page_elements,
            compute=rd.ActorPoolStrategy(size=page_elements_workers),
            fn_constructor_kwargs=dict(detect_kwargs),
        )

        # OCR-based extraction for tables/charts/infographics (single stage).
        ocr_flags = {}
        if kwargs.get("extract_tables") is True:
            ocr_flags["extract_tables"] = True
        if kwargs.get("extract_charts") is True:
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

        if ocr_flags:
            self._rd_dataset = self._rd_dataset.map_batches(
                OCRActor,
                batch_size=detect_batch_size,
                batch_format="pandas",
                num_cpus=ocr_cpus_per_actor,
                num_gpus=gpu_ocr,
                compute=rd.ActorPoolStrategy(size=detect_workers),
                fn_constructor_kwargs=ocr_flags,
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
            num_gpus=0,
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
            num_gpus=0,
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
        """
        from retriever.audio import ASRActor
        from retriever.audio import MediaChunkActor

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

        self._rd_dataset = self._rd_dataset.map_batches(
            MediaChunkActor,
            batch_size=audio_chunk_batch_size,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
            fn_constructor_kwargs={"params": AudioChunkParams(**self._extract_audio_chunk_kwargs)},
        )
        self._rd_dataset = self._rd_dataset.map_batches(
            ASRActor,
            batch_size=asr_batch_size,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
            fn_constructor_kwargs={"params": ASRParams(**self._extract_audio_asr_kwargs)},
        )
        return self

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "BatchIngestor":
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
        """

        resolved = _coerce_params(params, EmbedParams, kwargs)
        kwargs = {
            **resolved.model_dump(
                mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True
            ),
            **resolved.runtime.model_dump(mode="python", exclude_none=True),
            **resolved.batch_tuning.model_dump(mode="python", exclude_none=True),
        }

        def _endpoint_count(raw: Any) -> int:
            s = str(raw or "").strip()
            if not s:
                return 0
            return len([p for p in s.split(",") if p.strip()])

        embed_workers = kwargs.pop("embed_workers", 1)
        embed_batch_size = kwargs.pop("embed_batch_size", 256)
        embed_cpus_per_actor = float(kwargs.pop("embed_cpus_per_actor", 1))

        if "embedding_endpoint" not in kwargs and kwargs.get("embed_invoke_url"):
            kwargs["embedding_endpoint"] = kwargs.get("embed_invoke_url")

        endpoint_count = _endpoint_count(kwargs.get("embedding_endpoint"))
        if endpoint_count > 0 and int(embed_workers) != int(endpoint_count):
            logging.warning(
                "embed endpoint list has %d endpoint(s); overriding embed_workers from %d to %d",
                endpoint_count,
                int(embed_workers),
                int(endpoint_count),
            )
            embed_workers = int(endpoint_count)

        # Remaining kwargs are forwarded to the actor constructor.
        embed_modality = resolved.embed_modality
        text_elements_modality = resolved.text_elements_modality or embed_modality
        structured_elements_modality = resolved.structured_elements_modality or embed_modality
        self._tasks.append(("embed", dict(kwargs)))

        # Explode content rows before embedding so each table/chart/infographic
        # gets its own embedding vector (mirrors nv-ingest per-element embeddings).
        self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=256)

        from functools import partial
        from nemo_retriever.ingest_modes.inprocess import explode_content_to_rows

        _explode_fn = partial(
            explode_content_to_rows,
            modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
        )
        self._rd_dataset = self._rd_dataset.map_batches(
            _explode_fn,
            batch_size=embed_batch_size,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
        )

        # When using a remote NIM endpoint, no GPU is needed for embedding.
        endpoint = (kwargs.get("embedding_endpoint") or kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            gpu_per_stage = 0
        else:
            # Embedding is GPU-bound; only needs modest CPU for tokenisation.
            # Requesting all CPUs would prevent this stage from overlapping with
            # upstream extraction/detection in Ray Data's streaming pipeline.
            gpu_per_stage = getattr(self, "_gpu_embed", 1.0)

        self._rd_dataset = self._rd_dataset.map_batches(
            _BatchEmbedActor,
            batch_size=embed_batch_size,
            batch_format="pandas",
            num_cpus=embed_cpus_per_actor,
            num_gpus=gpu_per_stage,
            compute=rd.ActorPoolStrategy(size=embed_workers),
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
            num_gpus=0,
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
        runtime_metrics_dir = run_params.runtime_metrics_dir
        runtime_metrics_prefix = run_params.runtime_metrics_prefix
        t0 = time.monotonic()
        num_pages = self._rd_dataset.count()
        elapsed = time.monotonic() - t0

        print(f"[done] {len(self._input_documents)} files, {num_pages} pages in {elapsed:.1f}s")
        # region agent log
        _debug_log(
            run_id=str(runtime_metrics_prefix or "unknown"),
            hypothesis_id="H3",
            location="ingest_modes/batch.py:ingest",
            message="Pipeline completed with aggregate throughput",
            data={
                "input_files": len(self._input_documents),
                "num_pages": int(num_pages),
                "elapsed_seconds": float(elapsed),
                "pages_per_second_total": float(num_pages / elapsed) if elapsed > 0 else None,
                "runtime_metrics_dir": runtime_metrics_dir,
            },
        )
        # endregion

        # Best-effort runtime metrics capture for per-run stage-level debugging.
        if runtime_metrics_dir:
            metrics_dir = Path(runtime_metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            prefix = (runtime_metrics_prefix or "run").strip() or "run"

            stats_path = metrics_dir / f"{prefix}.rd_dataset.stats.txt"
            timeline_path = metrics_dir / f"{prefix}.ray.timeline.json"
            summary_path = metrics_dir / f"{prefix}.runtime.summary.json"

            stats_text = ""
            try:
                stats_text = str(self._rd_dataset.stats())
                stats_path.write_text(stats_text, encoding="utf-8")
            except Exception as e:
                print(f"Warning: failed writing dataset stats: {e}")

            try:
                ray.timeline(filename=str(timeline_path))
            except Exception as e:
                print(f"Warning: failed writing ray timeline: {e}")

            try:
                summary = {
                    "input_files": int(len(self._input_documents)),
                    "num_pages": int(num_pages),
                    "elapsed_seconds": float(elapsed),
                    "stats_path": str(stats_path),
                    "timeline_path": str(timeline_path),
                }
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"Warning: failed writing runtime summary: {e}")

        # Create LanceDB vector index after all streaming writes are complete.
        if hasattr(self, "_vdb_upload_kwargs") and self._vdb_upload_kwargs:
            self._create_lancedb_index()

        return num_pages

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
