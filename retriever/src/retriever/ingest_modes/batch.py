"""
Batch runmode.

Intended for large-scale batch execution over large inputs on multiple workers.
"""

from __future__ import annotations

import datetime as _dt
import glob
import logging
import os
import time
from typing import Any, List, Optional
from datetime import timedelta

from typing import Union

import ray
import ray.data as rd
from retriever.page_elements import PageElementDetectionActor
from retriever.ocr.ocr import OCRActor
from retriever.pdf.extract import PDFExtractionActor
from retriever.pdf.split import PDFSplitActor

from ..ingest import Ingestor


class _LanceDBWriteActor:
    """Ray Data actor that streams batches into LanceDB as they arrive.

    Creates the table on the first batch, then appends subsequent batches.
    Index creation is intentionally deferred until after the full pipeline
    has been consumed (handled by ``BatchIngestor.ingest()``).
    """

    def __init__(self, **kwargs: Any) -> None:
        import json
        from pathlib import Path

        self._json = json
        self._Path = Path

        self._lancedb_uri = str(kwargs.get("lancedb_uri", "lancedb"))
        self._table_name = str(kwargs.get("table_name", "nv-ingest"))
        self._overwrite = bool(kwargs.get("overwrite", True))
        self._embedding_column = str(kwargs.get("embedding_column", "text_embeddings_1b_v2"))
        self._embedding_key = str(kwargs.get("embedding_key", "embedding"))
        self._include_text = bool(kwargs.get("include_text", True))
        self._text_column = str(kwargs.get("text_column", "text"))

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

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = dict(kwargs)

        # If a remote NIM endpoint is configured, skip local model creation.
        endpoint = (kwargs.get("embedding_endpoint") or "").strip()
        if endpoint:
            self._model = None
            return

        from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder

        device = kwargs.get("device")
        hf_cache_dir = kwargs.get("hf_cache_dir")
        normalize = bool(kwargs.get("normalize", True))
        max_length = int(kwargs.get("max_length", 8192))
        # model_name may be a NIM alias (e.g. "nemo_retriever_v1") or a real HF
        # repo ID (e.g. "nvidia/llama-3.2-nv-embedqa-1b-v2"). Only forward it as
        # model_id when it looks like an HF repo (contains "/").
        model_name_raw = kwargs.get("model_name")
        model_id = model_name_raw if (isinstance(model_name_raw, str) and "/" in model_name_raw) else None

        self._model = LlamaNemotronEmbed1BV2Embedder(
            device=str(device) if device else None,
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
            normalize=normalize,
            max_length=max_length,
            model_id=model_id,
        )

    def __call__(self, batch_df: Any) -> Any:
        from retriever.ingest_modes.inprocess import embed_text_main_text_embed

        return embed_text_main_text_embed(batch_df, model=self._model, **self._kwargs)


class BatchIngestor(Ingestor):
    RUN_MODE = "batch"

    def __init__(self, documents: Optional[List[str]] = None, ray_address: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

        logging.basicConfig(level=logging.INFO)

        # Initialize Ray for distributed execution.
        ray.init(address=ray_address or "local", ignore_reinit_error=True)

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

    def extract(self, **kwargs: Any) -> "BatchIngestor":
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

        # -- Pop resource-tuning kwargs before forwarding to actors --
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

        # Reserve CPUs for GPU actors, then divide the rest among extract workers.
        total_gpu_cpus = (
            page_elements_workers * page_elements_cpus_per_actor
            + detect_workers * detect_stage_count * ocr_cpus_per_actor
        )
        cpus_for_extract = max(1, self._num_cpus - total_gpu_cpus)
        pdf_extract_workers = kwargs.pop("pdf_extract_workers", max(1, cpus_for_extract // 2))

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

        self._tasks.append(("extract", dict(kwargs)))

        # Stage-specific kwargs: upstream PDF stages accept many options (dpi, extract_*),
        # but downstream detect_* Ray actors accept only a small set. Passing the whole
        # dict can cause TypeErrors (e.g. unexpected `method=`).
        detect_passthrough_keys = {
            "inference_batch_size",
            "output_column",
            "num_detections_column",
            "counts_by_label_column",
        }
        detect_kwargs = {k: kwargs[k] for k in detect_passthrough_keys if k in kwargs}

        # Splitting pdfs is broken into a separate stage to help amortize downstream
        # processing if PDFs have vastly different numbers of pages.
        pdf_split_actor = PDFSplitActor(**kwargs)
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

    def embed(self, **kwargs: Any) -> "BatchIngestor":
        """
        Add a text-embedding stage to the batch pipeline.

        Uses a GPU actor pool so the HuggingFace model stays resident across
        batches.  Resource-tuning kwargs:

        - ``embed_workers``: ActorPool size (default 1).
        - ``embed_batch_size``: Ray Data batch size (default 256).
        - ``embed_cpus_per_actor``: CPUs reserved per embedding actor (default 1).
        - ``device``, ``hf_cache_dir``, ``normalize``, ``max_length``:
          forwarded to ``LlamaNemotronEmbed1BV2Embedder``.
        - ``embedding_endpoint``: optional NIM endpoint URL
          (e.g. ``"http://embedding:8000/v1"``).  When set, the actor
          delegates to the remote NIM instead of loading a local model,
          and no GPU is requested for this stage.
        """
        embed_workers = kwargs.pop("embed_workers", 1)
        embed_batch_size = kwargs.pop("embed_batch_size", 256)
        embed_cpus_per_actor = float(kwargs.pop("embed_cpus_per_actor", 1))

        # Remaining kwargs are forwarded to the actor constructor.
        self._tasks.append(("embed", dict(kwargs)))

        # Explode content rows before embedding so each table/chart/infographic
        # gets its own embedding vector (mirrors nv-ingest per-element embeddings).
        self._rd_dataset = self._rd_dataset.repartition(target_num_rows_per_block=256)

        from retriever.ingest_modes.inprocess import explode_content_to_rows

        self._rd_dataset = self._rd_dataset.map_batches(
            explode_content_to_rows,
            batch_size=embed_batch_size,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
        )

        # When using a remote NIM endpoint, no GPU is needed for embedding.
        endpoint = (kwargs.get("embedding_endpoint") or "").strip()
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
            fn_constructor_kwargs=dict(kwargs),
        )

        return self

    def vdb_upload(self, purge_results_after_upload: bool = True, **kwargs: Any) -> "BatchIngestor":
        """
        Add a streaming LanceDB upload stage to the batch pipeline.

        Instead of buffering the entire dataset into pandas, this adds a
        ``map_batches`` stage with a ``_LanceDBWriteActor`` that writes each
        batch to LanceDB as it streams through.  Index creation is deferred
        to ``ingest()`` (must happen after all writes).

        Accepts the same kwargs as
        ``inprocess.upload_embeddings_to_lancedb_inprocess``.
        """
        _ = purge_results_after_upload
        self._tasks.append(("vdb_upload", dict(kwargs)))
        self._vdb_upload_kwargs = dict(kwargs)

        # Streaming write stage — single actor, CPU-only, no GPU needed.
        self._rd_dataset = self._rd_dataset.map_batches(
            _LanceDBWriteActor,
            batch_format="pandas",
            num_cpus=1,
            num_gpus=0,
            compute=rd.ActorPoolStrategy(size=1),
            fn_constructor_kwargs=dict(kwargs),
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

    def ingest(self) -> int:
        """
        Execute the Ray Data pipeline and return the total number of pages.

        If a VDB upload stage was added (via ``vdb_upload()``), data is written
        to LanceDB in a streaming fashion by ``_LanceDBWriteActor``.  After the
        pipeline finishes, we create the LanceDB vector index (which must happen
        after all writes are complete).
        """
        t0 = time.monotonic()
        num_pages = self._rd_dataset.count()
        elapsed = time.monotonic() - t0

        print(f"[done] {len(self._input_documents)} files, {num_pages} pages in {elapsed:.1f}s")

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

        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=600))

        print(f"Wrote {n_vecs} rows to LanceDB uri={lancedb_uri!r} table={table_name!r}")
