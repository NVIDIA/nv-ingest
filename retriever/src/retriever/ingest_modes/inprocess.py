"""
In-process runmode.

Intended to run locally in a plain Python process with minimal assumptions
about surrounding framework/runtime.
"""

from __future__ import annotations

import glob
import json
import os
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from nemotron_page_elements_v3.model import define_model

import pandas as pd
from retriever.chart.chart_detection import detect_graphic_elements_v1_from_page_elements_v3
from retriever.infographic.infographic_detection import detect_infographic_elements_v1_from_page_elements_v3
from retriever.model.local import NemotronGraphicElementsV1, NemotronPageElementsV3, NemotronTableStructureV1
from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from retriever.page_elements import detect_page_elements_v3
from retriever.table.table_structure import detect_table_structure_v1_from_page_elements_v3
from retriever.text_embed.main_text_embed import TextEmbeddingConfig, create_text_embeddings_for_df

try:
    import pypdfium2 as pdfium
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]
    _PDFIUM_IMPORT_ERROR = e
else:  # pragma: no cover
    _PDFIUM_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from ..ingest import Ingestor
from ..pdf.extract import pdf_extraction


def embed_text_main_text_embed(
    batch_df: Any,
    *,
    model: Any,
    # Keep compatibility with previous `embed_text_1b_v2` signature so `.embed(...)` kwargs
    # don't need to change for inprocess mode.
    model_name: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    text_column: str = "text",
    inference_batch_size: int = 16,
    output_column: str = "text_embeddings_1b_v2",
    embedding_dim_column: str = "text_embeddings_1b_v2_dim",
    has_embedding_column: str = "text_embeddings_1b_v2_has_embedding",
    **_: Any,
) -> Any:
    """
    Inprocess embedding task implemented via `retriever.text_embed.main_text_embed`.

    This is a thin adapter that preserves the old output columns:
    - `output_column` (payload dict with embedding)
    - `embedding_dim_column` (int)
    - `has_embedding_column` (bool)

    It also writes `metadata.embedding` for compatibility with downstream uploaders.
    """
    _ = (model_name, embedding_endpoint)  # reserved for future remote execution support

    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_main_text_embed currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Build an embedder callable compatible with `create_text_embeddings_for_df`.
    def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vecs = model.embed(list(texts), batch_size=int(inference_batch_size))
        tolist = getattr(vecs, "tolist", None)
        if callable(tolist):
            return tolist()
        return vecs  # type: ignore[return-value]

    cfg = TextEmbeddingConfig(
        text_column=str(text_column),
        # Generate the same payload column name as the prior implementation.
        output_payload_column=str(output_column) if output_column else None,
        write_embedding_to_metadata=True,
        metadata_column="metadata",
        # Match chunking behavior as closely as possible to previous `embed_text_1b_v2`.
        batch_size=int(inference_batch_size),
        encoding_format="float",
        input_type="passage",
        truncate="END",
        dimensions=None,
        embedding_nim_endpoint="http://localhost:8012/v1",
        embedding_model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    )


    # # Retriever-local dataframe settings
    # text_column: str = "text"
    # write_embedding_to_metadata: bool = True
    # metadata_column: str = "metadata"
    # # Optional extra output column containing a payload dict (similar to embed_text_1b_v2)
    # output_payload_column: Optional[str] = None

    try:
        out_df, _info = create_text_embeddings_for_df(
            batch_df,
            task_config={
                "embedder": _embed,
                "endpoint_url": "http://localhost:8012/v1",
                # "endpoint_url": None, # inprocess uses local HF embedder
                "local_batch_size": int(inference_batch_size),
            },
            transform_config=cfg,
        )
    except BaseException as e:
        # Fail-soft: preserve batch shape and set an error payload per-row.
        err_payload = {"embedding": None, "error": {"stage": "embed", "type": e.__class__.__name__, "message": str(e)}}
        out_df = batch_df.copy()
        if output_column:
            out_df[output_column] = [err_payload for _ in range(len(out_df.index))]
        out_df[embedding_dim_column] = [0 for _ in range(len(out_df.index))]
        out_df[has_embedding_column] = [False for _ in range(len(out_df.index))]
        out_df["_contains_embeddings"] = [False for _ in range(len(out_df.index))]
        return out_df

    # Add backwards-compatible dim/has columns (these were produced by `embed_text_1b_v2`).
    if embedding_dim_column:
        def _dim(row: pd.Series) -> int:
            md = row.get("metadata")
            if isinstance(md, dict):
                emb = md.get("embedding")
                if isinstance(emb, list):
                    return int(len(emb))
            payload = row.get(output_column) if output_column else None
            if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
                return int(len(payload.get("embedding") or []))
            return 0

        out_df[embedding_dim_column] = out_df.apply(_dim, axis=1)
    else:
        out_df[embedding_dim_column] = [0 for _ in range(len(out_df.index))]

    out_df[has_embedding_column] = [bool(int(d) > 0) for d in out_df[embedding_dim_column].tolist()]
    return out_df


def _to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects into JSON-serializable types.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # Common numeric scalar types (numpy, pandas).
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    # Common array/tensor types (numpy, torch).
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    # Pandas timestamps.
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass

    return str(obj)


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)


def _safe_stem(name: str) -> str:
    s = str(name or "").strip() or "document"
    s = os.path.splitext(os.path.basename(s))[0] or "document"
    # Keep it filesystem-friendly.
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:160] if len(s) > 160 else s


def save_dataframe_to_disk_json(df: Any, *, output_directory: str) -> Any:
    """
    Pipeline task: persist a pandas DataFrame as a timestamped JSON file.

    Returns the input unchanged so it can stay in the pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"save_dataframe_to_disk_json expects pandas.DataFrame, got {type(df)!r}")
    if not isinstance(output_directory, str) or not output_directory.strip():
        raise ValueError("output_directory must be a non-empty string")

    out_dir = os.path.abspath(output_directory)
    os.makedirs(out_dir, exist_ok=True)

    # Try to derive a human-friendly filename prefix from the source PDF path.
    source_path = None
    try:
        if "metadata" in df.columns and len(df.index) > 0:
            md = df.iloc[0]["metadata"]
            if isinstance(md, dict):
                source_path = md.get("source_path")
    except Exception:
        source_path = None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = _safe_stem(str(source_path) if source_path else "results")
    uniq = uuid.uuid4().hex[:8]
    out_path = os.path.join(out_dir, f"{stem}.{ts}.{uniq}.json")

    payload = {
        "schema_version": 1,
        "run_mode": "inprocess",
        "created_at_utc": ts,
        "source_path": str(source_path) if source_path else None,
        "rows": int(len(df.index)),
        "columns": [str(c) for c in df.columns.tolist()],
        "records": _to_jsonable(df.to_dict(orient="records")),
    }

    _atomic_write_json(out_path, payload)
    return df


def _extract_embedding_from_row(
    row: pd.Series,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
) -> Optional[List[float]]:
    """
    Extract an embedding vector from a row.

    Supports:
    - `metadata.embedding` (preferred if present)
    - `embedding_column` payloads like `{"embedding": [...], ...}` (from `embed_text_1b_v2`)
    """
    meta = row.get("metadata")
    if isinstance(meta, dict):
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]

    payload = row.get(embedding_column)
    if isinstance(payload, dict):
        emb = payload.get(embedding_key)
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]
    return None


def _extract_source_path_and_page(row: pd.Series) -> Tuple[str, int]:
    """
    Best-effort extract of source path and page number for LanceDB row metadata.
    """
    path = ""
    page = -1

    v = row.get("path")
    if isinstance(v, str) and v.strip():
        path = v.strip()

    v = row.get("page_number")
    try:
        if v is not None:
            page = int(v)
    except Exception:
        pass

    meta = row.get("metadata")
    if isinstance(meta, dict):
        sp = meta.get("source_path")
        if isinstance(sp, str) and sp.strip():
            path = sp.strip()
        # Some schemas store page under content metadata; support if present.
        cm = meta.get("content_metadata")
        if isinstance(cm, dict) and page == -1:
            h = cm.get("hierarchy")
            if isinstance(h, dict) and "page" in h:
                try:
                    page = int(h.get("page"))
                except Exception:
                    pass

    return path, page


def upload_embeddings_to_lancedb_inprocess(
    df: Any,
    *,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
    overwrite: bool = True,
    create_index: bool = True,
    index_type: str = "IVF_HNSW_SQ",
    metric: str = "l2",
    num_partitions: int = 16,
    num_sub_vectors: int = 256,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
    include_text: bool = True,
    text_column: str = "text",
) -> Any:
    """
    Pipeline task: upload embeddings from a pandas DataFrame into LanceDB.

    The embedding is sourced from:
    - `metadata.embedding` if present, else
    - `embedding_column` payloads like those produced by `embed_text_1b_v2`

    Parameters (all can be passed via `.vdb_upload(...)` kwargs):
    - **lancedb_uri**: LanceDB URI (directory path), default `"lancedb"`
    - **table_name**: table name, default `"nv-ingest"`
    - **overwrite**: overwrite table (True) or append (False)
    - **create_index**: create vector index after upload
    - **index_type**: LanceDB index type (e.g. `"IVF_HNSW_SQ"`)
    - **metric**: distance metric (e.g. `"l2"`, `"cosine"`)
    - **num_partitions**: index partitions
    - **num_sub_vectors**: index sub-vectors
    - **embedding_column**: column name containing embedding payload dicts
    - **embedding_key**: key inside payload dict to read embedding list from
    - **include_text**: if True, store `text_column` content alongside the vector
    - **text_column**: column to read text from when `include_text=True`

    Returns the input DataFrame unchanged (so it can remain in the pipeline).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"upload_embeddings_to_lancedb_inprocess expects pandas.DataFrame, got {type(df)!r}")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        emb = _extract_embedding_from_row(r, embedding_column=str(embedding_column), embedding_key=str(embedding_key))
        if emb is None:
            continue

        path, page_number = _extract_source_path_and_page(r)
        p = Path(path) if path else None
        filename = p.name if p is not None else ""
        pdf_basename = p.stem if p is not None else ""
        pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""
        source_id = path or filename or pdf_basename

        # Provide fields compatible with `retriever.recall.core` which expects LanceDB hits
        # to include JSON-encoded `metadata` and `source` strings.
        metadata_obj: Dict[str, Any] = {"page_number": int(page_number) if page_number is not None else -1}
        if pdf_page:
            metadata_obj["pdf_page"] = pdf_page
        source_obj: Dict[str, Any] = {"source_id": str(path)}

        row_out: Dict[str, Any] = {
            "vector": emb,
            "pdf_page": pdf_page,
            "filename": filename,
            "pdf_basename": pdf_basename,
            "page_number": int(page_number) if page_number is not None else -1,
            "source_id": str(source_id),
            "path": str(path),
            "metadata": json.dumps(metadata_obj, ensure_ascii=False),
            "source": json.dumps(source_obj, ensure_ascii=False),
        }

        if include_text:
            t = r.get(text_column)
            row_out["text"] = str(t) if isinstance(t, str) else ""
        else:
            # Still include the column for compatibility with the recall script's `.select(["text",...])`.
            row_out["text"] = ""

        rows.append(row_out)

    if not rows:
        print("No embeddings found to upload to LanceDB (no rows had embeddings).")
        return df

    # Infer vector dim from first row.
    dim = 0
    for rr in rows:
        v = rr.get("vector")
        if isinstance(v, list) and v:
            dim = int(len(v))
            break
    if dim <= 0:
        raise ValueError("Failed to infer embedding dimension from DataFrame rows.")

    try:
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LanceDB upload requested but dependencies are missing. Install `lancedb` and `pyarrow`."
        ) from e

    db = lancedb.connect(uri=str(lancedb_uri))

    fields = [
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("pdf_page", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("pdf_basename", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("source_id", pa.string()),
        pa.field("path", pa.string()),
        # Compatibility columns expected by `retriever.recall.core`:
        pa.field("text", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("source", pa.string()),
    ]
    schema = pa.schema(fields)

    # Overwrite vs append.
    if overwrite:
        table = db.create_table(str(table_name), data=list(rows), schema=schema, mode="overwrite")
    else:
        try:
            table = db.open_table(str(table_name))
            table.add(list(rows))
        except Exception:
            table = db.create_table(str(table_name), data=list(rows), schema=schema, mode="create")

    if create_index:
        # LanceDB IVF-based indexes train k-means with K=num_partitions. K must be < N vectors.
        n_vecs = int(len(rows))
        if n_vecs < 2:
            print("Skipping LanceDB index creation (not enough vectors).")
        else:
            k = int(num_partitions)
            if k >= n_vecs:
                k = max(1, n_vecs - 1)
            try:
                table.create_index(
                    index_type=str(index_type),
                    metric=str(metric),
                    num_partitions=int(k),
                    num_sub_vectors=int(num_sub_vectors),
                    vector_column_name="vector",
                )
            except TypeError:
                # Older/newer LanceDB versions may have different signatures; fall back to minimal call.
                table.create_index(vector_column_name="vector")
            except Exception as e:
                # Don't fail ingestion due to index training; users can rebuild later with more data.
                print(f"Warning: failed to create LanceDB index (continuing without index): {e}")

    print(f"Wrote {len(rows)} rows to LanceDB uri={lancedb_uri!r} table={table_name!r}")
    return df


class InProcessIngestor(Ingestor):
    RUN_MODE = "inprocess"

    def __init__(self, documents: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

        # Keep backwards-compatibility with code that inspects `Ingestor._documents`
        # by ensuring both names refer to the same list.
        self._input_documents: List[str] = self._documents

        # Builder-style configuration recorded for later execution (TBD).
        self._tasks: List[tuple[Callable[..., Any], dict[str, Any]]] = []

    def files(self, documents: Union[str, List[str]]) -> "InProcessIngestor":
        """
        Add local files for in-process execution.

        Mirrors `BatchIngestor.files()` path/glob resolution, but does not create
        a Ray Data dataset.
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

        return self

    def extract(self, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure extraction for in-process execution (skeleton).

        TODO: implement actual local extraction (PDF split/extract, page elements,
        table structure, chart detection, etc.). For now this records the
        configuration so call sites can be wired up and chained.
        """
        # NOTE: `kwargs` passed to `.extract()` are intended primarily for PDF extraction
        # (e.g. `extract_text`, `dpi`, etc). Downstream model stages do NOT necessarily
        # accept the same keyword arguments. Keep per-stage kwargs isolated.
        print(f"Type kwargs: {type(kwargs)}")
        print(f"kwargs: {kwargs}")

        extract_kwargs = dict(kwargs)
        # Downstream in-process stages (page elements / table / chart / infographic) assume
        # `page_image.image_b64` exists. Ensure PDF extraction emits a page image unless
        # the caller explicitly disables it.
        if "extract_page_as_image" not in extract_kwargs:
            if any(extract_kwargs.get(k) is True for k in ("extract_text", "extract_images", "extract_tables", "extract_charts", "extract_infographics")):
                extract_kwargs["extract_page_as_image"] = True
        self._tasks.append((pdf_extraction, extract_kwargs))

        # Common, optional knobs shared by our detect_* helpers.
        detect_passthrough_keys = {
            "inference_batch_size",
            "output_column",
            "num_detections_column",
            "counts_by_label_column",
        }

        def _detect_kwargs_with_model(model_obj: Any) -> dict[str, Any]:
            d: dict[str, Any] = {"model": model_obj}
            for k in detect_passthrough_keys:
                if k in kwargs:
                    d[k] = kwargs[k]
            return d

        # NOTE: Page element detection is a common prerequisite for downstream
        # structure stages (tables/charts/infographics). We enable it whenever
        # any downstream extraction is requested.
        if any(
            kwargs.get(k) is True for k in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
        ):
            print("Adding page elements task")
            self._tasks.append((detect_page_elements_v3, _detect_kwargs_with_model(define_model("page_element_v3"))))

        if kwargs.get("extract_tables") is True:
            print("Adding table structure task")
            # Run table structure only on cropped "table" regions from page-elements.
            self._tasks.append(
                (detect_table_structure_v1_from_page_elements_v3, _detect_kwargs_with_model(NemotronTableStructureV1()))
            )

        if kwargs.get("extract_charts") is True:
            print("Adding chart detection task")
            # Run chart detection only on cropped "chart" regions from page-elements.
            self._tasks.append(
                (detect_graphic_elements_v1_from_page_elements_v3, _detect_kwargs_with_model(NemotronGraphicElementsV1()))
            )

        if kwargs.get("extract_infographics") is True:
            print("Adding infographic detection task")
            # Run infographic detection only on cropped "infographic"/"title" regions from page-elements.
            self._tasks.append(
                (
                    detect_infographic_elements_v1_from_page_elements_v3,
                    _detect_kwargs_with_model(NemotronGraphicElementsV1()),
                )
            )

        return self


    def embed(self, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure embedding for in-process execution (skeleton).

        This records an embedding task so call sites can chain `.embed(...)`
        after `.extract(...)`. The current implementation is a no-op placeholder.
        """
        # NOTE: inprocess mode uses a local HF embedder (no microservice).
        # Allow callers to control device / max_length to avoid OOMs.
        embed_kwargs = dict(kwargs)
        device = embed_kwargs.pop("device", None)
        hf_cache_dir = embed_kwargs.pop("hf_cache_dir", None)
        normalize = bool(embed_kwargs.pop("normalize", True))
        max_length = int(embed_kwargs.pop("max_length", 4096))

        embed_kwargs.setdefault("input_type", "passage")
        embed_kwargs["model"] = LlamaNemotronEmbed1BV2Embedder(
            device=str(device) if device is not None else None,
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            normalize=normalize,
            max_length=max_length,
        )
        self._tasks.append((embed_text_main_text_embed, embed_kwargs))
        return self

    def save_to_disk(
        self,
        output_directory: Optional[str] = None,
        cleanup: bool = True,
        compression: Optional[str] = "gzip",
    ) -> "InProcessIngestor":
        """
        Persist the current per-document DataFrame to disk as JSON.

        Writes one JSON file per input document under `output_directory`, using a
        UTC timestamp in the filename.
        """
        _ = (cleanup, compression)  # reserved for future use (parity with interface)
        if output_directory is None:
            raise ValueError("output_directory is required for inprocess save_to_disk()")

        self._tasks.append((save_dataframe_to_disk_json, {"output_directory": str(output_directory)}))
        return self

    def vdb_upload(self, purge_results_after_upload: bool = True, **kwargs: Any) -> "InProcessIngestor":
        """
        Upload the (embedding-enriched) results to a vector DB (LanceDB).

        This is an in-process uploader intended to run after `.embed(...)`.
        It reads embeddings from either:
        - `metadata.embedding` (if present), or
        - the embedding payload column produced by the embed stage (default: `text_embeddings_1b_v2`)

        Configuration is passed via kwargs:
        - **lancedb_uri**: str, default `"lancedb"`
        - **table_name**: str, default `"nv-ingest"`
        - **overwrite**: bool, default True (False appends)
        - **create_index**: bool, default True
        - **index_type**: str, default `"IVF_HNSW_SQ"`
        - **metric**: str, default `"l2"`
        - **num_partitions**: int, default 16
        - **num_sub_vectors**: int, default 256
        - **embedding_column**: str, default `"text_embeddings_1b_v2"`
        - **embedding_key**: str, default `"embedding"`
        - **include_text**: bool, default False
        - **text_column**: str, default `"text"`

        Notes:
        - `purge_results_after_upload` is accepted for API parity but is not used in inprocess mode.
        """
        _ = purge_results_after_upload  # parity with interface; inprocess does not purge by default
        self._tasks.append((upload_embeddings_to_lancedb_inprocess, dict(kwargs)))
        return self


    def ingest(
        self,
        show_progress: bool = False,
        return_failures: bool = False,
        save_to_disk: bool = False,
        return_traces: bool = False,
        **_: Any,
    ) -> list[Any]:

        if pdfium is None:  # pragma: no cover
            raise ImportError("pypdfium2 is required for inprocess ingestion.") from _PDFIUM_IMPORT_ERROR

        def _pdf_to_pages_df(path: str) -> pd.DataFrame:
            """
            Convert a multi-page PDF at `path` into a DataFrame where each row
            contains a *single-page* PDF's raw bytes.

            Columns:
            - bytes: single-page PDF bytes
            - path: original input path
            - page_number: 1-indexed page number
            """
            abs_path = os.path.abspath(path)
            out_rows: list[dict[str, Any]] = []
            doc = None
            try:
                doc = pdfium.PdfDocument(abs_path)
                for page_idx in range(len(doc)):
                    single = pdfium.PdfDocument.new()
                    try:
                        single.import_pages(doc, pages=[page_idx])
                        buf = BytesIO()
                        single.save(buf)
                        out_rows.append(
                            {
                                "bytes": buf.getvalue(),
                                "path": abs_path,
                                "page_number": page_idx + 1,
                            }
                        )
                    finally:
                        try:
                            single.close()
                        except Exception:
                            pass
            except BaseException as e:
                # Preserve shape expected downstream (pdf_extraction emits error
                # records per-row, so we return a single row to trigger that).
                out_rows.append({"bytes": b"", "path": abs_path, "page_number": 0, "error": str(e)})
            finally:
                try:
                    if doc is not None:
                        doc.close()
                except Exception:
                    pass

            return pd.DataFrame(out_rows)

        # Iterate through all configured documents; for each file build a per-page
        # DataFrame and run pdf_extraction on it.
        results: list[Any] = []
        docs = list(self._documents)
        doc_iter = docs
        if show_progress and tqdm is not None:
            doc_iter = tqdm(docs, desc="Processing documents", unit="doc")

        for doc_path in doc_iter:
            pages_df = _pdf_to_pages_df(doc_path)

            # If the pipeline was configured via .extract(...), use those kwargs.
            current: Any = pages_df
            for func, kwargs in self._tasks:
                if func is pdf_extraction:
                    current = func(pdf_binary=current, **kwargs)
                else:
                    current = func(current, **kwargs)
            results.append(current)

        _ = (show_progress, return_failures, save_to_disk, return_traces)  # reserved for future use
        return results