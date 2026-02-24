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
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from nemotron_page_elements_v3.model import define_model

import pandas as pd
from retriever.chart.chart_detection import detect_graphic_elements_v1_from_page_elements_v3
from retriever.model.local import NemotronGraphicElementsV1, NemotronOCRV1, NemotronPageElementsV3
from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from retriever.page_elements import detect_page_elements_v3
from retriever.ocr.ocr import ocr_page_elements
from retriever.text_embed.main_text_embed import TextEmbeddingConfig, create_text_embeddings_for_df

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from ..convert import SUPPORTED_EXTENSIONS, convert_to_pdf_bytes
from ..ingest import Ingestor
from ..pdf.extract import pdf_extraction
from ..pdf.split import _split_pdf_to_single_page_bytes, pdf_path_to_pages_df
from ..txt import txt_file_to_chunks_df

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _combine_text_with_content(row, text_column, content_columns):
    """Combine page text with OCR content text for embedding."""
    parts = []
    base = row.get(text_column)
    if isinstance(base, str) and base.strip():
        parts.append(base.strip())
    for col in content_columns:
        content_list = row.get(col)
        if isinstance(content_list, list):
            for item in content_list:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
    return "\n\n".join(parts) if parts else ""


def _deep_copy_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy a row dict but deep-copy any nested mutable values.

    This avoids the problem where multiple exploded rows share the same
    metadata dict reference, causing later embedding writes to overwrite
    earlier ones.  We only deep-copy dicts and lists (the common mutable
    types in row data); scalars and strings are immutable and safe to share.
    """
    import copy

    out: Dict[str, Any] = {}
    for k, v in row_dict.items():
        if isinstance(v, (dict, list)):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = v
    return out


def explode_content_to_rows(
    batch_df: Any,
    *,
    text_column: str = "text",
    content_columns: Sequence[str] = _CONTENT_COLUMNS,
) -> Any:
    """Expand each page row into multiple rows for per-element embedding.

    For each row in *batch_df*:
    - One row with the original page text (``text_column``).
    - One additional row per table / chart / infographic item whose
      ``"text"`` field is non-empty.  The item text replaces
      ``text_column`` so the downstream embedding stage embeds each
      element independently.

    This mirrors the nv-ingest pipeline where every structural element
    gets its own embedding vector, improving recall for queries that
    target specific tables or charts.

    If a row has no page text *and* no structured content, the original
    row is preserved as-is to maintain batch shape.
    """
    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    # Fast path: if none of the content columns exist there is nothing to explode.
    if not any(c in batch_df.columns for c in content_columns):
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        # Row for page text.
        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            new_rows.append(_deep_copy_row(row_dict))
            exploded_any = True

        # One row per structured content item.
        for col in content_columns:
            content_list = row_dict.get(col)
            if not isinstance(content_list, list):
                continue
            for item in content_list:
                if not isinstance(item, dict):
                    continue
                t = item.get("text", "")
                if not isinstance(t, str) or not t.strip():
                    continue
                content_row = _deep_copy_row(row_dict)
                content_row[text_column] = t.strip()
                new_rows.append(content_row)
                exploded_any = True

        # Preserve row if nothing was exploded (no text, no content).
        if not exploded_any:
            new_rows.append(row_dict)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def embed_text_main_text_embed(
    batch_df: Any,
    *,
    model: Any = None,
    # Keep compatibility with previous `embed_text_1b_v2` signature so `.embed(...)` kwargs
    # don't need to change for inprocess mode.
    model_name: Optional[str] = None,
    embedding_endpoint: Optional[str] = None,
    embed_invoke_url: Optional[str] = None,
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

    When *embedding_endpoint* is set (e.g. ``"http://embedding:8000/v1"``), a remote
    NIM endpoint is used instead of the local HF model.  The NIM handles prefixing
    (``input_type="passage"``) server-side.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_main_text_embed currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Resolve endpoint: strip whitespace, treat empty string as None.
    _endpoint = (embedding_endpoint or embed_invoke_url or "").strip() or None

    if _endpoint is None and model is None:
        raise ValueError("Either a local model or an embedding_endpoint must be provided.")

    # Build an embedder callable compatible with `create_text_embeddings_for_df`.
    # Only used when running with a local model (no NIM endpoint).
    _embed = None
    if _endpoint is None and model is not None:

        def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            prefixed = [f"passage: {t}" for t in texts]
            vecs = model.embed(prefixed, batch_size=int(inference_batch_size))
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
        embedding_nim_endpoint=_endpoint or "http://localhost:8012/v1",
        embedding_model=model_name or "nvidia/llama-3.2-nv-embedqa-1b-v2",
    )

    # Rows should already be exploded (one row per page text / table / chart)
    # by ``explode_content_to_rows`` before reaching this stage.
    embed_df = batch_df

    try:
        out_df, _info = create_text_embeddings_for_df(
            embed_df,
            task_config={
                "embedder": _embed,
                "endpoint_url": _endpoint,
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
    row: Any,
    *,
    embedding_column: str = "text_embeddings_1b_v2",
    embedding_key: str = "embedding",
) -> Optional[List[float]]:
    """
    Extract an embedding vector from a row (namedtuple or pd.Series).

    Supports:
    - `metadata.embedding` (preferred if present)
    - `embedding_column` payloads like `{"embedding": [...], ...}` (from `embed_text_1b_v2`)
    """
    meta = getattr(row, "metadata", None)
    if isinstance(meta, dict):
        emb = meta.get("embedding")
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]

    payload = getattr(row, embedding_column, None)
    if isinstance(payload, dict):
        emb = payload.get(embedding_key)
        if isinstance(emb, list) and emb:
            return emb  # type: ignore[return-value]
    return None


def _extract_source_path_and_page(row: Any) -> Tuple[str, int]:
    """
    Best-effort extract of source path and page number for LanceDB row metadata.
    """
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

    meta = getattr(row, "metadata", None)
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
    for r in df.itertuples(index=False):
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
            t = getattr(r, text_column, None)
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

        # Pipeline type: "pdf" (extract) or "txt" (extract_txt). Loader dispatch in ingest().
        self._pipeline_type: Literal["pdf", "txt"] = "pdf"
        self._extract_txt_kwargs: Dict[str, Any] = {}

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
            if any(
                extract_kwargs.get(k) is True
                for k in ("extract_text", "extract_images", "extract_tables", "extract_charts", "extract_infographics")
            ):
                extract_kwargs["extract_page_as_image"] = True
        self._pipeline_type = "pdf"
        self._tasks.append((pdf_extraction, extract_kwargs))

        # Common, optional knobs shared by our detect_* helpers.
        detect_passthrough_keys = {
            "inference_batch_size",
            "output_column",
            "num_detections_column",
            "counts_by_label_column",
        }

        def _stage_remote_kwargs(stage_name: str) -> dict[str, Any]:
            stage_prefix = f"{stage_name}_"
            out: dict[str, Any] = {}
            invoke_url = kwargs.get(f"{stage_prefix}invoke_url", kwargs.get("invoke_url"))
            if invoke_url:
                out["invoke_url"] = invoke_url
            api_key = kwargs.get(f"{stage_prefix}api_key", kwargs.get("api_key"))
            if api_key:
                out["api_key"] = api_key
            timeout = kwargs.get(f"{stage_prefix}request_timeout_s", kwargs.get("request_timeout_s"))
            if timeout is not None:
                out["request_timeout_s"] = timeout
            for k in ("remote_max_pool_workers", "remote_max_retries", "remote_max_429_retries"):
                stage_key = f"{stage_prefix}{k}"
                if stage_key in kwargs:
                    out[k] = kwargs[stage_key]
                elif k in kwargs:
                    out[k] = kwargs[k]
            return out

        def _detect_kwargs_with_model(model_obj: Any, *, stage_name: str, allow_remote: bool) -> dict[str, Any]:
            d: dict[str, Any] = {"model": model_obj}
            for k in detect_passthrough_keys:
                if k in kwargs:
                    d[k] = kwargs[k]
            if allow_remote:
                d.update(_stage_remote_kwargs(stage_name))
            return d

        # NOTE: Page element detection is a common prerequisite for downstream
        # structure stages (tables/charts/infographics). We enable it whenever
        # any downstream extraction is requested.
        if any(
            kwargs.get(k) is True for k in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
        ):
            print("Adding page elements task")
            self._tasks.append(
                (
                    detect_page_elements_v3,
                    _detect_kwargs_with_model(
                        define_model("page_element_v3"),
                        stage_name="page_elements",
                        allow_remote=True,
                    ),
                )
            )

        # OCR-based extraction for tables/charts/infographics.
        ocr_flags = {}
        if kwargs.get("extract_tables") is True:
            ocr_flags["extract_tables"] = True
        if kwargs.get("extract_charts") is True:
            ocr_flags["extract_charts"] = True
        if kwargs.get("extract_infographics") is True:
            ocr_flags["extract_infographics"] = True
        ocr_flags.update(_stage_remote_kwargs("ocr"))

        if ocr_flags:
            print("Adding OCR extraction task")
            ocr_model_dir = os.environ.get("NEMOTRON_OCR_MODEL_DIR", "")
            if not ocr_model_dir:
                raise RuntimeError(
                    "NEMOTRON_OCR_MODEL_DIR environment variable must be set to "
                    "the path of the Nemotron OCR v1 model directory."
                )
            self._tasks.append((ocr_page_elements, {"model": NemotronOCRV1(model_dir=ocr_model_dir), **ocr_flags}))

        return self

    def extract_txt(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 0,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> "InProcessIngestor":
        """
        Configure txt ingestion: tokenizer-based chunking only (no PDF extraction).

        Use with .files("*.txt").extract_txt(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_txt().
        """
        self._pipeline_type = "txt"
        self._extract_txt_kwargs = {
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "encoding": encoding,
            **kwargs,
        }
        return self

    def embed(self, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure embedding for in-process execution.

        This records an embedding task so call sites can chain `.embed(...)`
        after `.extract(...)` or `.extract_txt()`.

        When ``embedding_endpoint`` (or ``embed_invoke_url``) is provided (e.g.
        ``"http://embedding:8000/v1"``), a remote NIM endpoint is used for
        embedding instead of the local HF model.
        """
        # Explode content rows before embedding so each table/chart/infographic
        # gets its own embedding vector (mirrors nv-ingest per-element embeddings).
        self._tasks.append((explode_content_to_rows, {}))

        embed_kwargs = dict(kwargs)
        if "embedding_endpoint" not in embed_kwargs and embed_kwargs.get("embed_invoke_url"):
            embed_kwargs["embedding_endpoint"] = embed_kwargs.get("embed_invoke_url")

        # If a remote NIM endpoint is configured, skip local model creation.
        endpoint = (embed_kwargs.get("embedding_endpoint") or embed_kwargs.get("embed_invoke_url") or "").strip()
        if endpoint:
            embed_kwargs.setdefault("input_type", "passage")
            self._tasks.append((embed_text_main_text_embed, embed_kwargs))
            return self

        # Local HF embedder path.
        # Allow callers to control device / max_length to avoid OOMs.
        device = embed_kwargs.pop("device", None)
        hf_cache_dir = embed_kwargs.pop("hf_cache_dir", None)
        normalize = bool(embed_kwargs.pop("normalize", True))
        max_length = int(embed_kwargs.pop("max_length", 8192))

        # model_name may be a NIM alias (e.g. "nemo_retriever_v1") or a real HF
        # repo ID (e.g. "nvidia/llama-3.2-nv-embedqa-1b-v2"). Only forward it as
        # model_id when it looks like an HF repo (contains "/").
        model_name_raw = embed_kwargs.pop("model_name", None)
        model_id = model_name_raw if (isinstance(model_name_raw, str) and "/" in model_name_raw) else None

        embed_kwargs.setdefault("input_type", "passage")
        embed_kwargs["model"] = LlamaNemotronEmbed1BV2Embedder(
            device=str(device) if device is not None else None,
            hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            normalize=normalize,
            max_length=max_length,
            model_id=model_id,
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
        # Tasks that run once on combined results (after all docs). All others run per-doc.
        _post_tasks = (upload_embeddings_to_lancedb_inprocess, save_dataframe_to_disk_json)
        per_doc_tasks = [(f, k) for f, k in self._tasks if f not in _post_tasks]
        post_tasks = [(f, k) for f, k in self._tasks if f in _post_tasks]

        # Iterate through all configured documents; for each file build a per-page
        # DataFrame and run the per-doc task chain.
        results: list[Any] = []
        docs = list(self._documents)
        doc_iter = docs
        if show_progress and tqdm is not None:
            doc_iter = tqdm(docs, desc="Processing documents", unit="doc")

        if self._pipeline_type == "pdf":

            def _loader(p: str) -> pd.DataFrame:
                """
                Load a document as a per-page DataFrame. For .pdf use pdf_path_to_pages_df.
                For .docx/.pptx convert to PDF via LibreOffice then split (same schema).
                """
                abs_path = os.path.abspath(p)
                ext = os.path.splitext(abs_path)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and ext != ".pdf":
                    try:
                        with open(abs_path, "rb") as f:
                            file_bytes = f.read()
                        pdf_bytes = convert_to_pdf_bytes(file_bytes, ext)
                        pages = _split_pdf_to_single_page_bytes(pdf_bytes)
                        out_rows = [
                            {"bytes": b, "path": abs_path, "page_number": i + 1}
                            for i, b in enumerate(pages)
                        ]
                        return pd.DataFrame(out_rows)
                    except BaseException as e:
                        return pd.DataFrame(
                            [{"bytes": b"", "path": abs_path, "page_number": 0, "error": str(e)}]
                        )
                return pdf_path_to_pages_df(p)

        else:
            def _loader(p: str) -> pd.DataFrame:
                return txt_file_to_chunks_df(p, **self._extract_txt_kwargs)

        for doc_path in doc_iter:
            initial_df = _loader(doc_path)

            current: Any = initial_df
            for func, kwargs in per_doc_tasks:
                if func is pdf_extraction:
                    current = func(pdf_binary=current, **kwargs)
                else:
                    current = func(current, **kwargs)
            results.append(current)

        # Run upload/save once on combined results so overwrite=True keeps full corpus.
        if post_tasks and results and all(isinstance(r, pd.DataFrame) for r in results):
            combined = pd.concat(results, ignore_index=True)
            for func, kwargs in post_tasks:
                combined = func(combined, **kwargs)

        _ = (show_progress, return_failures, save_to_disk, return_traces)  # reserved for future use
        return results
