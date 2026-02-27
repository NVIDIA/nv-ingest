# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-process runmode.

Intended to run locally in a plain Python process with minimal assumptions
about surrounding framework/runtime.
"""

from __future__ import annotations

import glob
import json
import multiprocessing
import os
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


import pandas as pd
from nemo_retriever.model.local import NemotronOCRV1, NemotronPageElementsV3
from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from nemo_retriever.page_elements import detect_page_elements_v3
from nemo_retriever.ocr.ocr import _crop_b64_image_by_norm_bbox, ocr_page_elements
from nemo_retriever.text_embed.main_text_embed import TextEmbeddingConfig, create_text_embeddings_for_df

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    import pypdfium2 as pdfium
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore[assignment]
    _PDFIUM_IMPORT_ERROR = e

from ..utils.convert import SUPPORTED_EXTENSIONS, convert_to_pdf_bytes
from ..ingestor import Ingestor
from ..params import EmbedParams
from ..params.models import IMAGE_MODALITIES
from ..params import ExtractParams
from ..params import HtmlChunkParams
from ..params import IngestExecuteParams
from ..params import TextChunkParams
from ..params import VdbUploadParams
from ..pdf.extract import pdf_extraction
from ..pdf.split import _split_pdf_to_single_page_bytes, pdf_path_to_pages_df
from ..txt import txt_file_to_chunks_df
from ..html import html_file_to_chunks_df

_CONTENT_COLUMNS = ("table", "chart", "infographic")


def _coerce_params[T](params: T | None, model_cls: type[T], kwargs: dict[str, Any]) -> T:
    if params is None:
        return model_cls(**kwargs)
    if kwargs:
        return params.model_copy(update=kwargs)  # type: ignore[return-value]
    return params


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
    modality: str = "text",
    text_elements_modality: Optional[str] = None,
    structured_elements_modality: Optional[str] = None,
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

    When a row's effective modality is ``"image"`` or ``"text_image"``,
    the page image (or a cropped region for structured content) is
    carried in a new ``_image_b64`` column so that the downstream
    embedding stage can use it for multimodal embeddings.

    Each exploded row is tagged with ``_embed_modality`` so that
    ``embed_text_main_text_embed`` can embed different element types
    with different methods (e.g. text-only for page text but
    text+image for tables).

    Parameters
    ----------
    modality : str
        Base modality for all element types (default ``"text"``).
    text_elements_modality : str | None
        Override modality for page-text rows.  Falls back to *modality*.
    structured_elements_modality : str | None
        Override modality for table/chart/infographic rows.  Falls back
        to *modality*.
    """
    text_mod = text_elements_modality or modality
    struct_mod = structured_elements_modality or modality

    if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
        return batch_df

    # Whether *any* element type requires images.
    any_images = text_mod in IMAGE_MODALITIES or struct_mod in IMAGE_MODALITIES

    # Fast path: if none of the content columns exist there is nothing to explode.
    if not any(c in batch_df.columns for c in content_columns):
        batch_df = batch_df.copy()
        if text_mod in IMAGE_MODALITIES and "page_image" in batch_df.columns:
            batch_df["_image_b64"] = batch_df["page_image"].apply(
                lambda pi: pi.get("image_b64") if isinstance(pi, dict) else None
            )
        batch_df["_embed_modality"] = text_mod
        return batch_df

    new_rows: List[Dict[str, Any]] = []
    for _, row in batch_df.iterrows():
        row_dict = row.to_dict()
        exploded_any = False

        # Extract page-level image b64 once per source row.
        page_image = row_dict.get("page_image")
        page_image_b64: Optional[str] = None
        if any_images and isinstance(page_image, dict):
            page_image_b64 = page_image.get("image_b64")

        # Row for page text.
        page_text = row_dict.get(text_column)
        if isinstance(page_text, str) and page_text.strip():
            page_row = _deep_copy_row(row_dict)
            page_row["_embed_modality"] = text_mod
            if text_mod in IMAGE_MODALITIES:
                page_row["_image_b64"] = page_image_b64
            new_rows.append(page_row)
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
                content_row["_embed_modality"] = struct_mod
                if struct_mod in IMAGE_MODALITIES and page_image_b64:
                    bbox = item.get("bbox_xyxy_norm")
                    if bbox and len(bbox) == 4:
                        cropped_b64, _ = _crop_b64_image_by_norm_bbox(page_image_b64, bbox_xyxy_norm=bbox)
                        content_row["_image_b64"] = cropped_b64
                    else:
                        content_row["_image_b64"] = page_image_b64
                elif struct_mod in IMAGE_MODALITIES:
                    content_row["_image_b64"] = None
                new_rows.append(content_row)
                exploded_any = True

        # Preserve row if nothing was exploded (no text, no content).
        if not exploded_any:
            preserved = _deep_copy_row(row_dict)
            preserved["_embed_modality"] = text_mod
            if text_mod in IMAGE_MODALITIES:
                preserved["_image_b64"] = page_image_b64
            new_rows.append(preserved)

    return pd.DataFrame(new_rows).reset_index(drop=True)


def _embed_group(
    group_df: pd.DataFrame,
    *,
    group_modality: str,
    model: Any,
    endpoint: Optional[str],
    text_column: str,
    inference_batch_size: int,
    output_column: str,
    resolved_model_name: str,
) -> pd.DataFrame:
    """Embed a single modality group via ``create_text_embeddings_for_df``.

    This is an internal helper called by ``embed_text_main_text_embed``
    once per distinct ``_embed_modality`` value found in the batch.
    """
    _embed = None
    _multimodal_embedder = None

    if endpoint is None and model is not None:
        if group_modality in IMAGE_MODALITIES:
            _multimodal_embedder = model
        else:
            _skip_prefix = hasattr(model, "embed_queries")

            def _embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:  # noqa: F811
                batch = texts if _skip_prefix else [f"passage: {t}" for t in texts]
                vecs = model.embed(batch, batch_size=int(inference_batch_size))
                tolist = getattr(vecs, "tolist", None)
                if callable(tolist):
                    return tolist()
                return vecs  # type: ignore[return-value]

    # For remote image/text_image embedding, cap the HTTP batch size to avoid
    # sending huge JSON payloads (each item includes a base64 page image).
    _DEFAULT_REMOTE_IMAGE_BATCH_SIZE = 4
    effective_batch_size = inference_batch_size
    if endpoint is not None and group_modality in IMAGE_MODALITIES:
        effective_batch_size = min(inference_batch_size, _DEFAULT_REMOTE_IMAGE_BATCH_SIZE)

    cfg = TextEmbeddingConfig(
        text_column=str(text_column),
        output_payload_column=str(output_column) if output_column else None,
        write_embedding_to_metadata=True,
        metadata_column="metadata",
        batch_size=int(effective_batch_size),
        encoding_format="float",
        input_type="passage",
        truncate="END",
        dimensions=None,
        embedding_nim_endpoint=endpoint or "http://localhost:8012/v1",
        embedding_model=resolved_model_name or "nvidia/llama-3.2-nv-embedqa-1b-v2",
        embed_modality=group_modality,
    )

    out_df, _ = create_text_embeddings_for_df(
        group_df,
        task_config={
            "embedder": _embed,
            "multimodal_embedder": _multimodal_embedder,
            "endpoint_url": endpoint,
            "local_batch_size": int(inference_batch_size),
        },
        transform_config=cfg,
    )
    return out_df


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
    embed_modality: str = "text",
    **_: Any,
) -> Any:
    """
    Inprocess embedding task implemented via `nemo_retriever.text_embed.main_text_embed`.

    This is a thin adapter that preserves the old output columns:
    - `output_column` (payload dict with embedding)
    - `embedding_dim_column` (int)
    - `has_embedding_column` (bool)

    It also writes `metadata.embedding` for compatibility with downstream uploaders.

    When *embedding_endpoint* is set (e.g. ``"http://embedding:8000/v1"``), a remote
    NIM endpoint is used instead of the local HF model.  The NIM handles prefixing
    (``input_type="passage"``) server-side.

    Rows are grouped by their ``_embed_modality`` column (populated by
    ``explode_content_to_rows``) so that different element types can be
    embedded with different methods (e.g. text-only for page text,
    text+image for tables).  If no ``_embed_modality`` column exists,
    *embed_modality* is used uniformly.
    """
    if not isinstance(batch_df, pd.DataFrame):
        raise NotImplementedError("embed_text_main_text_embed currently only supports pandas.DataFrame input.")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0")

    # Resolve endpoint: strip whitespace, treat empty string as None.
    _endpoint = (embedding_endpoint or embed_invoke_url or "").strip() or None

    if _endpoint is None and model is None:
        raise ValueError("Either a local model or an embedding_endpoint must be provided.")

    # Resolve NIM aliases to the actual HF model ID.
    from nemo_retriever.model import resolve_embed_model

    _resolved_model_name = resolve_embed_model(model_name)

    # Determine per-row modalities.  If _embed_modality column exists (set by
    # explode_content_to_rows), group by it; otherwise fall back to the single
    # embed_modality parameter.
    has_per_row_modality = "_embed_modality" in batch_df.columns
    if has_per_row_modality:
        modalities = batch_df["_embed_modality"].fillna(embed_modality).unique().tolist()
    else:
        modalities = [embed_modality]

    try:
        if len(modalities) == 1:
            # Fast path: all rows share the same modality — process in one shot.
            out_df = _embed_group(
                batch_df,
                group_modality=modalities[0],
                model=model,
                endpoint=_endpoint,
                text_column=text_column,
                inference_batch_size=inference_batch_size,
                output_column=output_column,
                resolved_model_name=_resolved_model_name,
            )
        else:
            # Multiple modalities: group, embed each, reassemble in original order.
            parts: List[pd.DataFrame] = []
            for mod in modalities:
                mask = batch_df["_embed_modality"] == mod
                group_df = batch_df.loc[mask]
                if group_df.empty:
                    continue
                part = _embed_group(
                    group_df,
                    group_modality=mod,
                    model=model,
                    endpoint=_endpoint,
                    text_column=text_column,
                    inference_batch_size=inference_batch_size,
                    output_column=output_column,
                    resolved_model_name=_resolved_model_name,
                )
                parts.append(part)
            out_df = pd.concat(parts).sort_index()

    except BaseException as e:
        # Fail-soft: preserve batch shape and set an error payload per-row.
        import traceback as _tb

        print(f"Warning: embedding failed: {type(e).__name__}: {e}")
        _tb.print_exc()
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

    # Drop helper columns after embedding to free memory.
    for col in ("_image_b64", "_embed_modality"):
        if col in out_df.columns:
            out_df = out_df.drop(columns=[col])

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


def pages_df_from_pdf_bytes(pdf_bytes: Union[bytes, bytearray], source_path: str) -> pd.DataFrame:
    """
    Build a per-page DataFrame from raw PDF bytes (same schema as pdf_path_to_pages_df).

    Used by the online ingest mode to run the same pipeline on document bytes
    received via REST. Columns: bytes, path, page_number.
    """
    pages = _split_pdf_to_single_page_bytes(pdf_bytes)
    out_rows = [{"bytes": b, "path": source_path, "page_number": i + 1} for i, b in enumerate(pages)]
    return pd.DataFrame(out_rows)


def run_pipeline_tasks_on_df(
    initial_df: pd.DataFrame,
    per_doc_tasks: Sequence[Tuple[Callable[..., Any], Dict[str, Any]]],
    post_tasks: Optional[Sequence[Tuple[Callable[..., Any], Dict[str, Any]]]] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Run the inprocess pipeline task chain on a single document DataFrame.

    Returns (final_result, metrics) where metrics is a list of
    {"stage": str, "duration_sec": float} for each stage. Used by both
    InProcessIngestor.ingest() and the online Ray Serve deployment.
    """
    import time

    metrics: List[Dict[str, Any]] = []
    current: Any = initial_df
    for func, kwargs in per_doc_tasks:
        t0 = time.perf_counter()
        if func is pdf_extraction:
            current = func(pdf_binary=current, **kwargs)
        else:
            current = func(current, **kwargs)
        metrics.append({"stage": getattr(func, "__name__", "unknown"), "duration_sec": time.perf_counter() - t0})

    if post_tasks and current is not None:
        combined = current if isinstance(current, pd.DataFrame) else pd.concat(current, ignore_index=True)
        for func, kwargs in post_tasks:
            t0 = time.perf_counter()
            combined = func(combined, **kwargs)
            metrics.append({"stage": getattr(func, "__name__", "unknown"), "duration_sec": time.perf_counter() - t0})
        return combined, metrics
    return current, metrics


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
    hybrid: bool = False,
    fts_language: str = "English",
    **_ignored: Any,
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
    - **hybrid**: if True, additionally build an FTS index on `text`
    - **fts_language**: language passed to LanceDB FTS index creation

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

        # Provide fields compatible with `nemo_retriever.recall.core` which expects LanceDB hits
        # to include JSON-encoded `metadata` and `source` strings.
        metadata_obj: Dict[str, Any] = {"page_number": int(page_number) if page_number is not None else -1}
        if pdf_page:
            metadata_obj["pdf_page"] = pdf_page
        # Persist per-page detection counters for end-of-run summaries.
        # Mirrors batch.py so LanceDB-based summary reads also work.
        pe_num = getattr(r, "page_elements_v3_num_detections", None)
        if pe_num is not None:
            try:
                metadata_obj["page_elements_v3_num_detections"] = int(pe_num)
            except Exception:
                pass
        pe_counts = getattr(r, "page_elements_v3_counts_by_label", None)
        if isinstance(pe_counts, dict):
            metadata_obj["page_elements_v3_counts_by_label"] = {
                str(k): int(v) for k, v in pe_counts.items() if isinstance(k, str) and v is not None
            }
        for ocr_col in ("table", "chart", "infographic"):
            entries = getattr(r, ocr_col, None)
            if isinstance(entries, list):
                metadata_obj[f"ocr_{ocr_col}_detections"] = int(len(entries))
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
        # Compatibility columns expected by `nemo_retriever.recall.core`:
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

    if hybrid:
        # Build text index for hybrid retrieval when text is present.
        try:
            table.create_fts_index("text", language=str(fts_language))
        except Exception as e:
            # Keep ingestion resilient: vector data was already written.
            print(f"Warning: failed to create LanceDB FTS index (continuing without FTS): {e}")

    print(f"Wrote {len(rows)} rows to LanceDB uri={lancedb_uri!r} table={table_name!r}")
    return df


def pdf_to_pages_df(path: str) -> pd.DataFrame:
    """
    Convert a document at *path* into a DataFrame where each row
    contains a *single-page* PDF's raw bytes.

    For ``.docx`` / ``.pptx`` files the document is first converted
    to PDF via LibreOffice, then split into pages.  The original
    *path* is preserved so downstream metadata tracks the source file.

    Columns:
    - bytes: single-page PDF bytes
    - path: original input path
    - page_number: 1-indexed page number
    """
    if pdfium is None:  # pragma: no cover
        raise ImportError("pypdfium2 is required for inprocess ingestion.") from _PDFIUM_IMPORT_ERROR

    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()
    out_rows: list[dict[str, Any]] = []
    doc = None
    try:
        if ext in SUPPORTED_EXTENSIONS and ext != ".pdf":
            # Convert DOCX/PPTX to PDF bytes first.
            with open(abs_path, "rb") as f:
                file_bytes = f.read()
            pdf_bytes = convert_to_pdf_bytes(file_bytes, ext)
            doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
        else:
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


def _iter_page_chunks(path: str, chunk_size: int = 32) -> Iterator[pd.DataFrame]:
    """Yield DataFrames of *chunk_size* pages from a document.

    Reuses the same pdfium page-splitting logic as :func:`pdf_to_pages_df`
    but yields incrementally so that downstream work can begin before the
    entire document is split.  At most *chunk_size* single-page PDF blobs
    are in memory per yield.

    Handles DOCX/PPTX conversion (same as ``pdf_to_pages_df``).
    """
    if pdfium is None:  # pragma: no cover
        raise ImportError("pypdfium2 is required for inprocess ingestion.") from _PDFIUM_IMPORT_ERROR

    abs_path = os.path.abspath(path)
    ext = os.path.splitext(abs_path)[1].lower()
    doc = None
    try:
        if ext in SUPPORTED_EXTENSIONS and ext != ".pdf":
            with open(abs_path, "rb") as f:
                pdf_bytes = convert_to_pdf_bytes(f.read(), ext)
            doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
        else:
            doc = pdfium.PdfDocument(abs_path)

        chunk: list[dict] = []
        for page_idx in range(len(doc)):
            single = pdfium.PdfDocument.new()
            try:
                single.import_pages(doc, pages=[page_idx])
                buf = BytesIO()
                single.save(buf)
                chunk.append({"bytes": buf.getvalue(), "path": abs_path, "page_number": page_idx + 1})
            finally:
                try:
                    single.close()
                except Exception:
                    pass
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk)
    except BaseException as e:
        yield pd.DataFrame([{"bytes": b"", "path": abs_path, "page_number": 0, "error": str(e)}])
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def _process_doc_cpu(doc_path: str, cpu_tasks: list) -> pd.DataFrame:
    """Worker function for ProcessPoolExecutor.

    Runs ``pdf_to_pages_df`` followed by all CPU-bound pipeline tasks on a
    single document.  All arguments are picklable (strings, module-level
    functions, and dicts of simple types).
    """
    try:
        current: Any = pdf_to_pages_df(doc_path)
        for func, kwargs in cpu_tasks:
            if func is pdf_extraction:
                current = func(pdf_binary=current, **kwargs)
            else:
                current = func(current, **kwargs)
        return current
    except Exception as e:
        # Return a minimal error DataFrame so one failed document does not
        # halt the pipeline.
        return pd.DataFrame([{"bytes": b"", "path": doc_path, "page_number": 0, "error": f"{type(e).__name__}: {e}"}])


def _process_chunk_cpu(chunk_df: pd.DataFrame, cpu_tasks: list) -> pd.DataFrame:
    """Worker function for ProcessPoolExecutor — page-chunk variant.

    Runs CPU-bound pipeline tasks on a pre-split page-chunk DataFrame.
    Similar to :func:`_process_doc_cpu` but skips ``pdf_to_pages_df``
    since the page splitting has already been done by the caller.
    """
    try:
        current: Any = chunk_df
        for func, kwargs in cpu_tasks:
            if func is pdf_extraction:
                current = func(pdf_binary=current, **kwargs)
            else:
                current = func(current, **kwargs)
        return current
    except Exception as e:
        return pd.DataFrame([{"bytes": b"", "path": "", "page_number": 0, "error": f"{type(e).__name__}: {e}"}])


def _collect_summary_from_df(df: pd.DataFrame) -> dict:
    """Compute detection summary from a result DataFrame.

    Mirrors the batch pipeline's ``_collect_detection_summary`` but reads
    directly from the in-memory DataFrame instead of LanceDB.  Rows are
    deduplicated by ``(path, page_number)`` so exploded content rows don't
    inflate counts.
    """
    per_page: dict[tuple, dict] = {}

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        path = str(row_dict.get("path") or row_dict.get("source_id") or "")
        page_number = -1
        try:
            page_number = int(row_dict.get("page_number", -1))
        except (TypeError, ValueError):
            pass

        key = (path, page_number)

        meta = row_dict.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        entry = per_page.setdefault(
            key,
            {
                "pe": 0,
                "ocr_table": 0,
                "ocr_chart": 0,
                "ocr_infographic": 0,
                "pe_by_label": defaultdict(int),
            },
        )

        # Check metadata first, then fall back to direct DataFrame columns.
        # The batch pipeline stores these inside the metadata JSON, but the
        # inprocess pipeline keeps them as top-level DataFrame columns.
        try:
            pe = int(
                meta.get("page_elements_v3_num_detections") or row_dict.get("page_elements_v3_num_detections") or 0
            )
        except (TypeError, ValueError):
            pe = 0
        entry["pe"] = max(entry["pe"], pe)

        for field, meta_key, col_key in [
            ("ocr_table", "ocr_table_detections", "table"),
            ("ocr_chart", "ocr_chart_detections", "chart"),
            ("ocr_infographic", "ocr_infographic_detections", "infographic"),
        ]:
            try:
                val = int(meta.get(meta_key, 0) or 0)
            except (TypeError, ValueError):
                val = 0
            # Fall back to counting direct list columns (e.g. row["table"]).
            if val == 0:
                col_val = row_dict.get(col_key)
                if isinstance(col_val, list):
                    val = len(col_val)
            entry[field] = max(entry[field], val)

        label_counts = meta.get("page_elements_v3_counts_by_label") or row_dict.get("page_elements_v3_counts_by_label")
        if isinstance(label_counts, dict):
            for label, count in label_counts.items():
                try:
                    c = int(count or 0)
                except (TypeError, ValueError):
                    c = 0
                entry["pe_by_label"][str(label)] = max(entry["pe_by_label"][str(label)], c)

    pe_by_label_totals: dict[str, int] = defaultdict(int)
    pe_total = ocr_table_total = ocr_chart_total = ocr_infographic_total = 0
    for e in per_page.values():
        pe_total += e["pe"]
        ocr_table_total += e["ocr_table"]
        ocr_chart_total += e["ocr_chart"]
        ocr_infographic_total += e["ocr_infographic"]
        for label, count in e["pe_by_label"].items():
            pe_by_label_totals[label] += count

    return {
        "pages_seen": len(per_page),
        "page_elements_v3_total_detections": pe_total,
        "page_elements_v3_counts_by_label": dict(sorted(pe_by_label_totals.items())),
        "ocr_table_total_detections": ocr_table_total,
        "ocr_chart_total_detections": ocr_chart_total,
        "ocr_infographic_total_detections": ocr_infographic_total,
    }


def _print_ingest_summary(results: list, elapsed_s: float) -> None:
    """Print end-of-ingest summary matching batch pipeline output format."""
    dfs = [r for r in results if isinstance(r, pd.DataFrame) and not r.empty]
    if not dfs:
        print(f"\nIngest time: {elapsed_s:.2f}s (no documents processed)")
        return

    combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    summary = _collect_summary_from_df(combined)

    print("\nDetection summary (deduped by source/page_number):")
    print(f"  Pages seen: {summary['pages_seen']}")
    print(f"  PageElements v3 total detections: {summary['page_elements_v3_total_detections']}")
    print(f"  OCR table detections: {summary['ocr_table_total_detections']}")
    print(f"  OCR chart detections: {summary['ocr_chart_total_detections']}")
    print(f"  OCR infographic detections: {summary['ocr_infographic_total_detections']}")
    print("  PageElements v3 counts by label:")
    by_label = summary.get("page_elements_v3_counts_by_label", {})
    if not by_label:
        print("    (none)")
    else:
        for label, count in by_label.items():
            print(f"    {label}: {count}")

    pages = summary["pages_seen"]
    if elapsed_s > 0 and pages > 0:
        pps = pages / elapsed_s
        print(f"Pages processed: {pages}")
        print(f"Pages/sec: {pps:.2f}")
    else:
        print(f"\nIngest time: {elapsed_s:.2f}s")


class InProcessIngestor(Ingestor):
    RUN_MODE = "inprocess"

    def __init__(self, documents: Optional[List[str]] = None) -> None:
        super().__init__(documents=documents)

        # Keep backwards-compatibility with code that inspects `Ingestor._documents`
        # by ensuring both names refer to the same list.
        self._input_documents: List[str] = self._documents

        # Builder-style configuration recorded for later execution (TBD).
        self._tasks: List[tuple[Callable[..., Any], dict[str, Any]]] = []

        # Pipeline type: "pdf" (extract), "txt" (extract_txt), or "html" (extract_html). Loader dispatch in ingest().
        self._pipeline_type: Literal["pdf", "txt", "html"] = "pdf"
        self._extract_txt_kwargs: Dict[str, Any] = {}
        self._extract_html_kwargs: Dict[str, Any] = {}

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

    def extract(self, params: ExtractParams | None = None, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure extraction for in-process execution (skeleton).

        TODO: implement actual local extraction (PDF split/extract, page elements,
        table structure, chart detection, etc.). For now this records the
        configuration so call sites can be wired up and chained.
        """
        # NOTE: `kwargs` passed to `.extract()` are intended primarily for PDF extraction
        # (e.g. `extract_text`, `dpi`, etc). Downstream model stages do NOT necessarily
        # accept the same keyword arguments. Keep per-stage kwargs isolated.

        resolved = _coerce_params(params, ExtractParams, kwargs)
        kwargs = resolved.model_dump(mode="python")
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
            pe_invoke_url = kwargs.get("page_elements_invoke_url", kwargs.get("invoke_url", ""))
            pe_model = None if pe_invoke_url else NemotronPageElementsV3()
            self._tasks.append(
                (
                    detect_page_elements_v3,
                    _detect_kwargs_with_model(
                        pe_model,
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
            ocr_invoke_url = kwargs.get("ocr_invoke_url", kwargs.get("invoke_url", ""))
            if ocr_invoke_url:
                self._tasks.append((ocr_page_elements, {"model": None, **ocr_flags}))
            else:
                ocr_model_dir = (
                    kwargs.get("ocr_model_dir")
                    or os.environ.get("RETRIEVER_NEMOTRON_OCR_MODEL_DIR", "").strip()
                    or os.environ.get("NEMOTRON_OCR_MODEL_DIR", "").strip()
                    or os.environ.get("NEMOTRON_OCR_V1_MODEL_DIR", "").strip()
                )
                model = NemotronOCRV1(model_dir=str(ocr_model_dir)) if ocr_model_dir else NemotronOCRV1()
                self._tasks.append((ocr_page_elements, {"model": model, **ocr_flags}))

        return self

    def extract_txt(self, params: TextChunkParams | None = None, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure txt ingestion: tokenizer-based chunking only (no PDF extraction).

        Use with .files("*.txt").extract_txt(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_txt().
        """
        self._pipeline_type = "txt"
        resolved = _coerce_params(params, TextChunkParams, kwargs)
        self._extract_txt_kwargs = resolved.model_dump(mode="python")
        return self

    def extract_html(self, params: HtmlChunkParams | None = None, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure HTML ingestion: markitdown -> markdown -> tokenizer chunking (no PDF extraction).

        Use with .files("*.html").extract_html(...).embed().vdb_upload().ingest().
        Do not call .extract() when using .extract_html().
        """
        self._pipeline_type = "html"
        resolved = _coerce_params(params, HtmlChunkParams, kwargs)
        self._extract_html_kwargs = resolved.model_dump(mode="python")
        return self

    def embed(self, params: EmbedParams | None = None, **kwargs: Any) -> "InProcessIngestor":
        """
        Configure embedding for in-process execution.

        This records an embedding task so call sites can chain `.embed(...)`
        after `.extract(...)`, `.extract_txt()`, or `.extract_html()`.

        When ``embedding_endpoint`` (or ``embed_invoke_url``) is provided (e.g.
        ``"http://embedding:8000/v1"``), a remote NIM endpoint is used for
        embedding instead of the local HF model.
        """
        resolved = _coerce_params(params, EmbedParams, kwargs)
        embed_modality = resolved.embed_modality
        text_elements_modality = resolved.text_elements_modality or embed_modality
        structured_elements_modality = resolved.structured_elements_modality or embed_modality

        # Explode content rows before embedding so each table/chart/infographic
        # gets its own embedding vector (mirrors nv-ingest per-element embeddings).
        self._tasks.append(
            (
                explode_content_to_rows,
                {
                    "modality": embed_modality,
                    "text_elements_modality": text_elements_modality,
                    "structured_elements_modality": structured_elements_modality,
                },
            )
        )

        embed_kwargs = {
            **resolved.model_dump(
                mode="python", exclude={"runtime", "batch_tuning", "fused_tuning"}, exclude_none=True
            ),
            **resolved.runtime.model_dump(mode="python", exclude_none=True),
        }
        if "embedding_endpoint" not in embed_kwargs and embed_kwargs.get("embed_invoke_url"):
            embed_kwargs["embedding_endpoint"] = embed_kwargs.get("embed_invoke_url")

        # Ensure embed_modality is forwarded to the embedding function.
        embed_kwargs["embed_modality"] = embed_modality

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

        model_name_raw = embed_kwargs.pop("model_name", None)

        from nemo_retriever.model import is_vl_embed_model, resolve_embed_model

        model_id = resolve_embed_model(model_name_raw)

        embed_kwargs.setdefault("input_type", "passage")

        if is_vl_embed_model(model_name_raw):
            from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            embed_kwargs["model"] = LlamaNemotronEmbedVL1BV2Embedder(
                device=str(device) if device is not None else None,
                hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
                model_id=model_id,
            )
        else:
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

    def vdb_upload(self, params: VdbUploadParams | None = None, **kwargs: Any) -> "InProcessIngestor":
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
        p = params or VdbUploadParams()
        if kwargs:
            lancedb_kwargs = {k: v for k, v in kwargs.items() if k != "purge_results_after_upload"}
            if lancedb_kwargs:
                p = p.model_copy(update={"lancedb": p.lancedb.model_copy(update=lancedb_kwargs)})
            if "purge_results_after_upload" in kwargs:
                p = p.model_copy(update={"purge_results_after_upload": bool(kwargs["purge_results_after_upload"])})
        _ = p.purge_results_after_upload  # parity with interface; inprocess does not purge by default
        self._tasks.append((upload_embeddings_to_lancedb_inprocess, p.lancedb.model_dump(mode="python")))
        return self

    # Tasks that run once on combined results (after all docs). All others run per-doc.
    _POST_TASKS = (upload_embeddings_to_lancedb_inprocess, save_dataframe_to_disk_json)

    def get_pipeline_tasks(
        self,
    ) -> Tuple[List[Tuple[Callable[..., Any], Dict[str, Any]]], List[Tuple[Callable[..., Any], Dict[str, Any]]]]:
        """
        Return (per_doc_tasks, post_tasks) for use by ingest() or by the online
        serve deployment to run the same pipeline on a single document.
        """
        per_doc_tasks = [(f, k) for f, k in self._tasks if f not in self._POST_TASKS]
        post_tasks = [(f, k) for f, k in self._tasks if f in self._POST_TASKS]
        return per_doc_tasks, post_tasks

    def ingest(self, params: IngestExecuteParams | None = None, **kwargs: Any) -> list[Any]:
        run_params = _coerce_params(params, IngestExecuteParams, kwargs)
        show_progress = run_params.show_progress
        _ = (run_params.return_failures, run_params.save_to_disk, run_params.return_traces)
        parallel = run_params.parallel
        max_workers = run_params.max_workers
        gpu_devices = list(run_params.gpu_devices) if run_params.gpu_devices else None
        page_chunk_size = run_params.page_chunk_size

        _start = time.perf_counter()

        # -- Three-way task classification --------------------------------
        _post_task_fns = (upload_embeddings_to_lancedb_inprocess, save_dataframe_to_disk_json)
        _cpu_task_fns = (pdf_extraction,)

        cpu_tasks = [(f, k) for f, k in self._tasks if f in _cpu_task_fns]
        gpu_tasks = [(f, k) for f, k in self._tasks if f not in _cpu_task_fns and f not in _post_task_fns]
        post_tasks = [(f, k) for f, k in self._tasks if f in _post_task_fns]

        docs = list(self._documents)

        # -- Parallel execution branch ------------------------------------
        if parallel:
            if gpu_devices and len(gpu_devices) >= 1 and gpu_tasks:
                # Pipelined: GPU workers load models while CPU runs,
                # each completed chunk goes to GPU immediately.
                from .gpu_pool import GPUWorkerPool, gpu_tasks_to_descriptors

                descriptors = gpu_tasks_to_descriptors(gpu_tasks)
                errors: list[str] = []

                with GPUWorkerPool(gpu_devices, descriptors) as gpu_pool:
                    # Split all docs into page chunks (main thread, cheap PDF byte splitting)
                    chunks: list[pd.DataFrame] = []
                    chunk_to_doc: list[str] = []
                    doc_chunk_total: dict[str, int] = defaultdict(int)
                    for doc in docs:
                        for chunk_df in _iter_page_chunks(doc, page_chunk_size):
                            chunk_to_doc.append(doc)
                            doc_chunk_total[doc] += 1
                            chunks.append(chunk_df)

                    shard_id = 0
                    progress = None
                    if show_progress and tqdm is not None:
                        progress = tqdm(total=len(docs), desc="Processing files", unit="file")

                    shard_to_doc: dict[int, str] = {}
                    doc_done: dict[str, int] = defaultdict(int)

                    def _check_file_done(doc_path: str) -> None:
                        if doc_done[doc_path] >= doc_chunk_total[doc_path]:
                            if progress is not None:
                                progress.update(1)

                    with ProcessPoolExecutor(
                        max_workers=max_workers,
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as cpu_pool:
                        future_to_idx = {
                            cpu_pool.submit(_process_chunk_cpu, chunk, cpu_tasks): i for i, chunk in enumerate(chunks)
                        }

                        for future in as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            doc = chunk_to_doc[idx]
                            try:
                                result = future.result()
                                if isinstance(result, pd.DataFrame) and not result.empty:
                                    shard_to_doc[shard_id] = doc
                                    gpu_pool.submit(shard_id, result)
                                    shard_id += 1
                                else:
                                    doc_done[doc] += 1
                                    _check_file_done(doc)
                            except Exception as e:
                                errors.append(f"chunk {idx}: {type(e).__name__}: {e}")
                                print(f"Warning: failed to process chunk {idx}: {e}")
                                doc_done[doc] += 1
                                _check_file_done(doc)

                    if errors:
                        print(f"Warning: {len(errors)} chunk(s) failed CPU extraction")

                    # All CPU done, collect remaining GPU results
                    def _on_gpu_done(sid: int) -> None:
                        d = shard_to_doc.get(sid, "")
                        if d:
                            doc_done[d] += 1
                            _check_file_done(d)

                    combined = gpu_pool.collect_all(on_shard_done=_on_gpu_done)

                    if progress is not None:
                        progress.close()

                if combined.empty:
                    results: list = []
                    if show_progress:
                        _print_ingest_summary(results, time.perf_counter() - _start)
                    return results

                for func, kwargs in post_tasks:
                    combined = func(combined, **kwargs)

                results = [combined]
                if show_progress:
                    _print_ingest_summary(results, time.perf_counter() - _start)
                return results

            else:
                # Single-GPU or no-GPU path: CPU parallel on chunks, then sequential GPU
                chunks: list[pd.DataFrame] = []
                chunk_to_doc: list[str] = []
                doc_chunk_total: dict[str, int] = defaultdict(int)
                for doc in docs:
                    for chunk_df in _iter_page_chunks(doc, page_chunk_size):
                        chunk_to_doc.append(doc)
                        doc_chunk_total[doc] += 1
                        chunks.append(chunk_df)

                cpu_results: list[pd.DataFrame] = []
                errors: list[str] = []

                progress = None
                if show_progress and tqdm is not None:
                    progress = tqdm(total=len(docs), desc="Processing files", unit="file")

                doc_done: dict[str, int] = defaultdict(int)

                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    future_to_idx = {pool.submit(_process_chunk_cpu, c, cpu_tasks): i for i, c in enumerate(chunks)}

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        doc = chunk_to_doc[idx]
                        try:
                            result = future.result()
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                cpu_results.append(result)
                        except Exception as e:
                            errors.append(f"chunk {idx}: {type(e).__name__}: {e}")
                            print(f"Warning: failed to process chunk {idx}: {e}")
                        doc_done[doc] += 1
                        if doc_done[doc] >= doc_chunk_total[doc] and progress is not None:
                            progress.update(1)

                if progress is not None:
                    progress.close()

                if errors:
                    print(f"Warning: {len(errors)} chunk(s) failed CPU extraction")

                if not cpu_results:
                    results = []
                    if show_progress:
                        _print_ingest_summary(results, time.perf_counter() - _start)
                    return results

                combined = pd.concat(cpu_results, ignore_index=True)
                for func, kwargs in gpu_tasks:
                    combined = func(combined, **kwargs)

                for func, kwargs in post_tasks:
                    combined = func(combined, **kwargs)

                results = [combined]
                if show_progress:
                    _print_ingest_summary(results, time.perf_counter() - _start)
                return results

        # -- Sequential execution branch (default) ------------------------
        use_multi_gpu_seq = gpu_devices and len(gpu_devices) >= 1 and gpu_tasks

        if use_multi_gpu_seq:
            # Pipelined: GPU workers process earlier chunks while CPU
            # extracts later chunks of the same (or next) document.
            cpu_and_extract = [(f, k) for f, k in self._tasks if f in _cpu_task_fns]

            from .gpu_pool import GPUWorkerPool, gpu_tasks_to_descriptors

            descriptors = gpu_tasks_to_descriptors(gpu_tasks)

            with GPUWorkerPool(gpu_devices, descriptors) as gpu_pool:
                shard_id = 0
                shard_to_doc: dict[int, str] = {}
                doc_chunk_total: dict[str, int] = defaultdict(int)
                doc_done: dict[str, int] = defaultdict(int)

                for doc_path in docs:
                    for chunk_df in _iter_page_chunks(doc_path, page_chunk_size):
                        doc_chunk_total[doc_path] += 1
                        current: Any = chunk_df
                        for func, kwargs in cpu_and_extract:
                            if func is pdf_extraction:
                                current = func(pdf_binary=current, **kwargs)
                            else:
                                current = func(current, **kwargs)
                        if isinstance(current, pd.DataFrame) and not current.empty:
                            shard_to_doc[shard_id] = doc_path
                            gpu_pool.submit(shard_id, current)
                            shard_id += 1
                        else:
                            doc_done[doc_path] += 1

                progress = None
                if show_progress and tqdm is not None:
                    # Docs whose chunks all failed CPU are already done
                    already_done = sum(
                        1 for d in docs if d not in doc_chunk_total or doc_done.get(d, 0) >= doc_chunk_total[d]
                    )
                    progress = tqdm(total=len(docs), desc="Processing files", unit="file", initial=already_done)

                def _on_gpu_done(sid: int) -> None:
                    d = shard_to_doc.get(sid, "")
                    if d:
                        doc_done[d] += 1
                        if doc_done[d] >= doc_chunk_total[d] and progress is not None:
                            progress.update(1)

                combined = gpu_pool.collect_all(on_shard_done=_on_gpu_done)

                if progress is not None:
                    progress.close()

            if combined.empty:
                results = []
                if show_progress:
                    _print_ingest_summary(results, time.perf_counter() - _start)
                return results

            for func, kwargs in post_tasks:
                combined = func(combined, **kwargs)

            results = [combined]
            if show_progress:
                _print_ingest_summary(results, time.perf_counter() - _start)
            return results

        # Fully sequential: per-document CPU + GPU + post tasks
        per_doc_tasks = [(f, k) for f, k in self._tasks if f not in _post_task_fns]

        results: list[Any] = []
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
                        out_rows = [{"bytes": b, "path": abs_path, "page_number": i + 1} for i, b in enumerate(pages)]
                        return pd.DataFrame(out_rows)
                    except BaseException as e:
                        return pd.DataFrame([{"bytes": b"", "path": abs_path, "page_number": 0, "error": str(e)}])
                return pdf_path_to_pages_df(p)

        elif self._pipeline_type == "html":

            def _loader(p: str) -> pd.DataFrame:
                return html_file_to_chunks_df(p, params=HtmlChunkParams(**self._extract_html_kwargs))

        else:

            def _loader(p: str) -> pd.DataFrame:
                return txt_file_to_chunks_df(p, params=TextChunkParams(**self._extract_txt_kwargs))

        for doc_path in doc_iter:
            initial_df = _loader(doc_path)
            current, _ = run_pipeline_tasks_on_df(initial_df, per_doc_tasks, None)
            results.append(current)

        # Run upload/save once on combined results so overwrite=True keeps full corpus.
        if post_tasks and results and all(isinstance(r, pd.DataFrame) for r in results):
            combined = pd.concat(results, ignore_index=True)
            for func, kwargs in post_tasks:
                combined = func(combined, **kwargs)

        if show_progress:
            _print_ingest_summary(results, time.perf_counter() - _start)
        return results
