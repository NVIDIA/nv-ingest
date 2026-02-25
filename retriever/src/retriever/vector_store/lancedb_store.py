from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

from nv_ingest_client.util.vdb.lancedb import LanceDB
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBConfig:
    """
    Minimal config for writing embeddings into LanceDB.

    This module is intentionally lightweight: it can be used by the text-embedding
    stage (`retriever.text_embed.stage`) and by the vector-store CLI (`retriever.vector_store.stage`).
    """

    uri: str = "lancedb"
    table_name: str = "nv-ingest"
    overwrite: bool = True

    # Optional index creation (recommended for recall/search runs).
    create_index: bool = True
    index_type: str = "IVF_HNSW_SQ"
    metric: str = "l2"
    num_partitions: int = 16
    num_sub_vectors: int = 256
    hybrid: bool = False
    fts_language: str = "English"
    wait_timeout_seconds: int = 600


def _read_text_embeddings_json_df(path: Path) -> pd.DataFrame:
    """
    Read a `*.text_embeddings.json` file emitted by `retriever.text_embed.stage`.

    Expected wrapper shape:
      {
        ...,
        "df_records": [ { "document_type": ..., "metadata": {...}, ... }, ... ],
        ...
      }
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        raise ValueError(f"Failed reading JSON {path}: {e}") from e

    if isinstance(obj, dict):
        recs = obj.get("df_records")
        if isinstance(recs, list):
            return pd.DataFrame([r for r in recs if isinstance(r, dict)])
        # Fall back to a single record.
        return pd.DataFrame([obj])

    if isinstance(obj, list):
        return pd.DataFrame([r for r in obj if isinstance(r, dict)])

    return pd.DataFrame([])


def _iter_text_embeddings_json_files(input_dir: Path, *, recursive: bool) -> List[Path]:
    """
    Return sorted list of `*.text_embeddings.json` files.

    The stage5 default naming is: `<input>.text_embeddings.json` (where `<input>` is
    typically a stage4 output filename).
    """
    if recursive:
        files = list(input_dir.rglob("*.text_embeddings.json"))
    else:
        files = list(input_dir.glob("*.text_embeddings.json"))
    return sorted([p for p in files if p.is_file()])


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _extract_source_path_and_id(meta: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract a stable source path/id from metadata.

    Prefers:
      - metadata.source_metadata.source_id
      - metadata.source_metadata.source_name
      - metadata.custom_content.path
    """
    source = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
    source_id = source.get("source_id") or ""
    source_name = source.get("source_name") or ""

    custom = meta.get("custom_content") if isinstance(meta.get("custom_content"), dict) else {}
    custom_path = custom.get("path") or custom.get("input_pdf") or custom.get("pdf_path") or ""

    path = _safe_str(custom_path or source_id or source_name)
    sid = _safe_str(source_id or path or source_name)
    return path, sid


def _extract_page_number(meta: Dict[str, Any]) -> int:
    cm = meta.get("content_metadata") if isinstance(meta.get("content_metadata"), dict) else {}
    page = cm.get("hierarchy", {}).get("page", -1)
    try:
        return int(page)
    except Exception:
        return -1


def _build_lancedb_rows_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Transform an embeddings-enriched primitives DataFrame into LanceDB rows.

    Rows include:
      - vector (embedding)
      - pdf_basename
      - page_number
      - pdf_page (basename_page)
      - source_id
      - path
    """
    out: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue

        embedding = meta.get("embedding")
        if embedding is None:
            continue

        # Normalize embedding to list[float]
        if not isinstance(embedding, list):
            try:
                embedding = list(embedding)  # type: ignore[arg-type]
            except Exception:
                continue

        path, source_id = _extract_source_path_and_id(meta)
        page_number = _extract_page_number(meta)
        p = Path(path) if path else None
        filename = p.name if p is not None else ""
        pdf_basename = p.stem if p is not None else ""
        pdf_page = f"{pdf_basename}_{page_number}" if (pdf_basename and page_number >= 0) else ""

        if page_number == -1:
            logger.debug("Unable to determine page number for %s", path)

        out.append(
            {
                "vector": embedding,
                "pdf_page": pdf_page,
                "filename": filename,
                "pdf_basename": pdf_basename,
                "page_number": int(page_number),
                "source_id": source_id,
                "path": path,
            }
        )

    return out


def _infer_vector_dim(rows: Sequence[Dict[str, Any]]) -> int:
    for r in rows:
        v = r.get("vector")
        if isinstance(v, list) and v:
            return int(len(v))
    return 0


def create_lancedb_index(
    *,
    lancedb_uri: str,
    table_name: str,
    create_index: bool = True,
    index_type: str = "IVF_HNSW_SQ",
    metric: str = "l2",
    num_partitions: int = 16,
    num_sub_vectors: int = 256,
    hybrid: bool = False,
    fts_language: str = "English",
    vector_column_name: str = "vector",
    fts_text_column_name: str = "text",
    wait_timeout_seconds: int = 600,
) -> Dict[str, Any]:
    """
    Create LanceDB vector index and optional FTS index for hybrid search.
    """
    if not bool(create_index):
        return {"indexed": False, "reason": "disabled"}

    try:
        import lancedb  # type: ignore
    except Exception as e:
        raise RuntimeError("LanceDB index creation requested but `lancedb` is unavailable.") from e

    db = lancedb.connect(uri=str(lancedb_uri))
    table = db.open_table(str(table_name))
    n_vecs = int(table.count_rows())
    if n_vecs < 2:
        logger.info("Skipping LanceDB index creation (not enough vectors): rows=%s", n_vecs)
        return {"indexed": False, "reason": "not_enough_vectors", "rows": n_vecs}

    # LanceDB IVF training requires num_partitions < num_vectors.
    k = int(num_partitions)
    if k >= n_vecs:
        k = max(1, n_vecs - 1)

    try:
        table.create_index(
            index_type=str(index_type),
            metric=str(metric),
            num_partitions=int(k),
            num_sub_vectors=int(num_sub_vectors),
            vector_column_name=str(vector_column_name),
        )
    except TypeError:
        table.create_index(vector_column_name=str(vector_column_name))

    created_fts = False
    if bool(hybrid):
        try:
            table.create_fts_index(str(fts_text_column_name), language=str(fts_language))
            created_fts = True
        except Exception:
            logger.warning(
                "Hybrid indexing requested, but failed to create FTS index on column=%r. "
                "Continuing with vector index only.",
                fts_text_column_name,
                exc_info=True,
            )

    try:
        for index_stub in table.list_indices():
            table.wait_for_index([index_stub.name], timeout=timedelta(seconds=int(wait_timeout_seconds)))
    except Exception:
        # Keep indexing robust across LanceDB versions that differ in wait/list APIs.
        logger.debug("LanceDB index wait API unavailable; continuing without wait.", exc_info=True)

    return {
        "indexed": True,
        "rows": n_vecs,
        "num_partitions": int(k),
        "hybrid": bool(hybrid),
        "fts_index_created": created_fts,
    }


def _write_rows_to_lancedb(rows: Sequence[Dict[str, Any]], *, cfg: LanceDBConfig) -> None:
    if not rows:
        logger.warning("No embeddings rows provided; nothing to write to LanceDB.")
        return

    dim = _infer_vector_dim(rows)
    if dim <= 0:
        raise ValueError("Failed to infer embedding dimension from rows.")

    try:
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LanceDB write requested but dependencies are missing. "
            "Install `lancedb` and `pyarrow` in this environment."
        ) from e

    db = lancedb.connect(uri=cfg.uri)

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("pdf_page", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("pdf_basename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("source_id", pa.string()),
            pa.field("path", pa.string()),
        ]
    )

    mode = "overwrite" if cfg.overwrite else "append"
    db.create_table(cfg.table_name, data=list(rows), schema=schema, mode=mode)

    if cfg.create_index:
        create_lancedb_index(
            lancedb_uri=cfg.uri,
            table_name=cfg.table_name,
            create_index=bool(cfg.create_index),
            index_type=str(cfg.index_type),
            metric=str(cfg.metric),
            num_partitions=int(cfg.num_partitions),
            num_sub_vectors=int(cfg.num_sub_vectors),
            hybrid=bool(cfg.hybrid),
            fts_language=str(cfg.fts_language),
            wait_timeout_seconds=int(cfg.wait_timeout_seconds),
        )


def write_embeddings_to_lancedb(df_with_embeddings: pd.DataFrame, *, cfg: LanceDBConfig) -> None:
    """
    Write embeddings found in `df_with_embeddings.metadata.embedding` to LanceDB.

    This is used programmatically by `retriever.text_embed.stage.embed_text_from_primitives_df(...)`.
    """
    rows = _build_lancedb_rows_from_df(df_with_embeddings)
    _write_rows_to_lancedb(rows, cfg=cfg)


def write_text_embeddings_dir_to_lancedb(
    input_dir: Path,
    *,
    cfg: LanceDBConfig,
    recursive: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read `*.text_embeddings.json` files from `input_dir` and upload their embeddings to LanceDB.
    """
    input_dir = Path(input_dir)
    files = _iter_text_embeddings_json_files(input_dir, recursive=bool(recursive))
    if limit is not None:
        files = files[: int(limit)]

    processed = 0
    skipped = 0
    failed = 0

    lancedb = LanceDB(
        uri=cfg.uri,
        table_name=cfg.table_name,
        overwrite=cfg.overwrite,
        index_type=cfg.index_type,
        metric=cfg.metric,
        num_partitions=cfg.num_partitions,
        num_sub_vectors=cfg.num_sub_vectors,
        hybrid=cfg.hybrid,
        fts_language=cfg.fts_language,
    )

    results = []

    for p in files:
        df = _read_text_embeddings_json_df(p)
        rows = df.to_dict(orient="records")
        results.append(rows)

    if not results:
        logger.warning("No *.text_embeddings.json files found in %s; nothing to write.", input_dir)
        return {
            "input_dir": str(input_dir),
            "n_files": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
        }

    table = lancedb.create_index(records=results, table_name=cfg.table_name)
    if cfg.create_index:
        lancedb.write_to_index(
            records=results,
            table=table,
            index_type=cfg.index_type,
            metric=cfg.metric,
            num_partitions=cfg.num_partitions,
            num_sub_vectors=cfg.num_sub_vectors,
            hybrid=cfg.hybrid,
            fts_language=cfg.fts_language,
        )

    # all_rows: List[Dict[str, Any]] = []
    # for p in files:
    #     try:
    #         df = _read_text_embeddings_json_df(p)
    #         if df.empty:
    #             skipped += 1
    #             continue
    #         rows = _build_lancedb_rows_from_df(df)
    #         if not rows:
    #             skipped += 1
    #             continue
    #         all_rows.extend(rows)
    #         processed += 1
    #     except Exception:
    #         failed += 1
    #         logger.exception("Failed reading embeddings from %s", p)

    # # Write once so --overwrite behaves as expected.
    # _write_rows_to_lancedb(all_rows, cfg=cfg)

    return {
        "input_dir": str(input_dir),
        "n_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        # "rows_written": len(all_rows),
        "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
    }
