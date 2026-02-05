from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanceDBConfig:
    uri: str
    table: str = "embeddings"
    mode: str = "append"  # append | overwrite


EXPECTED_COLUMNS: Sequence[str] = (
    "id",
    "vector",
    "document_type",
    "source_id",
    "source_name",
    "source_location",
    "content_type",
    "content_subtype",
    "metadata_json",
    "path",
    "pdf_basename",
    "page_number",
    "pdf_page",
)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return json.dumps({"_unserializable": str(type(obj))}, ensure_ascii=False)


def _read_json_or_jsonl(path: Path) -> Any:
    """
    Read either:
      - JSON (object or list)
      - JSONL (one JSON object per line)
    """
    txt = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(txt)
    except Exception:
        # Fall back to jsonl.
        recs = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # Skip malformed lines rather than failing the entire ingest.
                continue
        return recs


def _extract_df_records(obj: Any) -> List[Dict[str, Any]]:
    """
    Extract DataFrame-shaped records from wrapper payloads.

    Supports common wrapper keys:
      - df_records (text embedding stage output)
      - extracted_df_records / primitives (pdf extraction sidecars)
    """
    if isinstance(obj, dict):
        for k in ("df_records", "extracted_df_records", "primitives"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # If it's already a "row-shaped" dict, allow it.
        return [obj]

    if isinstance(obj, list):
        # If jsonl contains exactly one wrapper payload, unwrap it too.
        if (
            len(obj) == 1
            and isinstance(obj[0], dict)
            and not ("metadata" in obj[0] and ("document_type" in obj[0] or "uuid" in obj[0]))
        ):
            return _extract_df_records(obj[0])
        return [x for x in obj if isinstance(x, dict)]

    return []


def primitives_df_from_embeddings_json(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read an embedding-stage JSON artifact (e.g. `*.text_embeddings.json`) and reconstruct
    a primitives DataFrame with a `metadata` column containing `embedding`.

    Expected wrapper shape (from `retriever.text_embed.stage` CLI):
      { ..., "df_records": [ {"document_type": ..., "metadata": {..., "embedding": [...]}, ...}, ... ] }
    """
    p = Path(path)
    obj = _read_json_or_jsonl(p)
    records = _extract_df_records(obj)
    df = pd.DataFrame(records)
    return df


def write_embeddings_json_to_lancedb(path: Union[str, Path], *, cfg: LanceDBConfig) -> int:
    """
    Convenience wrapper: read `*.text_embeddings.json` (wrapper payload with `df_records`)
    and write `metadata.embedding` vectors into LanceDB.
    """
    df = primitives_df_from_embeddings_json(path)
    return write_embeddings_to_lancedb(df, cfg=cfg)


def _iter_embedding_rows(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    """Yield LanceDB rows from an nv-ingest primitives DataFrame.

    Assumes embedding lives in `metadata["embedding"]`.
    """
    for i, row in df.iterrows():
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        emb = meta.get("embedding")
        if emb is None:
            continue

        custom = meta.get("custom_content") if isinstance(meta.get("custom_content"), dict) else {}
        source = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
        content_md = meta.get("content_metadata") if isinstance(meta.get("content_metadata"), dict) else {}
        page_number = content_md.get("page_number")
        path = custom.get("path")
        pdf_basename = None
        if isinstance(path, str) and path:
            try:
                pdf_basename = Path(path).name
            except Exception:
                pdf_basename = path

        yield {
            "id": meta.get("uuid") or row.get("uuid") or f"row:{i}",
            "vector": emb,
            "document_type": row.get("document_type"),
            "source_id": source.get("source_id"),
            "source_name": source.get("source_name"),
            "source_location": source.get("source_location"),
            "content_type": content_md.get("type"),
            "content_subtype": content_md.get("subtype"),
            "metadata_json": _safe_json(meta),
            "path": path,
            "pdf_basename": pdf_basename,
            "page_number": page_number,
            "pdf_page": f"{pdf_basename}_{page_number}" if (pdf_basename is not None and page_number is not None) else None,
        }


def _table_column_names(tbl: Any) -> set[str]:
    # LanceDB returns a pyarrow.Schema for tbl.schema in most versions
    schema = getattr(tbl, "schema", None)
    if schema is None:
        return set()
    names = getattr(schema, "names", None)
    if names is not None:
        return set(names)
    try:
        return set(schema)  # type: ignore[arg-type]
    except Exception:
        return set()


def _migrate_table_schema_overwrite(
    *,
    db: Any,
    table_name: str,
    expected_columns: Sequence[str],
) -> None:
    """Rewrite the table with missing columns added as nulls.

    This is a compatibility path for older tables created before we added
    `pdf_page` / `pdf_basename` / `page_number` etc.
    """
    tbl = db.open_table(table_name)
    existing = tbl.to_pandas()
    for col in expected_columns:
        if col not in existing.columns:
            existing[col] = None
    db.create_table(table_name, data=existing.to_dict(orient="records"), mode="overwrite")


def write_embeddings_to_lancedb(
    df: pd.DataFrame,
    *,
    cfg: LanceDBConfig,
) -> int:
    """Write embeddings from `df` to a LanceDB table.

    Returns the number of rows written.
    """
    import lancedb  # type: ignore

    rows = list(_iter_embedding_rows(df))
    if not rows:
        return 0

    db = lancedb.connect(cfg.uri)

    if cfg.mode == "overwrite":
        tbl = db.create_table(cfg.table, data=rows, mode="overwrite")
        _ = tbl
        return len(rows)

    # append mode: create if missing, else add
    if cfg.table in db.table_names():
        tbl = db.open_table(cfg.table)
        table_cols = _table_column_names(tbl)
        row_cols = set(rows[0].keys())
        missing = sorted((row_cols - table_cols))
        if missing:
            logger.warning(
                "LanceDB table '%s' missing columns %s; rewriting table to add them.",
                cfg.table,
                missing,
            )
            _migrate_table_schema_overwrite(db=db, table_name=cfg.table, expected_columns=EXPECTED_COLUMNS)
            tbl = db.open_table(cfg.table)
        tbl.add(rows)
    else:
        _ = db.create_table(cfg.table, data=rows, mode="create")

    return len(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ingest nv-ingest embedding JSON artifacts into LanceDB.\n\n"
            "Reads wrapper payloads (e.g. '*.text_embeddings.json') and writes rows based on "
            "`metadata.embedding` (stored as column 'vector')."
        )
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to embedding JSON artifact (e.g. '*.text_embeddings.json'). Supports JSON or JSONL.",
    )
    p.add_argument("--uri", required=True, help="LanceDB URI (e.g. './lancedb').")
    p.add_argument("--table", default="embeddings", help="LanceDB table name.")
    p.add_argument("--mode", default="append", choices=["append", "overwrite"], help="Write mode.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    n = write_embeddings_json_to_lancedb(
        args.input,
        cfg=LanceDBConfig(uri=str(args.uri), table=str(args.table), mode=str(args.mode)),
    )
    print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

