from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from retriever._local_deps import ensure_nv_ingest_api_importable
from rich.console import Console
import typer

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)


console = Console()
app = typer.Typer(help="Extract PDF primitives from PDF files.")


def _atomic_write_json(path: Path, obj: Any) -> None:
    """
    Atomically write JSON to disk (write temp file then replace).

    Kept local to the PDF stage to avoid importing heavier stage utilities.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            import os

            os.fsync(f.fileno())
        except Exception:
            pass
    tmp_path.replace(path)


def _to_jsonable(obj: Any) -> Any:
    """
    Best-effort conversion to JSON-serializable builtins.

    PDF primitive metadata can contain numpy/pandas scalar-like objects depending on backend;
    this sanitizes them without pulling in optional heavy deps.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    # numpy scalar / pandas scalar-like
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass

    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass

    # fallback
    return str(obj)


def _pdf_json_out_path(pdf_path: Path) -> Path:
    # Mirror stage2 naming: append suffix to the full filename.
    return pdf_path.with_name(pdf_path.name + ".pdf_extraction.json")


def _iter_ledger_pdf_paths(df_ledger: pd.DataFrame) -> Iterable[Tuple[str, str, Optional[str]]]:
    """
    Yield (source_id, source_name, pdf_path_str_or_none) from the input ledger.
    """
    for _, row in df_ledger.iterrows():
        source_id = str(row.get("source_id", ""))
        source_name = str(row.get("source_name", ""))
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        custom = meta.get("custom_content") if isinstance(meta.get("custom_content"), dict) else {}
        path = custom.get("path")
        path_str = str(path) if isinstance(path, (str, Path)) and str(path) else None
        yield source_id, source_name, path_str


def _primitive_source_id(meta: Any) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    source = meta.get("source_metadata") if isinstance(meta.get("source_metadata"), dict) else {}
    sid = source.get("source_id") or meta.get("source_id")
    return str(sid) if sid is not None else None


def _primitive_pdf_path(meta: Any) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    custom = meta.get("custom_content") if isinstance(meta.get("custom_content"), dict) else {}
    path = custom.get("path")
    return str(path) if isinstance(path, str) and path else None


def _write_pdf_extraction_json_outputs(
    *,
    df_ledger: pd.DataFrame,
    extracted_df: pd.DataFrame,
    task_config: Dict[str, Any],
    elapsed_seconds: float,
    output_dir: Optional[str | Path] = None,
) -> List[Path]:
    """
    Write one JSON per input PDF (sidecar) summarizing extracted primitives.

    The JSON payload mirrors the stage2 family shape:
      {schema_version, stage, model, <input>, <outputs>, timing}
    """
    out_paths: List[Path] = []

    # Pre-bucket primitives by best-effort identifiers.
    prims_by_source_id: Dict[str, List[Dict[str, Any]]] = {}
    prims_by_pdf_path: Dict[str, List[Dict[str, Any]]] = {}

    for i, row in extracted_df.iterrows():
        meta = row.get("metadata")
        uuid = None
        if isinstance(meta, dict):
            uuid = meta.get("uuid")
        if uuid is None:
            uuid = row.get("uuid") or f"row:{i}"

        prim = {
            "uuid": str(uuid),
            "document_type": row.get("document_type"),
            "metadata": _to_jsonable(meta),
        }

        sid = _primitive_source_id(meta)
        if sid:
            prims_by_source_id.setdefault(sid, []).append(prim)

        p = _primitive_pdf_path(meta)
        if p:
            prims_by_pdf_path.setdefault(p, []).append(prim)

    out_base_dir: Optional[Path] = None
    if output_dir is not None:
        out_base_dir = Path(output_dir)

    method = str(task_config.get("method") or "pdf_extraction")
    params = task_config.get("params") if isinstance(task_config.get("params"), dict) else {}

    for source_id, source_name, pdf_path_str in _iter_ledger_pdf_paths(df_ledger):
        if not pdf_path_str:
            # No stable path to write next to; skip.
            continue

        # Prefer source_id mapping; fall back to path mapping.
        prims = prims_by_source_id.get(source_id) or prims_by_pdf_path.get(pdf_path_str) or []

        pdf_path = Path(pdf_path_str)
        out_path = (
            (out_base_dir / _pdf_json_out_path(pdf_path).name)
            if out_base_dir is not None
            else _pdf_json_out_path(pdf_path)
        )

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "stage": 1,
            "model": method,
            "pdf": {"path": str(pdf_path), "source_id": source_id, "source_name": source_name},
            "task": {"method": method, "params": _to_jsonable(params)},
            "primitives": prims,
            "timing": {"seconds": float(elapsed_seconds)},
        }
        _atomic_write_json(out_path, payload)
        out_paths.append(out_path)

    return out_paths


def make_pdf_task_config(
    *,
    method: str = "pdfium",
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    extract_page_as_image: bool = False,
    text_depth: str = "page",
    extract_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the `task_config` dict expected by `nv-ingest-api` PDF extraction internals."""
    params: Dict[str, Any] = {
        "extract_text": extract_text,
        "extract_images": extract_images,
        "extract_tables": extract_tables,
        "extract_charts": extract_charts,
        "extract_infographics": extract_infographics,
        "extract_page_as_image": extract_page_as_image,
        "text_depth": text_depth,
    }
    if extract_method is not None:
        # Some callsites use params["extract_method"] while others use task_config["method"].
        params["extract_method"] = extract_method

    return {"method": method, "params": params}


def _validate_pdf_ledger_df(df: pd.DataFrame) -> None:
    required = {"content", "source_id", "source_name", "document_type", "metadata"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"PDF ledger DataFrame is missing required columns: {missing}")


@traceable_func(trace_name="retriever::pdf_extraction")
def extract_pdf_primitives_from_ledger_df(
    df_ledger: pd.DataFrame,
    *,
    task_config: Dict[str, Any],
    extractor_config: PDFExtractorSchema,
    write_json_outputs: bool = True,
    json_output_dir: Optional[str | Path] = None,
    trace_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Pure-Python PDF extraction (no Ray required).

    Returns:
      - extracted_df: columns ["document_type", "metadata", "uuid"]
      - info: includes `execution_trace_log`
    """
    _validate_pdf_ledger_df(df_ledger)

    # `nv-ingest-api` engines treat this as a string-keyed mapping of trace timestamps.
    # The Ray pipeline stage uses `{}` as well; using a list can trigger
    # "list indices must be integers or slices, not str".
    execution_trace_log: Dict[str, Any] = {}
    try:
        t0 = time.perf_counter()
        extracted_df, info = extract_primitives_from_pdf_internal(
            df_extraction_ledger=df_ledger,
            task_config=task_config,
            extractor_config=extractor_config,
            execution_trace_log=execution_trace_log,
        )
        elapsed = time.perf_counter() - t0
    except Exception:
        logger.exception("PDF extraction failed")
        raise

    info = dict(info or {})
    info.setdefault("execution_trace_log", execution_trace_log)

    if write_json_outputs:
        try:
            written = _write_pdf_extraction_json_outputs(
                df_ledger=df_ledger,
                extracted_df=extracted_df,
                task_config=task_config,
                elapsed_seconds=float(elapsed),
                output_dir=json_output_dir,
            )
            info["json_outputs"] = [str(p) for p in written]
        except Exception:
            # JSON output is best-effort; don't fail extraction if serialization/IO fails.
            logger.exception("Failed writing PDF extraction JSON outputs")
            info["json_outputs_error"] = "failed_writing_json_outputs"

    return extracted_df, info


@app.command("page-elements")
def render_page_elements(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .page_element_detections.png files."),
    min_score: float = typer.Option(0.0, "--min-score", help="Only draw detections with score >= min_score."),
    line_width: int = typer.Option(3, "--line-width", min=1, help="Bounding box line width in pixels."),
    draw_labels: bool = typer.Option(True, "--draw-labels/--no-draw-labels", help="Draw label + score text."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of images processed."),
) -> None:
    print("something")
