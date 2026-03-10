# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from nemo_retriever.ingest_config import load_ingest_config_section
from nemo_retriever.pdf.config import load_pdf_extractor_schema_from_dict
from nemo_retriever.pdf.io import pdf_files_to_ledger_df
from rich.console import Console
from tqdm import tqdm
import typer

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)


console = Console()
app = typer.Typer(help="Extract PDF primitives from PDF files.")


def _argv_has_any(flags: Iterable[str]) -> bool:
    """
    Return True if any of the given flags appear in sys.argv.

    Supports `--flag value` and `--flag=value` forms.
    """
    argv = sys.argv[1:]
    for f in flags:
        if f in argv:
            return True
        prefix = f + "="
        if any(a.startswith(prefix) for a in argv):
            return True
    return False


def _read_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception as e:
        raise typer.BadParameter(f"Failed reading YAML config {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"YAML config must be a mapping/object at top-level: {path}")
    return data


def _cfg_get(raw: Dict[str, Any], *keys: str) -> Any:
    """
    Return the first present key among `keys` from raw, else None.
    """
    for k in keys:
        if k in raw:
            return raw.get(k)
    return None


def _normalize_page_elements_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize YAML config into flat keys matching CLI parameter names.

    Supports either flat keys (e.g. `yolox_grpc_endpoint`) or nested keys under:
      endpoints.yolox.{grpc,http}
      endpoints.nemotron_parse.{grpc,http,model_name}
      extract.{text,images,tables,charts,infographics,page_as_image,text_depth}
      outputs.{write_json,json_output_dir}
    """
    raw = dict(raw or {})

    endpoints = raw.get("endpoints") if isinstance(raw.get("endpoints"), dict) else {}
    yolox = endpoints.get("yolox") if isinstance(endpoints.get("yolox"), dict) else {}
    nemo = endpoints.get("nemotron_parse") if isinstance(endpoints.get("nemotron_parse"), dict) else {}

    extract = raw.get("extract") if isinstance(raw.get("extract"), dict) else {}
    outputs = raw.get("outputs") if isinstance(raw.get("outputs"), dict) else {}

    out: Dict[str, Any] = {}
    out["input_dir"] = _cfg_get(raw, "input_dir", "input-dir")
    out["method"] = _cfg_get(raw, "method")
    out["auth_token"] = _cfg_get(raw, "auth_token", "auth-token")

    out["yolox_grpc_endpoint"] = _cfg_get(raw, "yolox_grpc_endpoint", "yolox-grpc-endpoint") or _cfg_get(
        yolox, "grpc", "grpc_endpoint", "grpc-endpoint"
    )
    out["yolox_http_endpoint"] = _cfg_get(raw, "yolox_http_endpoint", "yolox-http-endpoint") or _cfg_get(
        yolox, "http", "http_endpoint", "http-endpoint"
    )

    out["nemotron_parse_grpc_endpoint"] = _cfg_get(
        raw, "nemotron_parse_grpc_endpoint", "nemotron-parse-grpc-endpoint"
    ) or _cfg_get(nemo, "grpc", "grpc_endpoint", "grpc-endpoint")
    out["nemotron_parse_http_endpoint"] = _cfg_get(
        raw, "nemotron_parse_http_endpoint", "nemotron-parse-http-endpoint"
    ) or _cfg_get(nemo, "http", "http_endpoint", "http-endpoint")
    out["nemotron_parse_model_name"] = _cfg_get(
        raw, "nemotron_parse_model_name", "nemotron-parse-model-name"
    ) or _cfg_get(nemo, "model_name", "model-name")

    out["extract_text"] = (
        _cfg_get(raw, "extract_text", "extract-text") if "extract_text" in raw else _cfg_get(extract, "text")
    )
    out["extract_images"] = (
        _cfg_get(raw, "extract_images", "extract-images") if "extract_images" in raw else _cfg_get(extract, "images")
    )
    out["extract_tables"] = (
        _cfg_get(raw, "extract_tables", "extract-tables") if "extract_tables" in raw else _cfg_get(extract, "tables")
    )
    out["extract_charts"] = (
        _cfg_get(raw, "extract_charts", "extract-charts") if "extract_charts" in raw else _cfg_get(extract, "charts")
    )
    out["extract_infographics"] = (
        _cfg_get(raw, "extract_infographics", "extract-infographics")
        if "extract_infographics" in raw
        else _cfg_get(extract, "infographics")
    )
    out["extract_page_as_image"] = (
        _cfg_get(raw, "extract_page_as_image", "extract-page-as-image")
        if "extract_page_as_image" in raw
        else _cfg_get(extract, "page_as_image", "page-as-image", "page_as_image")
    )
    out["text_depth"] = _cfg_get(raw, "text_depth", "text-depth") or _cfg_get(extract, "text_depth", "text-depth")

    out["write_json_outputs"] = (
        _cfg_get(raw, "write_json_outputs", "write-json-outputs")
        if "write_json_outputs" in raw
        else _cfg_get(outputs, "write_json", "write-json", "write_json_outputs")
    )
    out["json_output_dir"] = _cfg_get(raw, "json_output_dir", "json-output-dir") or _cfg_get(
        outputs, "json_output_dir", "json-output-dir"
    )
    out["limit"] = _cfg_get(raw, "limit")

    # Drop Nones so "not specified" stays not specified.
    return {k: v for k, v in out.items() if v is not None}


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


def _safe_pdf_page_count(path: str) -> int:
    """
    Best-effort page count for a PDF path.

    Used only for progress reporting; failures should not fail extraction.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(path)
        try:
            n = int(len(doc))
        except Exception:
            n = int(doc.get_page_count())  # type: ignore[attr-defined]
        try:
            doc.close()
        except Exception:
            pass
        return max(n, 0)
    except Exception:
        # If we can't read page count, fall back to 0 (still shows PDF progress).
        return 0


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

    Additionally, it includes `extracted_df_records`, a JSON-friendly list of dicts
    with the same columns as the raw `extracted_df` output:
      [{"document_type": ..., "metadata": ..., "uuid": ...}, ...]
    so downstream stages can reconstruct a `pandas.DataFrame` directly.
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

        # Keep a DataFrame-shaped record so downstream stages can do:
        #   df = pd.DataFrame(payload["extracted_df_records"])
        record = {
            "uuid": str(uuid),
            "document_type": row.get("document_type"),
            "metadata": _to_jsonable(meta),
        }

        sid = _primitive_source_id(meta)
        if sid:
            prims_by_source_id.setdefault(sid, []).append(record)

        p = _primitive_pdf_path(meta)
        if p:
            prims_by_pdf_path.setdefault(p, []).append(record)

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
        records = prims_by_source_id.get(source_id) or prims_by_pdf_path.get(pdf_path_str) or []

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
            # Back-compat: keep the old key.
            "primitives": records,
            # New: explicitly labeled as the raw extracted_df rows.
            "extracted_df_records": records,
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
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help=(
            "Optional ingest YAML config file. If omitted, we auto-discover ./ingest-config.yaml then "
            "$HOME/.ingest-config.yaml. Explicitly passed CLI flags override YAML."
        ),
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan recursively for *.pdf (can be provided via --config).",
    ),
    method: str = typer.Option(
        "pdfium",
        "--method",
        help="PDF extraction method (e.g. 'pdfium', 'pdfium_hybrid', 'ocr', 'nemotron_parse', 'tika').",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help="Auth token for NIM-backed services (e.g. YOLOX / Nemotron Parse).",
    ),
    yolox_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--yolox-grpc-endpoint",
        help="YOLOX gRPC endpoint (e.g. 'page-elements:8001'). Required for method 'pdfium' family.",
    ),
    yolox_http_endpoint: Optional[str] = typer.Option(
        None,
        "--yolox-http-endpoint",
        help="YOLOX HTTP endpoint (e.g. 'http://page-elements:8000/v1/infer'). Required for method 'pdfium' family.",
    ),
    nemotron_parse_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-grpc-endpoint",
        help="Nemotron Parse gRPC endpoint (required for method 'nemotron_parse').",
    ),
    nemotron_parse_http_endpoint: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-http-endpoint",
        help="Nemotron Parse HTTP endpoint (required for method 'nemotron_parse').",
    ),
    nemotron_parse_model_name: Optional[str] = typer.Option(
        None,
        "--nemotron-parse-model-name",
        help="Nemotron Parse model name (optional; defaults to schema default).",
    ),
    extract_text: bool = typer.Option(True, "--extract-text/--no-extract-text", help="Extract text primitives."),
    extract_images: bool = typer.Option(
        False, "--extract-images/--no-extract-images", help="Extract image primitives."
    ),
    extract_tables: bool = typer.Option(
        False, "--extract-tables/--no-extract-tables", help="Extract table primitives."
    ),
    extract_charts: bool = typer.Option(
        False, "--extract-charts/--no-extract-charts", help="Extract chart primitives."
    ),
    extract_infographics: bool = typer.Option(
        False, "--extract-infographics/--no-extract-infographics", help="Extract infographic primitives."
    ),
    extract_page_as_image: bool = typer.Option(
        False, "--extract-page-as-image/--no-extract-page-as-image", help="Extract full page images as primitives."
    ),
    text_depth: str = typer.Option(
        "page",
        "--text-depth",
        help="Text depth for extracted text primitives: 'page' or 'document'.",
    ),
    write_json_outputs: bool = typer.Option(
        True,
        "--write-json-outputs/--no-write-json-outputs",
        help="Write one <pdf>.pdf_extraction.json sidecar per input PDF.",
    ),
    json_output_dir: Optional[Path] = typer.Option(
        None,
        "--json-output-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional directory to write JSON outputs into (instead of next to PDFs).",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optionally limit number of PDFs processed."),
) -> None:
    """
    Scan `input_dir` for PDFs and run nv-ingest-api PDF extraction, writing primitives JSON sidecars.

    This command is intentionally "directory-first" so you can point it at a folder of PDFs
    and get per-PDF outputs without having to build a ledger by hand.
    """
    # Load consolidated ingest config (section: pdf).
    cfg_raw = _normalize_page_elements_config(load_ingest_config_section(config, section="pdf"))

    # Merge: YAML provides defaults; explicit CLI flags override YAML.
    if not _argv_has_any(["--input-dir"]):
        if "input_dir" in cfg_raw:
            input_dir = Path(str(cfg_raw["input_dir"]))

    if not _argv_has_any(["--method"]):
        method = str(cfg_raw.get("method", method))
    if not _argv_has_any(["--auth-token"]):
        auth_token = cfg_raw.get("auth_token", auth_token)

    if not _argv_has_any(["--yolox-grpc-endpoint"]):
        yolox_grpc_endpoint = cfg_raw.get("yolox_grpc_endpoint", yolox_grpc_endpoint)
    if not _argv_has_any(["--yolox-http-endpoint"]):
        yolox_http_endpoint = cfg_raw.get("yolox_http_endpoint", yolox_http_endpoint)

    if not _argv_has_any(["--nemotron-parse-grpc-endpoint"]):
        nemotron_parse_grpc_endpoint = cfg_raw.get("nemotron_parse_grpc_endpoint", nemotron_parse_grpc_endpoint)
    if not _argv_has_any(["--nemotron-parse-http-endpoint"]):
        nemotron_parse_http_endpoint = cfg_raw.get("nemotron_parse_http_endpoint", nemotron_parse_http_endpoint)
    if not _argv_has_any(["--nemotron-parse-model-name"]):
        nemotron_parse_model_name = cfg_raw.get("nemotron_parse_model_name", nemotron_parse_model_name)

    if not _argv_has_any(["--extract-text", "--no-extract-text"]):
        extract_text = bool(cfg_raw.get("extract_text", extract_text))
    if not _argv_has_any(["--extract-images", "--no-extract-images"]):
        extract_images = bool(cfg_raw.get("extract_images", extract_images))
    if not _argv_has_any(["--extract-tables", "--no-extract-tables"]):
        extract_tables = bool(cfg_raw.get("extract_tables", extract_tables))
    if not _argv_has_any(["--extract-charts", "--no-extract-charts"]):
        extract_charts = bool(cfg_raw.get("extract_charts", extract_charts))
    if not _argv_has_any(["--extract-infographics", "--no-extract-infographics"]):
        extract_infographics = bool(cfg_raw.get("extract_infographics", extract_infographics))
    if not _argv_has_any(["--extract-page-as-image", "--no-extract-page-as-image"]):
        extract_page_as_image = bool(cfg_raw.get("extract_page_as_image", extract_page_as_image))

    if not _argv_has_any(["--text-depth"]):
        text_depth = str(cfg_raw.get("text_depth", text_depth))

    if not _argv_has_any(["--write-json-outputs", "--no-write-json-outputs"]):
        write_json_outputs = bool(cfg_raw.get("write_json_outputs", write_json_outputs))
    if not _argv_has_any(["--json-output-dir"]):
        if "json_output_dir" in cfg_raw and cfg_raw["json_output_dir"] is not None:
            json_output_dir = Path(str(cfg_raw["json_output_dir"]))

    if not _argv_has_any(["--limit"]):
        limit = cfg_raw.get("limit", limit)

    if input_dir is None:
        raise typer.BadParameter("Missing --input-dir (or set input_dir in --config YAML).")

    pdfs = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDFs found[/red] under: {input_dir}")
        raise typer.Exit(code=2)

    if limit is not None:
        pdfs = pdfs[: int(limit)]

    method = str(method or "pdfium")

    extractor_cfg: Dict[str, Any] = {}
    if method in {"pdfium", "pdfium_hybrid", "ocr"}:
        if not (yolox_grpc_endpoint or yolox_http_endpoint):
            print("YOLOX NIM endpoints not set, using HuggingFace model instead.")

        extractor_cfg["pdfium_config"] = {
            "auth_token": auth_token,
            "yolox_endpoints": [yolox_grpc_endpoint, yolox_http_endpoint],
        }
    elif method == "nemotron_parse":
        if not (nemotron_parse_grpc_endpoint or nemotron_parse_http_endpoint):
            raise typer.BadParameter(
                "Nemotron Parse endpoint required for method 'nemotron_parse'. "
                "Set --nemotron-parse-grpc-endpoint or --nemotron-parse-http-endpoint."
            )
        extractor_cfg["nemotron_parse_config"] = {
            "auth_token": auth_token,
            # Nemotron Parse may still rely on YOLOX for region proposals depending on config.
            "yolox_endpoints": [yolox_grpc_endpoint, yolox_http_endpoint],
            "nemotron_parse_endpoints": [nemotron_parse_grpc_endpoint, nemotron_parse_http_endpoint],
            "nemotron_parse_model_name": nemotron_parse_model_name,
        }

    extractor_schema = load_pdf_extractor_schema_from_dict(extractor_cfg)
    task_cfg = make_pdf_task_config(
        method=method,
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        extract_page_as_image=extract_page_as_image,
        text_depth=text_depth,
    )

    # Progress reporting:
    # - PDF progress: updates per completed PDF.
    # - Page progress: updates by page-count *after* each PDF completes (best-effort).
    page_counts = [_safe_pdf_page_count(p) for p in tqdm(pdfs, desc="Counting pages", unit="pdf")]
    total_pages = sum(int(n) for n in page_counts)

    extracted_chunks: List[pd.DataFrame] = []
    all_json_outputs: List[str] = []
    info_last: Dict[str, Any] = {}

    pdf_bar = tqdm(total=len(pdfs), desc="PDFs", unit="pdf")
    page_bar = tqdm(total=total_pages, desc="Pages", unit="page")
    try:
        for i, (pdf_path, n_pages) in enumerate(zip(pdfs, page_counts)):
            pdf_bar.set_postfix_str(Path(pdf_path).name)

            df_ledger = pdf_files_to_ledger_df([pdf_path], start_index=i)
            df_one, info_one = extract_pdf_primitives_from_ledger_df(
                df_ledger,
                task_config=task_cfg,
                extractor_config=extractor_schema,
                write_json_outputs=bool(write_json_outputs),
                json_output_dir=json_output_dir,
            )
            info_last = dict(info_one or {})

            if not df_one.empty:
                extracted_chunks.append(df_one)

            wrote = info_last.get("json_outputs")
            if isinstance(wrote, list):
                all_json_outputs.extend(str(x) for x in wrote)

            pdf_bar.update(1)
            page_bar.update(int(n_pages))
    finally:
        pdf_bar.close()
        page_bar.close()

    extracted_df = (
        pd.concat(extracted_chunks, ignore_index=True)
        if extracted_chunks
        else pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})
    )
    info = dict(info_last or {})
    if all_json_outputs:
        info["json_outputs"] = all_json_outputs

    wrote = info.get("json_outputs")
    wrote_n = len(wrote) if isinstance(wrote, list) else 0
    console.print(f"[green]Done[/green] pdfs={len(pdfs)} primitives={len(extracted_df)} wrote_json={wrote_n}")
