#!/usr/bin/env python3
from __future__ import annotations

import base64
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import typer

# If this file is executed directly (not via installed package), ensure `retriever/src`
# is on sys.path so `import retriever...` works from a monorepo checkout.
_THIS_FILE = Path(__file__).resolve()
_RETRIEVER_SRC = _THIS_FILE.parents[1]  # .../retriever/src
if (_RETRIEVER_SRC / "retriever").is_dir() and str(_RETRIEVER_SRC) not in sys.path:
    sys.path.insert(0, str(_RETRIEVER_SRC))

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

from retriever.pdf.config import load_pdf_extractor_schema_from_dict
from retriever.pdf.stage import extract_pdf_primitives_from_ledger_df, make_pdf_task_config

logger = logging.getLogger(__name__)
app = typer.Typer(
    help=(
        "Ray Data batch pipeline for PDF extraction (stage1).\n"
        "\n"
        "1) Ingest: read all PDFs into a Ray Dataset with `read_binary*`\n"
        "2) Actor stage: run the same nv-ingest PDF extraction logic as `retriever local stage1`\n"
    )
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

@dataclass(frozen=True)
class LedgerRow:
    source_id: str
    source_name: str
    content_b64: str
    pdf_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "content": self.content_b64,
            "document_type": "pdf",
            "metadata": {"custom_content": {"path": self.pdf_path}},
        }


def _iter_pdf_paths(
    *,
    input_dir: Optional[Path],
    pdf_list: Optional[Path],
    limit_pdfs: Optional[int],
) -> List[str]:
    pdfs: List[str] = []

    if input_dir is not None:
        pdfs.extend(str(p) for p in sorted(input_dir.rglob("*.pdf")))

    if pdf_list is not None:
        lines = pdf_list.read_text(encoding="utf-8").splitlines()
        for ln in lines:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            pdfs.append(str(Path(s)))

    # De-dupe while preserving order
    seen = set()
    ordered: List[str] = []
    for p in pdfs:
        if p in seen:
            continue
        seen.add(p)
        ordered.append(p)

    if limit_pdfs is not None:
        ordered = ordered[: int(limit_pdfs)]

    return ordered


def _normalize_pdf_binary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Ray's binary reader output to stable keys:
      - pdf_path: str
      - pdf_bytes: bytes
    """
    path = row.get("path") or row.get("file_path") or row.get("uri") or row.get("filename")
    b = row.get("bytes") or row.get("data") or row.get("content")
    pdf_path = str(path) if path is not None else ""

    pdf_bytes: bytes
    if isinstance(b, bytes):
        pdf_bytes = b
    elif isinstance(b, bytearray):
        pdf_bytes = bytes(b)
    elif isinstance(b, memoryview):
        pdf_bytes = b.tobytes()
    else:
        # Best-effort fallback; extraction will fail gracefully downstream if this isn't valid bytes.
        pdf_bytes = bytes(b) if b is not None else b""  # type: ignore[arg-type]

    return {"pdf_path": pdf_path, "pdf_bytes": pdf_bytes}


def _binary_batch_to_ledger_df(batch: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for _, r in batch.iterrows():
        pdf_path = str(r.get("pdf_path") or "")
        b = r.get("pdf_bytes")

        if isinstance(b, bytes):
            raw = b
        elif isinstance(b, bytearray):
            raw = bytes(b)
        elif isinstance(b, memoryview):
            raw = b.tobytes()
        else:
            raw = bytes(b) if b is not None else b""  # type: ignore[arg-type]

        content_b64 = base64.b64encode(raw).decode("utf-8")
        led = LedgerRow(
            source_id=pdf_path,
            source_name=pdf_path,
            content_b64=content_b64,
            pdf_path=pdf_path,
        )
        rows.append(led.to_dict())

    return pd.DataFrame(rows)


def _build_pdf_extractor_and_task_cfg(
    *,
    method: str,
    auth_token: Optional[str],
    yolox_grpc_endpoint: Optional[str],
    yolox_http_endpoint: Optional[str],
    nemotron_parse_grpc_endpoint: Optional[str],
    nemotron_parse_http_endpoint: Optional[str],
    nemotron_parse_model_name: Optional[str],
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    extract_infographics: bool,
    extract_page_as_image: bool,
    text_depth: str,
) -> tuple[Any, Dict[str, Any]]:
    """
    Mirror the config behavior from `retriever.pdf.stage` stage1 CLI.
    """
    method = str(method or "pdfium")

    extractor_cfg: Dict[str, Any] = {}
    if method in {"pdfium", "pdfium_hybrid", "ocr"}:
        if not (yolox_grpc_endpoint or yolox_http_endpoint):
            logger.info("YOLOX NIM endpoints not set; falling back to HuggingFace model.")
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
        extract_text=bool(extract_text),
        extract_images=bool(extract_images),
        extract_tables=bool(extract_tables),
        extract_charts=bool(extract_charts),
        extract_infographics=bool(extract_infographics),
        extract_page_as_image=bool(extract_page_as_image),
        text_depth=str(text_depth or "page"),
    )
    return extractor_schema, task_cfg


class PDFExtractionActorBatchFn:
    """
    Actor-based Ray Data stage that runs stage1 PDF extraction over binary-ingested PDFs.
    """

    def __init__(
        self,
        *,
        method: str,
        auth_token: Optional[str],
        yolox_grpc_endpoint: Optional[str],
        yolox_http_endpoint: Optional[str],
        nemotron_parse_grpc_endpoint: Optional[str],
        nemotron_parse_http_endpoint: Optional[str],
        nemotron_parse_model_name: Optional[str],
        extract_text: bool,
        extract_images: bool,
        extract_tables: bool,
        extract_charts: bool,
        extract_infographics: bool,
        extract_page_as_image: bool,
        text_depth: str,
        write_json_outputs: bool,
        json_output_dir: Optional[str],
    ) -> None:
        # Ensure nv-ingest-api is importable on workers too.
        ensure_nv_ingest_api_importable()

        extractor_schema, task_cfg = _build_pdf_extractor_and_task_cfg(
            method=str(method),
            auth_token=auth_token,
            yolox_grpc_endpoint=yolox_grpc_endpoint,
            yolox_http_endpoint=yolox_http_endpoint,
            nemotron_parse_grpc_endpoint=nemotron_parse_grpc_endpoint,
            nemotron_parse_http_endpoint=nemotron_parse_http_endpoint,
            nemotron_parse_model_name=nemotron_parse_model_name,
            extract_text=bool(extract_text),
            extract_images=bool(extract_images),
            extract_tables=bool(extract_tables),
            extract_charts=bool(extract_charts),
            extract_infographics=bool(extract_infographics),
            extract_page_as_image=bool(extract_page_as_image),
            text_depth=str(text_depth),
        )
        self._extractor_schema = extractor_schema
        self._task_cfg = task_cfg
        self._write_json_outputs = bool(write_json_outputs)
        self._json_output_dir = str(json_output_dir) if json_output_dir else None

    def __call__(self, batch: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        df_ledger = _binary_batch_to_ledger_df(batch)
        extracted_df, _info = extract_pdf_primitives_from_ledger_df(
            df_ledger,
            task_config=self._task_cfg,
            extractor_config=self._extractor_schema,
            write_json_outputs=bool(self._write_json_outputs),
            json_output_dir=self._json_output_dir,
        )
        # Always return a DataFrame (Ray expects batch outputs to be tabular).
        if extracted_df is None or not isinstance(extracted_df, pd.DataFrame):
            return pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})
        return extracted_df


def _write_jsonl_driver_side(ds: Any, *, output_dir: Path, rows_per_file: int = 50_000) -> None:
    """
    Write JSON Lines without pandas by streaming Arrow batches on the driver.

    Ray's built-in `Dataset.write_json()` currently converts blocks to pandas, which can
    fail in environments with incompatible pandas builds. This avoids pandas entirely.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    row_idx_in_file = 0
    f = (output_dir / f"part-{file_idx:05d}.jsonl").open("w", encoding="utf-8")
    try:
        for batch in ds.iter_batches(batch_format="pyarrow", batch_size=1024):
            rows = batch.to_pylist()
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                row_idx_in_file += 1
                if row_idx_in_file >= int(rows_per_file):
                    f.close()
                    file_idx += 1
                    row_idx_in_file = 0
                    f = (output_dir / f"part-{file_idx:05d}.jsonl").open("w", encoding="utf-8")
    finally:
        try:
            f.close()
        except Exception:
            pass


@app.command("run")
def run(
    input_dir: Optional[Path] = typer.Option(
        None, "--input-dir", exists=True, file_okay=False, dir_okay=True, help="Directory to scan for *.pdf"
    ),
    pdf_list: Optional[Path] = typer.Option(
        None,
        "--pdf-list",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Text file with one PDF path per line (comments with # supported).",
    ),
    limit_pdfs: Optional[int] = typer.Option(None, "--limit-pdfs", help="Optionally limit number of PDFs."),
    # Ray
    ray_address: Optional[str] = typer.Option(
        None, "--ray-address", help="Ray cluster address (omit for local). Example: 'ray://host:10001' or 'auto'."
    ),
    # Stage1 PDF extraction (matches `retriever pdf page-elements` defaults closely)
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
        help="YOLOX gRPC endpoint (e.g. 'page-elements:8001'). Used by method 'pdfium' family.",
    ),
    yolox_http_endpoint: Optional[str] = typer.Option(
        None,
        "--yolox-http-endpoint",
        help="YOLOX HTTP endpoint (e.g. 'http://page-elements:8000/v1/infer'). Used by method 'pdfium' family.",
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
    extract_images: bool = typer.Option(False, "--extract-images/--no-extract-images", help="Extract image primitives."),
    extract_tables: bool = typer.Option(False, "--extract-tables/--no-extract-tables", help="Extract table primitives."),
    extract_charts: bool = typer.Option(False, "--extract-charts/--no-extract-charts", help="Extract chart primitives."),
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
        help="Write one <pdf>.pdf_extraction.json sidecar per input PDF (best-effort).",
    ),
    json_output_dir: Optional[Path] = typer.Option(
        None,
        "--json-output-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional directory to write JSON outputs into (instead of next to PDFs).",
    ),
    pdf_batch_size: int = typer.Option(8, "--pdf-batch-size", min=1, help="Ray Data batch size for PDF extraction."),
    pdf_actors: int = typer.Option(1, "--pdf-actors", min=1, help="Number of PDF extraction actors."),
    # Output
    output_format: str = typer.Option(
        "parquet",
        "--output-format",
        help="Output format: 'parquet' (recommended) or 'jsonl' (pandas-free fallback).",
    ),
    jsonl_rows_per_file: int = typer.Option(
        50_000,
        "--jsonl-rows-per-file",
        min=1,
        help="When --output-format=jsonl, max rows per output file (driver-side writer).",
    ),
    output_dir: Path = typer.Option(
        Path("ray_ocr_outputs"),
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory to write output shards into.",
    ),
) -> None:
    """
    Run the pipeline.
    """
    import ray  # type: ignore
    import ray.data as rd

    logging.basicConfig(level=logging.INFO)

    pdfs = _iter_pdf_paths(input_dir=input_dir, pdf_list=pdf_list, limit_pdfs=limit_pdfs)
    if not pdfs:
        raise typer.BadParameter("No PDFs found. Provide --input-dir and/or --pdf-list.")

    ray.init(address=ray_address, ignore_reinit_error=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building binary dataset from PDFs: pdfs=%s", len(pdfs))
    # Read all pdfs files into memory has it makes controlling parallelism easier and allows for distributed execution.
    pdf_bin = rd.read_binary_files(pdfs, include_paths=True)

    # Normalize schema so downstream stages are stable.
    pdf_bin = pdf_bin.map(_normalize_pdf_binary_row)

    logger.info("Actor stage: PDF extraction (actors=%s, batch_size=%s)", pdf_actors, pdf_batch_size)
    extracted = pdf_bin.map_batches(
        PDFExtractionActorBatchFn,
        batch_format="pandas",
        batch_size=int(pdf_batch_size),
        num_cpus=8,
        compute=rd.ActorPoolStrategy(size=int(pdf_actors)),
        fn_constructor_kwargs={
            "method": str(method),
            "auth_token": auth_token,
            "yolox_grpc_endpoint": yolox_grpc_endpoint,
            "yolox_http_endpoint": yolox_http_endpoint,
            "nemotron_parse_grpc_endpoint": nemotron_parse_grpc_endpoint,
            "nemotron_parse_http_endpoint": nemotron_parse_http_endpoint,
            "nemotron_parse_model_name": nemotron_parse_model_name,
            "extract_text": bool(extract_text),
            "extract_images": bool(extract_images),
            "extract_tables": bool(extract_tables),
            "extract_charts": bool(extract_charts),
            "extract_infographics": bool(extract_infographics),
            "extract_page_as_image": bool(extract_page_as_image),
            "text_depth": str(text_depth),
            "write_json_outputs": bool(write_json_outputs),
            "json_output_dir": str(json_output_dir) if json_output_dir is not None else None,
        },
    )

    fmt = str(output_format or "parquet").strip().lower()
    if fmt == "parquet":
        logger.info("Writing output Parquet shards to %s", output_dir)
        extracted.write_parquet(str(output_dir))
    elif fmt == "jsonl":
        logger.info("Writing output JSONL shards to %s (driver-side)", output_dir)
        _write_jsonl_driver_side(extracted, output_dir=output_dir, rows_per_file=int(jsonl_rows_per_file))
    else:
        raise typer.BadParameter("Unsupported --output-format. Use 'parquet' or 'jsonl'.")

    logger.info("Done.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

