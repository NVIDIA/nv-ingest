#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer

# If this file is executed directly (not via installed package), ensure `retriever/src`
# is on sys.path so `import retriever...` works from a monorepo checkout.
_THIS_FILE = Path(__file__).resolve()
_RETRIEVER_SRC = _THIS_FILE.parents[1]  # .../retriever/src
if (_RETRIEVER_SRC / "retriever").is_dir() and str(_RETRIEVER_SRC) not in sys.path:
    sys.path.insert(0, str(_RETRIEVER_SRC))

from retriever._local_deps import ensure_nv_ingest_api_importable

ensure_nv_ingest_api_importable()

logger = logging.getLogger(__name__)
app = typer.Typer(
    help=(
        "Simple Ray Data batch pipeline:\n"
        "1) CPU: rasterize PDFs into per-page images\n"
        "2) GPU: run Nemotron OCR (HF: nvidia/nemotron-ocr-v1) per page\n"
    )
)

_PAGE_SCHEMA_KEYS = (
    "pdf_path",
    "page_index",
    "page_number",
    "page_image_b64",
    "error",
    "error_detail",
)


@dataclass(frozen=True)
class PageRecord:
    pdf_path: str
    page_index: int  # 0-based
    page_number: int  # 1-based
    page_image_b64: str  # base64-encoded PNG (no data: prefix)

    def to_row(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "page_index": int(self.page_index),
            "page_number": int(self.page_number),
            "page_image_b64": self.page_image_b64,
            "error": None,
            "error_detail": None,
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


def _pdf_to_pages(
    pdf_path: str,
    *,
    render_dpi: int,
    pages_per_batch: int,
    image_format: str = "PNG",
) -> List[Dict[str, Any]]:
    """
    CPU-heavy step: rasterize a single PDF into per-page base64 images.

    Uses the same PDFium stack used by `retriever/pdf/stage.py` (via `pypdfium2`).
    """
    import pypdfium2 as pdfium  # type: ignore

    from nv_ingest_api.util.image_processing.transforms import numpy_to_base64
    from nv_ingest_api.util.pdf.pdfium import pdfium_pages_to_numpy

    path = Path(pdf_path)
    if not path.is_file():
        return [
            {
                "pdf_path": str(path),
                "page_index": None,
                "page_number": None,
                "page_image_b64": "",
                "error": "missing_file",
                "error_detail": None,
            }
        ]

    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception as e:
        return [
            {
                "pdf_path": str(path),
                "page_index": None,
                "page_number": None,
                "page_image_b64": "",
                "error": "open_failed",
                "error_detail": str(e),
            }
        ]

    out: List[Dict[str, Any]] = []
    try:
        n_pages = int(len(doc))

        # Process pages in small chunks to amortize render overhead.
        i = 0
        while i < n_pages:
            j = min(i + int(pages_per_batch), n_pages)
            pages = [doc.get_page(k) for k in range(i, j)]
            try:
                imgs, _pads = pdfium_pages_to_numpy(pages, render_dpi=int(render_dpi))
                for k, arr in enumerate(imgs):
                    page_index = i + k
                    page_number = page_index + 1
                    b64 = numpy_to_base64(arr, format=str(image_format))
                    rec = PageRecord(
                        pdf_path=str(path),
                        page_index=page_index,
                        page_number=page_number,
                        page_image_b64=b64,
                    )
                    out.append(rec.to_row())
            finally:
                for p in pages:
                    try:
                        p.close()
                    except Exception:
                        pass
            i = j
    except Exception as e:
        out.append(
            {
                "pdf_path": str(path),
                "page_index": None,
                "page_number": None,
                "page_image_b64": "",
                "error": "render_failed",
                "error_detail": str(e),
            }
        )
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return out


def _pdf_to_pages_row(
    row: Dict[str, Any],
    *,
    render_dpi: int,
    pages_per_batch: int,
) -> List[Dict[str, Any]]:
    pdf_path = row.get("pdf_path")
    if not isinstance(pdf_path, str) or not pdf_path:
        return [
            {
                "pdf_path": str(pdf_path) if pdf_path is not None else "",
                "page_index": None,
                "page_number": None,
                "page_image_b64": "",
                "error": "missing_pdf_path",
                "error_detail": None,
            }
        ]
    return _pdf_to_pages(
        pdf_path,
        render_dpi=int(render_dpi),
        pages_per_batch=int(pages_per_batch),
    )


def _ocr_output_to_text(obj: Any) -> str:
    """
    Best-effort extraction of text from `nemotron_ocr` outputs.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for k in ("text", "output_text", "generated_text", "ocr_text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Common structured output: {"texts": [...]}
        texts = obj.get("texts")
        if isinstance(texts, list):
            parts = [t for t in texts if isinstance(t, str) and t.strip()]
            if parts:
                return " ".join(parts).strip()
    if isinstance(obj, list):
        parts = [_ocr_output_to_text(x) for x in obj]
        parts = [p for p in parts if p]
        return "\n".join(parts).strip()
    return str(obj).strip()


class NemotronOcrBatchFn:
    """
    Stateful Ray Data map_batches fn that runs on GPU (actor-based).

    Loads the HF model repo (default: nvidia/nemotron-ocr-v1) once per actor.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_dir: Optional[str],
        model_cache_dir: Optional[str],
        merge_level: str,
    ) -> None:
        self._merge_level = str(merge_level)

        resolved_dir: Optional[str] = str(model_dir) if model_dir else None
        if resolved_dir is None:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "huggingface_hub is required to download the OCR model. "
                    "Install dependencies or pass --model-dir to a local copy."
                ) from e

            dl_kwargs: Dict[str, Any] = {"repo_id": str(model_id)}
            if model_cache_dir:
                # Force a stable per-run directory so `NemotronOCR` sees a normal filesystem path.
                cache_dir = Path(model_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                dl_kwargs["local_dir"] = str(cache_dir / model_id.replace("/", "__"))
                dl_kwargs["local_dir_use_symlinks"] = False

            resolved_dir = snapshot_download(**dl_kwargs)

        # Prefer the existing local wrapper if available; it expects a model directory.
        try:
            from retriever.model.local.nemotron_ocr_v1 import NemotronOCRV1
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Nemotron OCR runtime is not available. This pipeline currently expects "
                "`nemotron_ocr` via the existing `retriever.model.local.NemotronOCRV1` wrapper.\n"
                "Either install the OCR runtime, or adjust this script to use a Transformers pipeline."
            ) from e

        self._model = NemotronOCRV1(model_dir=str(resolved_dir))

    def __call__(self, batch):
        """
        Ray Data map_batches handler.

        We intentionally avoid pandas here because some environments ship a pandas build
        that breaks Ray's internal pandas conversion utilities.
        """
        import pyarrow as pa  # type: ignore

        if isinstance(batch, pa.Table):
            d = batch.to_pydict()
        elif isinstance(batch, dict):
            d = dict(batch)
        else:
            # Fall back to best-effort conversion (Ray may pass other batch types).
            try:
                d = dict(batch)  # type: ignore[arg-type]
            except Exception:
                d = {"_raw": [str(batch)]}

        # Ensure all expected keys exist so output schema is stable.
        n = 0
        for v in d.values():
            if isinstance(v, list):
                n = len(v)
                break
        for k in _PAGE_SCHEMA_KEYS:
            if k not in d:
                d[k] = [None] * n

        images = d.get("page_image_b64", [])
        prior_errs = d.get("error", [None] * n)

        texts: List[str] = []
        ocr_errors: List[Optional[str]] = []

        for i in range(n):
            if prior_errs[i] not in {None, ""}:
                texts.append("")
                ocr_errors.append("skipped_input_error")
                continue

            b64 = images[i]
            if not isinstance(b64, str) or not b64:
                texts.append("")
                ocr_errors.append("missing_page_image_b64")
                continue
            try:
                out = self._model.invoke(b64, merge_level=self._merge_level)
                texts.append(_ocr_output_to_text(out))
                ocr_errors.append(None)
            except Exception as e:
                texts.append("")
                ocr_errors.append(str(e))

        d["ocr_text"] = texts
        d["ocr_error"] = ocr_errors
        return pa.Table.from_pydict(d)


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
    # CPU rasterization
    render_dpi: int = typer.Option(200, "--render-dpi", min=50, max=1200, help="PDF page render DPI (CPU step)."),
    pages_per_batch: int = typer.Option(
        4, "--pages-per-batch", min=1, help="How many pages to render at once per PDF worker."
    ),
    cpu_per_pdf: int = typer.Option(1, "--cpu-per-pdf", min=1, help="Ray CPUs reserved per PDF render task."),
    # GPU OCR
    model_id: str = typer.Option(
        "nvidia/nemotron-ocr-v1",
        "--model-id",
        help="HuggingFace model repo id for Nemotron OCR v1.",
    ),
    model_dir: Optional[Path] = typer.Option(
        None, "--model-dir", exists=True, file_okay=False, dir_okay=True, help="Use a local model directory."
    ),
    model_cache_dir: Optional[Path] = typer.Option(
        None,
        "--model-cache-dir",
        file_okay=False,
        dir_okay=True,
        help="Optional directory to download/cache the HF model into.",
    ),
    merge_level: str = typer.Option(
        "paragraph", "--merge-level", help="OCR merge level (e.g. word/sentence/paragraph)."
    ),
    ocr_batch_size: int = typer.Option(4, "--ocr-batch-size", min=1, help="Ray Data batch size for OCR stage."),
    ocr_actors: int = typer.Option(1, "--ocr-actors", min=1, help="Number of GPU OCR actors (1 GPU each)."),
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
    import ray.data  # type: ignore

    logging.basicConfig(level=logging.INFO)

    pdfs = _iter_pdf_paths(input_dir=input_dir, pdf_list=pdf_list, limit_pdfs=limit_pdfs)
    if not pdfs:
        raise typer.BadParameter("No PDFs found. Provide --input-dir and/or --pdf-list.")

    ray.init(address=ray_address, ignore_reinit_error=True)

    logger.info("Building dataset: pdfs=%s", len(pdfs))
    ds = ray.data.from_items([{"pdf_path": p} for p in pdfs])

    logger.info("CPU stage: rasterize PDFs into pages")
    split_fn = partial(_pdf_to_pages_row, render_dpi=int(render_dpi), pages_per_batch=int(pages_per_batch))
    pages = ds.flat_map(
        split_fn,
        num_cpus=float(cpu_per_pdf),
    )

    logger.info("GPU stage: Nemotron OCR (actors=%s, batch_size=%s)", ocr_actors, ocr_batch_size)
    # ocr = pages.map_batches(
    #     NemotronOcrBatchFn,
    #     batch_format="pyarrow",
    #     batch_size=int(ocr_batch_size),
    #     compute=ray.data.ActorPoolStrategy(size=int(ocr_actors)),
    #     num_gpus=1,
    #     fn_constructor_kwargs={
    #         "model_id": str(model_id),
    #         "model_dir": str(model_dir) if model_dir is not None else None,
    #         "model_cache_dir": str(model_cache_dir) if model_cache_dir is not None else None,
    #         "merge_level": str(merge_level),
    #     },
    # )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(ds.count(pages))
    # pages.write_json(str(output_dir))
    # fmt = str(output_format or "parquet").strip().lower()
    # if fmt == "parquet":
    #     logger.info("Writing output Parquet shards to %s", output_dir)
    #     ocr.write_parquet(str(output_dir))
    # elif fmt == "jsonl":
    #     logger.info("Writing output JSONL shards to %s (driver-side)", output_dir)
    #     _write_jsonl_driver_side(ocr, output_dir=output_dir, rows_per_file=int(jsonl_rows_per_file))
    # else:
    #     raise typer.BadParameter("Unsupported --output-format. Use 'parquet' or 'jsonl'.")
    logger.info("Done.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

