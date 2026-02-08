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

from retriever.chart.config import load_chart_extractor_schema_from_dict
from retriever.chart.stage import extract_chart_data_from_primitives_df
from retriever.pdf.config import load_pdf_extractor_schema_from_dict
from retriever.pdf.stage import extract_pdf_primitives_from_ledger_df, make_pdf_task_config
from retriever.table.config import load_table_extractor_schema_from_dict
from retriever.table.stage import extract_table_data_from_primitives_df
from retriever.text_embed.config import load_text_embedding_schema_from_dict
from retriever.text_embed.stage import embed_text_from_primitives_df

logger = logging.getLogger(__name__)
app = typer.Typer(
    help=(
        "Ray Data batch pipeline for staged ingestion.\n"
        "\n"
        "1) Ingest: read all PDFs into a Ray Dataset with `read_binary*`\n"
        "2) Actor stages: stage1 PDF extraction + stage3 table + stage4 chart + stage5 text embeddings\n"
        "3) Driver sink (optional): stage6 upload embeddings to LanceDB for later querying\n"
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


def _read_yaml_mapping(path: Optional[Path]) -> Dict[str, Any]:
    """
    Read a YAML mapping from disk, returning {} when unset/empty.

    We read YAML on the driver and pass dicts into Ray actors to avoid depending on
    shared filesystem paths across nodes.
    """
    if path is None:
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise typer.BadParameter(f"Failed reading YAML config {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"YAML config must be a mapping/object at top-level: {path}")
    return data


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


class TableExtractionActorBatchFn:
    """
    Actor-based stage3: table extraction enrichment over primitives DataFrames.
    """

    def __init__(self, *, config_dict: Dict[str, Any]) -> None:
        ensure_nv_ingest_api_importable()
        self._schema = load_table_extractor_schema_from_dict(config_dict or {})

    def __call__(self, batch: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        out_df, _info = extract_table_data_from_primitives_df(batch, extractor_config=self._schema, task_config={})
        if out_df is None or not isinstance(out_df, pd.DataFrame):
            return batch
        return out_df


class ChartExtractionActorBatchFn:
    """
    Actor-based stage4: chart extraction enrichment over primitives DataFrames.
    """

    def __init__(self, *, config_dict: Dict[str, Any]) -> None:
        ensure_nv_ingest_api_importable()
        self._schema = load_chart_extractor_schema_from_dict(config_dict or {})

    def __call__(self, batch: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        out_df, _info = extract_chart_data_from_primitives_df(batch, extractor_config=self._schema, task_config={})
        if out_df is None or not isinstance(out_df, pd.DataFrame):
            return batch
        return out_df


class TextEmbeddingActorBatchFn:
    """
    Actor-based stage5: text embeddings enrichment over primitives DataFrames.

    This keeps any local HF embedder (if used) loaded once per actor.
    """

    def __init__(
        self,
        *,
        config_dict: Dict[str, Any],
        task_config: Dict[str, Any],
    ) -> None:
        ensure_nv_ingest_api_importable()

        self._schema = load_text_embedding_schema_from_dict(config_dict or {})
        self._task_cfg = dict(task_config or {})

        # If no endpoint URL is present, force local HuggingFace embeddings and keep the
        # embedder loaded once per actor (avoids per-batch model reload).
        endpoint_url = self._task_cfg.get("endpoint_url")
        if endpoint_url is None:
            try:
                from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import (
                    LlamaNemotronEmbed1BV2Embedder,
                )

                local_device = self._task_cfg.get("local_hf_device")
                local_cache_dir = self._task_cfg.get("local_hf_cache_dir")
                local_batch_size = int(self._task_cfg.get("local_hf_batch_size") or 64)

                embedder = LlamaNemotronEmbed1BV2Embedder(
                    device=str(local_device) if local_device is not None else None,
                    hf_cache_dir=str(local_cache_dir) if local_cache_dir is not None else None,
                    normalize=True,
                )

                def _embed(texts):
                    vecs = embedder.embed(texts, batch_size=local_batch_size)
                    return vecs.tolist()

                # Force the API transform to use the callable path.
                self._task_cfg["endpoint_url"] = None
                self._task_cfg["embedder"] = _embed
                self._task_cfg["local_batch_size"] = local_batch_size
            except Exception as e:
                raise RuntimeError(
                    "Local HF embeddings requested (no --embed-endpoint-url), but local embedder init failed. "
                    "Verify torch/CUDA and the `llama_nemotron_embed_1b_v2` package are installed."
                ) from e

    def __call__(self, batch: "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd

        out_df, _info = embed_text_from_primitives_df(
            batch,
            transform_config=self._schema,
            task_config=self._task_cfg,
        )
        if out_df is None or not isinstance(out_df, pd.DataFrame):
            return batch
        return out_df


@dataclass(frozen=True)
class LanceDBUploadConfig:
    uri: str
    table_name: str
    overwrite: bool
    create_index: bool
    index_type: str
    metric: str
    num_partitions: int
    num_sub_vectors: int


def _infer_vector_dim_from_lancedb_rows(rows: List[Dict[str, Any]]) -> int:
    for r in rows:
        v = r.get("vector")
        if isinstance(v, list) and v:
            return int(len(v))
    return 0


def _upload_embeddings_to_lancedb_driver_side(
    ds: Any,
    *,
    cfg: LanceDBUploadConfig,
    batch_size: int = 2048,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Driver-side upload of embeddings into LanceDB.

    This mirrors the intent of `retriever.vector_store.stage` / stage6, but uploads directly
    from the final Ray Dataset (after stage5) instead of reading `*.text_embeddings.json` files.

    We reuse `nv_ingest_client.util.vdb.lancedb.create_lancedb_results(...)` to build rows with:
      - vector, text, metadata (JSON string), source (JSON string)
    so stage7 can query the resulting LanceDB table.
    """
    try:
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore
        from nv_ingest_client.util.vdb.lancedb import create_lancedb_results
    except Exception as e:
        raise RuntimeError(
            "LanceDB upload requested but dependencies are missing. "
            "Ensure `lancedb`, `pyarrow`, and `nv_ingest_client` are installed."
        ) from e

    db = lancedb.connect(uri=str(cfg.uri))
    table = None
    schema = None

    total_rows_written = 0
    batches_seen = 0
    batches_skipped = 0

    for batch in ds.iter_batches(batch_format="pandas", batch_size=int(batch_size)):
        batches_seen += 1
        if limit_batches is not None and batches_seen > int(limit_batches):
            break

        # Convert to element dicts compatible with `create_lancedb_results`.
        try:
            elements = batch.to_dict(orient="records")
        except Exception:
            elements = []

        # `create_lancedb_results` expects: list[result], each result is list[element]
        lancedb_rows: List[Dict[str, Any]] = create_lancedb_results([elements])
        if not lancedb_rows:
            batches_skipped += 1
            continue

        if table is None:
            dim = _infer_vector_dim_from_lancedb_rows(lancedb_rows)
            if dim <= 0:
                batches_skipped += 1
                continue

            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("text", pa.string()),
                    pa.field("metadata", pa.string()),
                    pa.field("source", pa.string()),
                ]
            )

            mode = "overwrite" if cfg.overwrite else "append"
            table = db.create_table(str(cfg.table_name), data=lancedb_rows, schema=schema, mode=mode)
            total_rows_written += len(lancedb_rows)
            continue

        # Subsequent batches: append rows.
        try:
            table.add(lancedb_rows)
        except Exception:
            # Fall back: open and add (handles older/newer LanceDB behaviors).
            table = db.open_table(str(cfg.table_name))
            table.add(lancedb_rows)
        total_rows_written += len(lancedb_rows)

    if table is None:
        return {
            "uploaded": False,
            "reason": "no_embeddings_rows_found",
            "batches_seen": batches_seen,
            "batches_skipped": batches_skipped,
            "rows_written": 0,
            "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name},
        }

    if cfg.create_index:
        try:
            table.create_index(
                index_type=str(cfg.index_type),
                metric=str(cfg.metric),
                num_partitions=int(cfg.num_partitions),
                num_sub_vectors=int(cfg.num_sub_vectors),
                vector_column_name="vector",
            )
            # Best-effort wait when supported.
            try:
                for stub in table.list_indices():
                    table.wait_for_index([stub.name])
            except Exception:
                pass
        except TypeError:
            table.create_index(vector_column_name="vector")

    return {
        "uploaded": True,
        "batches_seen": batches_seen,
        "batches_skipped": batches_skipped,
        "rows_written": int(total_rows_written),
        "lancedb": {"uri": cfg.uri, "table_name": cfg.table_name, "overwrite": cfg.overwrite},
    }


def _write_jsonl_driver_side(
    ds: Any, *, output_dir: Path, rows_per_file: int = 50_000, file_ext: str = "jsonl"
) -> None:
    """
    Write JSON Lines without pandas by streaming Arrow batches on the driver.

    Ray's built-in `Dataset.write_json()` currently converts blocks to pandas, which can
    fail in environments with incompatible pandas builds. This avoids pandas entirely.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    row_idx_in_file = 0
    ext = str(file_ext or "jsonl").lstrip(".")
    f = (output_dir / f"part-{file_idx:05d}.{ext}").open("w", encoding="utf-8")
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
                    f = (output_dir / f"part-{file_idx:05d}.{ext}").open("w", encoding="utf-8")
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
    pdf_gpus_per_actor: float = typer.Option(
        1.0, "--pdf-gpus-per-actor", min=0.0, help="GPUs reserved per PDF extraction actor."
    ),
    # Stage3 (table extraction)
    run_table: bool = typer.Option(True, "--table/--no-table", help="Enable stage3 table extraction."),
    table_config: Optional[Path] = typer.Option(
        None,
        "--table-config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config for table stage.",
    ),
    table_batch_size: int = typer.Option(64, "--table-batch-size", min=1, help="Ray Data batch size for table stage."),
    table_actors: int = typer.Option(1, "--table-actors", min=1, help="Number of table extraction actors."),
    table_cpus_per_actor: float = typer.Option(8.0, "--table-cpus-per-actor", min=0.0, help="CPUs per table actor."),
    table_gpus_per_actor: float = typer.Option(1.0, "--table-gpus-per-actor", min=0.0, help="GPUs per table actor."),
    # Stage4 (chart extraction)
    run_chart: bool = typer.Option(True, "--chart/--no-chart", help="Enable stage4 chart extraction."),
    chart_config: Optional[Path] = typer.Option(
        None,
        "--chart-config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config for chart stage.",
    ),
    chart_batch_size: int = typer.Option(64, "--chart-batch-size", min=1, help="Ray Data batch size for chart stage."),
    chart_actors: int = typer.Option(1, "--chart-actors", min=1, help="Number of chart extraction actors."),
    chart_cpus_per_actor: float = typer.Option(8.0, "--chart-cpus-per-actor", min=0.0, help="CPUs per chart actor."),
    chart_gpus_per_actor: float = typer.Option(1.0, "--chart-gpus-per-actor", min=0.0, help="GPUs per chart actor."),
    # Stage5 (text embeddings)
    run_embed: bool = typer.Option(True, "--embed/--no-embed", help="Enable stage5 text embedding."),
    embed_config: Optional[Path] = typer.Option(
        None,
        "--embed-config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Optional YAML config for TextEmbeddingSchema.",
    ),
    embed_api_key: Optional[str] = typer.Option(
        None, "--embed-api-key", help="Optional API key override for embedding."
    ),
    embed_endpoint_url: Optional[str] = typer.Option(
        None,
        "--embed-endpoint-url",
        help="Optional embedding endpoint override (use 'none' to force local HF fallback).",
    ),
    embed_model_name: Optional[str] = typer.Option(
        None, "--embed-model-name", help="Optional embedding model name override."
    ),
    embed_dimensions: Optional[int] = typer.Option(
        None, "--embed-dimensions", help="Optional embedding dimensions override."
    ),
    embed_local_hf_fallback: bool = typer.Option(
        True, "--embed-local-hf-fallback/--no-embed-local-hf-fallback", help="Enable local HF fallback if no endpoint."
    ),
    embed_local_hf_device: Optional[str] = typer.Option(
        None, "--embed-local-hf-device", help="Device for local HF embeddings (e.g. 'cuda', 'cpu', 'cuda:0')."
    ),
    embed_local_hf_cache_dir: Optional[Path] = typer.Option(
        None, "--embed-local-hf-cache-dir", file_okay=False, dir_okay=True, help="Optional HF cache dir for embeddings."
    ),
    embed_local_hf_batch_size: int = typer.Option(
        64, "--embed-local-hf-batch-size", min=1, help="Batch size for local HF embedding inference."
    ),
    embed_batch_size: int = typer.Option(
        256, "--embed-batch-size", min=1, help="Ray Data batch size for embedding stage."
    ),
    embed_actors: int = typer.Option(1, "--embed-actors", min=1, help="Number of embedding actors."),
    embed_cpus_per_actor: float = typer.Option(
        4.0, "--embed-cpus-per-actor", min=0.0, help="CPUs per embedding actor."
    ),
    embed_gpus_per_actor: float = typer.Option(
        1.0, "--embed-gpus-per-actor", min=0.0, help="GPUs per embedding actor."
    ),
    # Stage6 (vector DB upload; driver-side)
    vdb_upload: bool = typer.Option(
        False, "--vdb-upload/--no-vdb-upload", help="Upload embeddings to LanceDB (stage6)."
    ),
    vdb_upload_batch_size: int = typer.Option(
        2048, "--vdb-upload-batch-size", min=1, help="Driver-side batch size for LanceDB upload."
    ),
    lancedb_uri: str = typer.Option("lancedb", "--lancedb-uri", help="LanceDB database URI (directory path)."),
    lancedb_table_name: str = typer.Option("nv-ingest", "--table-name", help="LanceDB table name."),
    lancedb_overwrite: bool = typer.Option(
        True,
        "--overwrite/--append",
        help="Overwrite table (default) or append to existing.",
    ),
    lancedb_create_index: bool = typer.Option(
        True, "--create-index/--no-create-index", help="Create a vector index after upload."
    ),
    lancedb_index_type: str = typer.Option("IVF_HNSW_SQ", "--index-type", help="LanceDB index type."),
    lancedb_metric: str = typer.Option("l2", "--metric", help="Distance metric for the index."),
    lancedb_num_partitions: int = typer.Option(16, "--num-partitions", min=1, help="Index partitions."),
    lancedb_num_sub_vectors: int = typer.Option(256, "--num-sub-vectors", min=1, help="Index sub-vectors."),
    materialize: bool = typer.Option(
        True,
        "--materialize/--no-materialize",
        help="Materialize the final dataset once to avoid recomputing for multiple sinks (output + vdb upload).",
    ),
    # Output
    output_format: str = typer.Option(
        "json",
        "--output-format",
        help="Output format: 'json' (default), 'jsonl', or 'parquet'.",
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

    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "NEMOTRON_OCR_MODEL_DIR": "/home/local/jdyer/Development/nv-ingest/models/nemotron-ocr-v1/checkpoints/"
            }
        },
    )

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
        num_gpus=float(pdf_gpus_per_actor),
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

    if run_table:
        table_cfg_dict = _read_yaml_mapping(table_config)
        logger.info("Actor stage: table extraction (actors=%s, batch_size=%s)", table_actors, table_batch_size)
        extracted = extracted.map_batches(
            TableExtractionActorBatchFn,
            batch_format="pandas",
            batch_size=int(table_batch_size),
            num_cpus=float(table_cpus_per_actor),
            num_gpus=float(table_gpus_per_actor),
            compute=rd.ActorPoolStrategy(size=int(table_actors)),
            fn_constructor_kwargs={"config_dict": table_cfg_dict},
        )

    if run_chart:
        chart_cfg_dict = _read_yaml_mapping(chart_config)
        logger.info("Actor stage: chart extraction (actors=%s, batch_size=%s)", chart_actors, chart_batch_size)
        extracted = extracted.map_batches(
            ChartExtractionActorBatchFn,
            batch_format="pandas",
            batch_size=int(chart_batch_size),
            num_cpus=float(chart_cpus_per_actor),
            num_gpus=float(chart_gpus_per_actor),
            compute=rd.ActorPoolStrategy(size=int(chart_actors)),
            fn_constructor_kwargs={"config_dict": chart_cfg_dict},
        )

    if run_embed:
        embed_cfg_dict = _read_yaml_mapping(embed_config)
        # Task-config overrides (takes precedence over schema defaults).
        embed_task_cfg: Dict[str, Any] = {}
        if embed_api_key is not None:
            embed_task_cfg["api_key"] = embed_api_key
        # IMPORTANT:
        # - If --embed-endpoint-url is NOT provided, we intentionally force local HF embeddings by setting
        #   endpoint_url=None (even if the schema has a default endpoint configured).
        # - If it IS provided, use it (with 'none' meaning force-local).
        if embed_endpoint_url is None:
            embed_task_cfg["endpoint_url"] = None
        else:
            v = embed_endpoint_url.strip()
            embed_task_cfg["endpoint_url"] = None if v.lower() in ("", "none", "null") else v
        if embed_model_name is not None:
            embed_task_cfg["model_name"] = embed_model_name
        if embed_dimensions is not None:
            embed_task_cfg["dimensions"] = int(embed_dimensions)
        embed_task_cfg["use_local_hf_if_no_endpoint"] = bool(embed_local_hf_fallback)
        if embed_local_hf_device is not None:
            embed_task_cfg["local_hf_device"] = str(embed_local_hf_device)
        if embed_local_hf_cache_dir is not None:
            embed_task_cfg["local_hf_cache_dir"] = str(embed_local_hf_cache_dir)
        if embed_local_hf_batch_size is not None:
            embed_task_cfg["local_hf_batch_size"] = int(embed_local_hf_batch_size)

        logger.info("Actor stage: text embeddings (actors=%s, batch_size=%s)", embed_actors, embed_batch_size)
        extracted = extracted.map_batches(
            TextEmbeddingActorBatchFn,
            batch_format="pandas",
            batch_size=int(embed_batch_size),
            num_cpus=float(embed_cpus_per_actor),
            num_gpus=float(embed_gpus_per_actor),
            compute=rd.ActorPoolStrategy(size=int(embed_actors)),
            fn_constructor_kwargs={"config_dict": embed_cfg_dict, "task_config": embed_task_cfg},
        )

    if vdb_upload and not run_embed:
        raise typer.BadParameter("--vdb-upload requires --embed (stage5) to be enabled.")

    # If we have multiple sinks, materialize once to avoid recomputation.
    if materialize and (vdb_upload or str(output_format or "").strip()):
        if hasattr(extracted, "materialize"):
            extracted = extracted.materialize()

    fmt = str(output_format or "parquet").strip().lower()
    if fmt == "parquet":
        logger.info("Writing output Parquet shards to %s", output_dir)
        extracted.write_parquet(str(output_dir))
    elif fmt in {"json", "jsonl"}:
        # Ray's write_json writes JSON Lines shards; it may use pandas internally in some versions.
        try:
            logger.info("Writing output JSON shards to %s", output_dir)
            extracted.write_json(str(output_dir))
        except Exception:
            logger.exception("Dataset.write_json failed; falling back to driver-side JSONL writer")
            ext = "json" if fmt == "json" else "jsonl"
            _write_jsonl_driver_side(
                extracted,
                output_dir=output_dir,
                rows_per_file=int(jsonl_rows_per_file),
                file_ext=ext,
            )
    else:
        raise typer.BadParameter("Unsupported --output-format. Use 'json', 'jsonl', or 'parquet'.")

    if vdb_upload:
        logger.info("Driver stage: uploading embeddings to LanceDB uri=%s table=%s", lancedb_uri, lancedb_table_name)
        info = _upload_embeddings_to_lancedb_driver_side(
            extracted,
            cfg=LanceDBUploadConfig(
                uri=str(lancedb_uri),
                table_name=str(lancedb_table_name),
                overwrite=bool(lancedb_overwrite),
                create_index=bool(lancedb_create_index),
                index_type=str(lancedb_index_type),
                metric=str(lancedb_metric),
                num_partitions=int(lancedb_num_partitions),
                num_sub_vectors=int(lancedb_num_sub_vectors),
            ),
            batch_size=int(vdb_upload_batch_size),
        )
        logger.info("LanceDB upload done: %s", info)

    logger.info("Done.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
