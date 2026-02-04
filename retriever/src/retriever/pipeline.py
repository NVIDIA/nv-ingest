from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional

import pandas as pd

from retriever.chart.config import ChartExtractionStageConfig, load_chart_extractor_schema_from_dict
from retriever.chart.stage import extract_chart_data_from_primitives_df
from retriever.infographic.config import (
    InfographicExtractionStageConfig,
    load_infographic_extractor_schema_from_dict,
)
from retriever.infographic.stage import extract_infographic_data_from_primitives_df
from retriever.pdf.config import PDFExtractionStageConfig, load_pdf_extractor_schema_from_dict
from retriever.pdf.io import pdf_files_to_ledger_df
from retriever.pdf.stage import extract_pdf_primitives_from_ledger_df, make_pdf_task_config
from retriever.table.config import TableExtractionStageConfig, load_table_extractor_schema_from_dict
from retriever.table.stage import extract_table_data_from_primitives_df
from retriever.text_embed.config import TextEmbeddingStageConfig, load_text_embedding_schema_from_dict
from retriever.text_embed.stage import embed_text_from_primitives_df
from retriever.vector_store.lancedb_store import LanceDBConfig


Runner = Literal["python", "ray-data"]


@dataclass(frozen=True)
class PDFPipelineResult:
    extracted_df: pd.DataFrame
    info: Dict[str, Any]


def run_pdf_extraction(
    pdf_paths: Iterable[str],
    *,
    runner: Runner = "python",
    stage_cfg: PDFExtractionStageConfig = PDFExtractionStageConfig(),
    extractor_cfg_dict: Dict[str, Any],
    task_cfg: Optional[Dict[str, Any]] = None,
) -> PDFPipelineResult:
    """Run the PDF extraction stage over local PDF files.

    - **Pure Python runner**: no Ray required.
    - **Ray Data runner**: uses `ray.data.Dataset.map_batches` (Ray must be initialized).
    """
    extractor_schema = load_pdf_extractor_schema_from_dict(extractor_cfg_dict)
    df_ledger = pdf_files_to_ledger_df(pdf_paths)

    if task_cfg is None:
        task_cfg = make_pdf_task_config(method="pdfium")

    if runner == "python":
        extracted_df, info = extract_pdf_primitives_from_ledger_df(
            df_ledger,
            task_config=task_cfg,
            extractor_config=extractor_schema,
        )
        return PDFPipelineResult(extracted_df=extracted_df, info=info)

    if runner == "ray-data":
        # Lazy imports so `runner="python"` works without Ray.
        import ray.data  # type: ignore

        from retriever.pdf.ray_data import extract_pdf_primitives_ray_data

        ds = ray.data.from_pandas(df_ledger)
        out = extract_pdf_primitives_ray_data(
            ds,
            task_config=task_cfg,
            extractor_config=extractor_schema,
            batch_size=stage_cfg.batch_size,
        )
        return PDFPipelineResult(extracted_df=out.to_pandas(), info={"runner": "ray-data"})

    raise ValueError(f"Unknown runner: {runner!r}")


@dataclass(frozen=True)
class FullPipelineResult:
    df: pd.DataFrame
    info: Dict[str, Any]


def _safe_pdf_page_count(path: str) -> int:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(path)
        # pypdfium2 documents are typically sized via __len__
        try:
            return int(len(doc))
        except Exception:
            return int(doc.get_page_count())  # type: ignore[attr-defined]
    except Exception:
        # If we can't read page count, fall back to 0 (still shows PDF progress).
        return 0


def run_full_pipeline(
    pdf_paths: Iterable[str],
    *,
    runner: Runner = "python",
    # Per-stage configs
    pdf_stage_cfg: PDFExtractionStageConfig = PDFExtractionStageConfig(),
    infographic_stage_cfg: InfographicExtractionStageConfig = InfographicExtractionStageConfig(),
    table_stage_cfg: TableExtractionStageConfig = TableExtractionStageConfig(),
    chart_stage_cfg: ChartExtractionStageConfig = ChartExtractionStageConfig(),
    embed_stage_cfg: TextEmbeddingStageConfig = TextEmbeddingStageConfig(),
    # Schema dicts (pipeline-yaml-like)
    pdf_extractor_cfg_dict: Dict[str, Any],
    infographic_extractor_cfg_dict: Dict[str, Any],
    table_extractor_cfg_dict: Dict[str, Any],
    chart_extractor_cfg_dict: Dict[str, Any],
    text_embedding_cfg_dict: Dict[str, Any],
    # Task dicts
    pdf_task_cfg: Optional[Dict[str, Any]] = None,
    embed_task_cfg: Optional[Dict[str, Any]] = None,
    lancedb: Optional[LanceDBConfig] = None,
    progress: bool = False,
) -> FullPipelineResult:
    """Run PDF extraction + (infographic/table/chart) enrichment + embedding.

    Notes:
    - The infographic/table/chart stages *enrich* structured primitives produced by PDF extraction.
      If there are no candidates of a given subtype, the stage is effectively a no-op.
    - Ray must be initialized by the caller when runner="ray-data".
    """
    # Build validated schemas once up-front.
    pdf_schema = load_pdf_extractor_schema_from_dict(pdf_extractor_cfg_dict)
    infographic_schema = load_infographic_extractor_schema_from_dict(infographic_extractor_cfg_dict)
    table_schema = load_table_extractor_schema_from_dict(table_extractor_cfg_dict)
    chart_schema = load_chart_extractor_schema_from_dict(chart_extractor_cfg_dict)
    embed_schema = load_text_embedding_schema_from_dict(text_embedding_cfg_dict)

    if pdf_task_cfg is None:
        pdf_task_cfg = make_pdf_task_config(method="pdfium")
    if embed_task_cfg is None:
        embed_task_cfg = {}

    if runner == "python":
        pdf_paths_list = list(pdf_paths)

        if progress:
            from tqdm.auto import tqdm  # type: ignore

            page_counts = [_safe_pdf_page_count(p) for p in tqdm(pdf_paths_list, desc="Counting pages", unit="pdf")]
            total_pages = sum(page_counts)
            pdf_bar = tqdm(total=len(pdf_paths_list), desc="PDFs", unit="pdf")
            page_bar = tqdm(total=total_pages, desc="Pages", unit="page")
            try:
                extracted_chunks: list[pd.DataFrame] = []
                info_pdf_last: Dict[str, Any] = {}
                for i, (p, n_pages) in enumerate(zip(pdf_paths_list, page_counts)):
                    df_ledger = pdf_files_to_ledger_df([p], start_index=i)
                    df_one, info_pdf_last = extract_pdf_primitives_from_ledger_df(
                        df_ledger, task_config=pdf_task_cfg, extractor_config=pdf_schema
                    )
                    if not df_one.empty:
                        extracted_chunks.append(df_one)
                    pdf_bar.update(1)
                    page_bar.update(int(n_pages))
                df = pd.concat(extracted_chunks, ignore_index=True) if extracted_chunks else pd.DataFrame()
                info_pdf = info_pdf_last
            finally:
                pdf_bar.close()
                page_bar.close()
        else:
            df_ledger = pdf_files_to_ledger_df(pdf_paths_list)
            df, info_pdf = extract_pdf_primitives_from_ledger_df(
                df_ledger, task_config=pdf_task_cfg, extractor_config=pdf_schema
            )

        df, info_info = extract_infographic_data_from_primitives_df(
            df, extractor_config=infographic_schema, task_config={}
        )
        df, info_table = extract_table_data_from_primitives_df(df, extractor_config=table_schema, task_config={})
        df, info_chart = extract_chart_data_from_primitives_df(df, extractor_config=chart_schema, task_config={})
        df, info_embed = embed_text_from_primitives_df(
            df,
            transform_config=embed_schema,
            task_config=embed_task_cfg,
            lancedb=lancedb,
        )

        return FullPipelineResult(
            df=df,
            info={
                "runner": "python",
                "pdf": info_pdf,
                "infographic": info_info,
                "table": info_table,
                "chart": info_chart,
                "embed": info_embed,
            },
        )

    if runner == "ray-data":
        import ray.data  # type: ignore

        from retriever.chart.ray_data import extract_chart_data_ray_data
        from retriever.infographic.ray_data import extract_infographic_data_ray_data
        from retriever.pdf.ray_data import extract_pdf_primitives_ray_data
        from retriever.table.ray_data import extract_table_data_ray_data
        from retriever.text_embed.ray_data import embed_text_ray_data

        df_ledger = pdf_files_to_ledger_df(pdf_paths)
        ds = ray.data.from_pandas(df_ledger)

        ds = extract_pdf_primitives_ray_data(
            ds,
            task_config=pdf_task_cfg,
            extractor_config=pdf_schema,
            batch_size=pdf_stage_cfg.batch_size,
        )
        ds = extract_infographic_data_ray_data(
            ds,
            extractor_config=infographic_schema,
            task_config={},
            batch_size=infographic_stage_cfg.batch_size,
        )
        ds = extract_table_data_ray_data(
            ds,
            extractor_config=table_schema,
            task_config={},
            batch_size=table_stage_cfg.batch_size,
        )
        ds = extract_chart_data_ray_data(
            ds,
            extractor_config=chart_schema,
            task_config={},
            batch_size=chart_stage_cfg.batch_size,
        )
        ds = embed_text_ray_data(
            ds,
            transform_config=embed_schema,
            task_config=embed_task_cfg,
            batch_size=embed_stage_cfg.batch_size,
        )

        return FullPipelineResult(df=ds.to_pandas(), info={"runner": "ray-data"})

    raise ValueError(f"Unknown runner: {runner!r}")
