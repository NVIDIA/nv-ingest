from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from retriever._local_deps import ensure_nv_ingest_api_importable
from retriever.config_utils import endpoints_from_yaml

ensure_nv_ingest_api_importable()

from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema


def load_pdf_extractor_schema_from_dict(cfg: Dict[str, Any]) -> PDFExtractorSchema:
    """Build `PDFExtractorSchema` from a pipeline-like config dict.

    `config/default_pipeline.yaml` uses list form for endpoints; `PDFExtractorSchema` uses tuples.
    """
    cfg = dict(cfg or {})

    pdfium = cfg.get("pdfium_config")
    if isinstance(pdfium, dict) and "yolox_endpoints" in pdfium:
        pdfium = dict(pdfium)
        pdfium["yolox_endpoints"] = endpoints_from_yaml(pdfium.get("yolox_endpoints"))
        cfg["pdfium_config"] = pdfium

    nemotron = cfg.get("nemotron_parse_config")
    if isinstance(nemotron, dict):
        nemotron = dict(nemotron)
        if "yolox_endpoints" in nemotron:
            nemotron["yolox_endpoints"] = endpoints_from_yaml(nemotron.get("yolox_endpoints"))
        if "nemotron_parse_endpoints" in nemotron:
            nemotron["nemotron_parse_endpoints"] = endpoints_from_yaml(nemotron.get("nemotron_parse_endpoints"))
        cfg["nemotron_parse_config"] = nemotron

    return PDFExtractorSchema(**cfg)


@dataclass(frozen=True)
class PDFExtractionStageConfig:
    """Stage-level knobs for running PDF extraction via pure Python or Ray Data."""

    batch_size: int = 16
    # How we label metrics/spans for this stage.
    stage_name: str = "pdf_extraction"

