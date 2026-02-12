from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from retriever.config_utils import endpoints_from_yaml

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema


def load_chart_extractor_schema_from_dict(cfg: Dict[str, Any]) -> ChartExtractorSchema:
    cfg = dict(cfg or {})
    endpoint_cfg = cfg.get("endpoint_config")
    if isinstance(endpoint_cfg, dict):
        endpoint_cfg = dict(endpoint_cfg)
        if "yolox_endpoints" in endpoint_cfg:
            endpoint_cfg["yolox_endpoints"] = endpoints_from_yaml(endpoint_cfg.get("yolox_endpoints"))
        if "ocr_endpoints" in endpoint_cfg:
            endpoint_cfg["ocr_endpoints"] = endpoints_from_yaml(endpoint_cfg.get("ocr_endpoints"))
        cfg["endpoint_config"] = endpoint_cfg
    return ChartExtractorSchema(**cfg)


@dataclass(frozen=True)
class ChartExtractionStageConfig:
    batch_size: int = 64
    stage_name: str = "chart_extraction"
