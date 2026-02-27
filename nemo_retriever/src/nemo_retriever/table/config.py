# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from nemo_retriever.config_utils import endpoints_from_yaml

from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema


def load_table_extractor_schema_from_dict(cfg: Dict[str, Any]) -> TableExtractorSchema:
    cfg = dict(cfg or {})
    endpoint_cfg = cfg.get("endpoint_config")
    if isinstance(endpoint_cfg, dict):
        endpoint_cfg = dict(endpoint_cfg)
        if "yolox_endpoints" in endpoint_cfg:
            endpoint_cfg["yolox_endpoints"] = endpoints_from_yaml(endpoint_cfg.get("yolox_endpoints"))
        if "ocr_endpoints" in endpoint_cfg:
            endpoint_cfg["ocr_endpoints"] = endpoints_from_yaml(endpoint_cfg.get("ocr_endpoints"))
        cfg["endpoint_config"] = endpoint_cfg
    return TableExtractorSchema(**cfg)


@dataclass(frozen=True)
class TableExtractionStageConfig:
    batch_size: int = 64
    stage_name: str = "table_extraction"
