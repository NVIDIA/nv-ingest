# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from retriever.config_utils import endpoints_from_yaml

from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema


def load_infographic_extractor_schema_from_dict(cfg: Dict[str, Any]) -> InfographicExtractorSchema:
    cfg = dict(cfg or {})
    endpoint_cfg = cfg.get("endpoint_config")
    if isinstance(endpoint_cfg, dict) and "ocr_endpoints" in endpoint_cfg:
        endpoint_cfg = dict(endpoint_cfg)
        ocr_endpoints = endpoints_from_yaml(endpoint_cfg.get("ocr_endpoints"))

        # Special-case: treat "no endpoints" as "use local model".
        # The nv-ingest-api schema forbids endpoint_config with both endpoints empty,
        # so we drop endpoint_config entirely to allow stage code to choose local fallback.
        if not (ocr_endpoints[0] or ocr_endpoints[1]):
            cfg.pop("endpoint_config", None)
        else:
            endpoint_cfg["ocr_endpoints"] = ocr_endpoints
            cfg["endpoint_config"] = endpoint_cfg
    return InfographicExtractorSchema(**cfg)


@dataclass(frozen=True)
class InfographicExtractionStageConfig:
    batch_size: int = 64
    stage_name: str = "infographic_extraction"
