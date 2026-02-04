"""
Chart extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/chart primitives by populating
`metadata.table_metadata.table_content` using YOLOX + OCR fusion.
"""

from .config import ChartExtractionStageConfig, load_chart_extractor_schema_from_dict
from .ray_data import extract_chart_data_ray_data
from .stage import extract_chart_data_from_primitives_df

from .__main__ import app

__all__ = [
    "app",
    "ChartExtractionStageConfig",
    "extract_chart_data_from_primitives_df",
    "extract_chart_data_ray_data",
    "load_chart_extractor_schema_from_dict",
]
