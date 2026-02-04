"""
Table extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/table primitives by populating
`metadata.table_metadata.table_content` using OCR (and optionally YOLOX).
"""

from .config import TableExtractionStageConfig, load_table_extractor_schema_from_dict
from .ray_data import extract_table_data_ray_data
from .stage import extract_table_data_from_primitives_df

__all__ = [
    "TableExtractionStageConfig",
    "extract_table_data_from_primitives_df",
    "extract_table_data_ray_data",
    "load_table_extractor_schema_from_dict",
]

