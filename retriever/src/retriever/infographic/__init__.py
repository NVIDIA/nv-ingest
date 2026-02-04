"""
Infographic extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/infographic primitives by populating
`metadata.table_metadata.table_content` via OCR, using `nv-ingest-api` internals.
"""

from .config import InfographicExtractionStageConfig, load_infographic_extractor_schema_from_dict
from .ray_data import extract_infographic_data_ray_data
from .stage import extract_infographic_data_from_primitives_df

__all__ = [
    "InfographicExtractionStageConfig",
    "extract_infographic_data_from_primitives_df",
    "extract_infographic_data_ray_data",
    "load_infographic_extractor_schema_from_dict",
]

