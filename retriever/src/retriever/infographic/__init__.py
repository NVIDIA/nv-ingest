"""
Infographic extraction stage (pure Python + Ray Data adapters).

This stage enriches existing STRUCTURED/infographic primitives by populating
`metadata.table_metadata.table_content` via OCR, using `nv-ingest-api` internals.
"""

from .infographic_detection import InfographicDetectionActor, detect_infographic_elements_v1

__all__ = ["InfographicDetectionActor", "detect_infographic_elements_v1"]

# Optional imports: infographic *extraction* depends on nv-ingest-api (and its deps).
# We keep detection importable even in lightweight environments.
try:  # pragma: no cover
    from .config import InfographicExtractionStageConfig, load_infographic_extractor_schema_from_dict
    from .ray_data import extract_infographic_data_ray_data
    from .stage import extract_infographic_data_from_primitives_df

    __all__ += [
        "InfographicExtractionStageConfig",
        "extract_infographic_data_from_primitives_df",
        "extract_infographic_data_ray_data",
        "load_infographic_extractor_schema_from_dict",
    ]
except Exception:
    pass

