"""
Text embedding stage (pure Python + Ray Data adapters).

This stage calls `nv-ingest-api`'s `transform_create_text_embeddings_internal`
to populate `metadata.embedding` (or a custom target field).
"""

from .text_embed import TextEmbedActor, embed_text_1b_v2

__all__ = ["TextEmbedActor", "embed_text_1b_v2"]

# Optional imports: the "full" embedding stage depends on nv-ingest-api (and its deps).
# Keep lightweight embedding importable even in minimal environments.
try:  # pragma: no cover
    from .config import TextEmbeddingStageConfig, load_text_embedding_schema_from_dict
    from .ray_data import embed_text_ray_data
    from .stage import embed_text_from_primitives_df

    __all__ += [
        "TextEmbeddingStageConfig",
        "embed_text_from_primitives_df",
        "embed_text_ray_data",
        "load_text_embedding_schema_from_dict",
    ]
except Exception:
    pass
