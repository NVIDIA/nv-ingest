from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema


def load_text_embedding_schema_from_dict(cfg: Dict[str, Any]) -> TextEmbeddingSchema:
    return TextEmbeddingSchema(**(cfg or {}))


@dataclass(frozen=True)
class TextEmbeddingStageConfig:
    batch_size: int = 256
    stage_name: str = "text_embedding"
