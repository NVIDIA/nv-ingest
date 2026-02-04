from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .stage import embed_text_from_primitives_df

logger = logging.getLogger(__name__)


def embed_text_ray_data(
    ds: "ray.data.Dataset",
    *,
    transform_config: Any,
    task_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 256,
) -> "ray.data.Dataset":
    """Ray Data adapter for text embedding (mutates metadata in-place)."""
    import ray.data  # type: ignore

    def _map_batch(batch: pd.DataFrame) -> pd.DataFrame:
        out, _info = embed_text_from_primitives_df(
            batch,
            transform_config=transform_config,
            task_config=task_config,
        )
        return out

    logger.info("Running Ray Data text embedding: batch_size=%s", batch_size)
    return ds.map_batches(_map_batch, batch_format="pandas", batch_size=batch_size)
