from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from .stage import extract_infographic_data_from_primitives_df

logger = logging.getLogger(__name__)


def extract_infographic_data_ray_data(
    ds: "ray.data.Dataset",
    *,
    extractor_config: Any,
    task_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
) -> "ray.data.Dataset":
    """Ray Data adapter for infographic extraction (mutates metadata in-place)."""
    import ray.data  # type: ignore

    def _map_batch(batch: pd.DataFrame) -> pd.DataFrame:
        out, _info = extract_infographic_data_from_primitives_df(
            batch,
            extractor_config=extractor_config,
            task_config=task_config,
        )
        return out

    logger.info("Running Ray Data infographic extraction: batch_size=%s", batch_size)
    return ds.map_batches(_map_batch, batch_format="pandas", batch_size=batch_size)
