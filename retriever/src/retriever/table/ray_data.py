from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd

from retriever.metrics import LoggingMetrics, Metrics

from .stage import extract_table_data_from_primitives_df

logger = logging.getLogger(__name__)


def extract_table_data_ray_data(
    ds: "ray.data.Dataset",
    *,
    extractor_config: Any,
    task_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 64,
    metrics_factory: Optional[Callable[[], Metrics]] = None,
) -> "ray.data.Dataset":
    """Ray Data adapter for table extraction (mutates metadata in-place)."""
    import ray.data  # type: ignore

    if metrics_factory is None:
        metrics_factory = lambda: LoggingMetrics(extra={"runner": "ray_data"})  # noqa: E731

    def _map_batch(batch: pd.DataFrame) -> pd.DataFrame:
        metrics = metrics_factory()
        out, _info = extract_table_data_from_primitives_df(
            batch,
            extractor_config=extractor_config,
            task_config=task_config,
            metrics=metrics,
        )
        return out

    logger.info("Running Ray Data table extraction: batch_size=%s", batch_size)
    return ds.map_batches(_map_batch, batch_format="pandas", batch_size=batch_size)

