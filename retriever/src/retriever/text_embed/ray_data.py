# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from .text_embed import embed_text_1b_v2

logger = logging.getLogger(__name__)


def embed_text_ray_data(
    ds: "ray.data.Dataset",  # noqa: F821
    *,
    model: Any,
    task_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 256,
) -> "ray.data.Dataset":  # noqa: F821
    """
    Ray Data adapter for lightweight local text embedding.

    Notes:
    - This is independent of `nv-ingest-api`.
    - For actor-based embedding (so the embedder/model stays resident), use `TextEmbedActor`
      from `retriever.text_embed.text_embed` with Ray's ActorPoolStrategy.
    """
    import ray.data  # type: ignore  # noqa: F401

    task_config = dict(task_config or {})

    def _map_batch(batch: pd.DataFrame) -> pd.DataFrame:
        return embed_text_1b_v2(batch, model=model, **task_config)

    logger.info("Running Ray Data local text embedding: batch_size=%s", batch_size)
    return ds.map_batches(_map_batch, batch_format="pandas", batch_size=batch_size)
