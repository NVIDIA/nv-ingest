# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional, Any

import pandas as pd
import ray
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.orchestration.ray.util.pipeline.mixins import SingletonStageMixin
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except


@ray.remote
class GatherStage(RayActorStage, SingletonStageMixin):
    """
    A Ray actor stage that gathers fragments produced by scatter stages.

    This stage accumulates incoming fragments using `fragment_id` metadata and,
    once all fragments are received, concatenates their payloads (as DataFrames)
    into a single message. The original `fragment_id` is cleared and the combined
    payload is set on the base message (fragment 0), which is then forwarded.
    """

    def __init__(self, config: BaseModel) -> None:
        super().__init__(config)
        self._fragment_cache: Dict[str, List[Optional[Any]]] = {}

    @traceable("gather_stage")
    @nv_ingest_node_failure_try_except(annotation_id="gather_stage", raise_on_failure=True)
    def on_data(self, control_message: Any) -> Optional[Any]:
        fragment_id = control_message.get_metadata("fragment_id")

        if fragment_id is None:
            return control_message

        fragment_count = control_message.get_metadata("fragment_count")
        fragment_index = control_message.get_metadata("fragment_index")

        if fragment_count is None or fragment_index is None:
            raise ValueError("Fragmented message missing required metadata.")

        if fragment_id not in self._fragment_cache:
            self._fragment_cache[fragment_id] = [None] * fragment_count

        cache = self._fragment_cache[fragment_id]
        cache[fragment_index] = control_message

        if any(part is None for part in cache):
            return []  # Still waiting

        # Coalesce and emit
        dfs = [msg.payload() for msg in cache]
        combined_df = pd.concat(dfs, ignore_index=True)

        base_msg = cache[0]
        base_msg.set_metadata("fragment_id", None)
        base_msg.set_metadata("fragment_count", None)
        base_msg.set_metadata("fragment_index", None)
        base_msg.payload(combined_df)

        del self._fragment_cache[fragment_id]

        self._logger.info(
            f"GatherStage: Assembled {fragment_count} fragments into message with {combined_df.shape[0]} rows"
        )

        return base_msg
