# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except


@ray.remote
class DefaultDrainSink(RayActorSinkStage):
    def __init__(self, config: Any, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=False, stage_name=stage_name)

        self._last_sunk_count = 0
        self._sunk_count = 0

    @nv_ingest_node_failure_try_except()
    def on_data(self, message: IngestControlMessage) -> IngestControlMessage:
        self._sunk_count += 1

        return message

    @ray.method(num_returns=1)
    def get_stats(self) -> Dict[str, Any]:
        delta = self._sunk_count - self._last_sunk_count
        self._last_sunk_count = self._sunk_count

        return {
            "active_processing": False,
            "delta_processed": delta,
            "elapsed": 0.0,
            "prcessing_rate_cps": 0.0,
            "processed": self._sunk_count,
            "successful_queue_reads": self.stats.get("successful_queue_reads", 0),
            "successful_queue_writes": 0,
            "queue_full": 0,
        }
