# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any
from pydantic import BaseModel
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


@ray.remote
class ThroughputMonitorStage(RayActorStage):
    """
    A Ray actor stage that monitors throughput by counting messages.

    Every 100 messages, it calculates the throughput (messages per second) and logs the measure.
    It also adds the throughput as metadata on the control message before passing it on.
    """

    def __init__(self, config: BaseModel) -> None:
        # Initialize base attributes (e.g., self._running, self.start_time) via the base class.
        super().__init__(config)
        self.count = 0
        self.last_emit_time = None  # Timestamp when the last throughput measure was emitted

    async def on_data(self, message: Any) -> Any:
        """
        Process an incoming control message. Increment the internal counter and, every 100 messages,
        calculate and log the throughput. The throughput value is also added to the message metadata.

        Parameters
        ----------
        message : Any
            The incoming control message.

        Returns
        -------
        Any
            The (possibly modified) control message.
        """
        self.count += 1
        if self.last_emit_time is None:
            self.last_emit_time = time.time()

        if self.count % 1000 == 0:
            now = time.time()
            elapsed = now - self.last_emit_time
            throughput = 1000 / elapsed if elapsed > 0 else 0
            logger.warning(
                f"ThroughputMonitorStage: Processed {self.count} messages. Throughput: {throughput:.2f} messages/sec"
            )
            try:
                # Attempt to add throughput information to the message metadata.
                message.set_metadata("throughput", throughput)
            except Exception:
                # If the message doesn't support metadata, skip.
                pass
            self.last_emit_time = now

        return message
