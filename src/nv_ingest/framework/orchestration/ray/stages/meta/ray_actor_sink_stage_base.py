# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


class RayActorSinkStage(RayActorStage, ABC):
    """
    Abstract base class for sink stages in a RayPipeline.
    Sink stages do not support an output edge; instead, they implement write_output
    to deliver their final processed messages.
    """

    def set_output_edge(self, edge_handle: Any) -> bool:
        raise NotImplementedError("Sink stages do not support an output edge.")

    @abstractmethod
    async def write_output(self, control_message: Any) -> Any:
        """
        Write the final processed control message to the ultimate destination.
        Must be implemented by concrete sink stages.
        """
        pass

    async def _processing_loop(self) -> None:
        while self.running:
            try:
                control_message = await self.read_input()
                if control_message is None:
                    await asyncio.sleep(self.config.poll_interval)
                    continue
                updated_cm = await self.on_data(control_message)
                if updated_cm:
                    await self.write_output(updated_cm)
                self.stats["processed"] += 1
            except Exception as e:
                logger.exception(f"Error in sink processing loop: {e}")
                await asyncio.sleep(self.config.poll_interval)
