# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional
import ray
import logging

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


class RayActorSourceStage(RayActorStage, ABC):
    """
    Abstract base class for source stages in a RayPipeline.
    Source stages do not support an input queue.
    Instead, they must implement get_input() to fetch control messages from an external source.
    """

    def __init__(self, config: Any, log_to_stdout=False, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=log_to_stdout, stage_name=stage_name)
        self.paused = False

    def on_data(self, IngestControlMessage):
        return NotImplemented("Source stages do not implement on_data().")

    @ray.method(num_returns=1)
    def set_input_queue(self, queue_handle: Any) -> bool:
        raise NotImplementedError("Source stages do not support an input queue.")

    @abstractmethod
    def _read_input(self) -> Any:
        """
        For source stages, read_input simply calls get_input().
        """
        pass

    @ray.method(num_returns=1)
    def pause(self) -> bool:
        """
        Pause the source stage so that it will not write to its output queue.
        """
        self.paused = True
        logger.info("Source stage paused.")
        return True

    @ray.method(num_returns=1)
    def resume(self) -> bool:
        """
        Resume the source stage to allow writing to its output queue.
        """
        self.paused = False
        logger.info("Source stage resumed.")
        return True
