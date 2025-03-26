# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC, abstractmethod

import ray
import logging

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


class RayActorSinkStage(RayActorStage, ABC):
    """
    Abstract base class for sink stages in a RayPipeline.
    Sink stages do not support an output queue; instead, they implement write_output
    to deliver their final processed messages.
    """

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        raise NotImplementedError("Sink stages do not support an output queue.")

    @abstractmethod
    def write_output(self, control_message: any) -> any:
        """
        Write the final processed control message to the ultimate destination.
        Must be implemented by concrete sink stages.
        """
        pass

    def _processing_loop(self) -> None:
        while self.running:
            try:
                control_message = self.read_input()
                if control_message is None:
                    time.sleep(self.config.poll_interval)
                    continue
                updated_cm = self.on_data(control_message)
                if updated_cm:
                    self.write_output(updated_cm)
                self.stats["processed"] += 1
            except Exception as e:
                logger.exception(f"Error in sink processing loop: {e}")
                time.sleep(self.config.poll_interval)
