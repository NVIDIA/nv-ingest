# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

from abc import ABC, abstractmethod


class RayActorSourceStage(RayActorStage, ABC):
    """
    Abstract base class for source stages in a RayPipeline.
    Source stages do not support an input queue.
    Instead, they must implement get_input() to fetch control messages from an external source.
    """

    @ray.method(num_returns=1)
    def set_input_queue(self, queue_handle: any) -> bool:
        raise NotImplementedError("Source stages do not support an input queue.")

    def get_input(self) -> any:
        """
        Source stages must implement get_input() to fetch control messages from an external source.
        """
        pass

    @abstractmethod
    def read_input(self) -> any:
        """
        For source stages, read_input simply calls get_input().
        """
        pass
