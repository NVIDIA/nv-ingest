# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod, ABC
from typing import Any

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage


class RayActorSourceStage(RayActorStage, ABC):
    """
    Abstract base class for source stages in a RayPipeline.
    Source stages do not support an input edge.

    Instead, they must implement get_input() to fetch control messages from an external source.
    """

    def set_input_edge(self, edge_handle: Any) -> bool:
        raise NotImplementedError("Source stages do not support an input edge.")

    @abstractmethod
    async def read_input(self) -> Any:
        """
        Source stages must implement get_input() to fetch control messages from an external source.
        """
        pass
