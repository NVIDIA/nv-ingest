# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Added this no-op UDF ray stage to the pipeline to help speed up the LLM api calls

"""
UDF Parallel Stage - A high-concurrency no-op stage for parallel UDF execution.

This stage does nothing except pass messages through, but with high replica count
it provides a parallel execution pool for UDFs to achieve N-way concurrency.
"""

import logging
from typing import Any, Optional
from pydantic import BaseModel
import ray

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)

logger = logging.getLogger(__name__)


@ray.remote
class UDFParallelStage(RayActorStage):
    """
    A no-op pass-through stage designed for parallel UDF execution.

    This stage simply returns the input message unchanged, but when configured
    with multiple replicas, it provides a high-concurrency pool for UDFs to
    achieve parallel execution without blocking.
    """

    def __init__(self, config: BaseModel, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        logger.info(f"UDFParallelStage initialized: {stage_name}")

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    def on_data(self, message: Any) -> Any:
        """
        Pass-through processing that simply returns the message unchanged.

        The @udf_intercept_hook decorator allows UDFs to target this stage,
        and multiple replicas provide parallel execution capacity.

        Parameters
        ----------
        message : Any
            The incoming control message.

        Returns
        -------
        Any
            The unmodified control message.
        """
        # No-op: just return the message
        return message
