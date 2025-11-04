# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Optional
from pydantic import BaseModel
import ray

# Import the base class for our stages.
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.schemas.framework_job_counter_schema import JobCounterSchema
from nv_ingest.framework.util.telemetry.global_stats import GlobalStats
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.internal.primitives.tracing.tagging import traceable

# Import the JobCounter schema and global stats singleton.


logger = logging.getLogger(__name__)


@ray.remote
class JobCounterStage(RayActorStage):
    """
    A Ray actor stage that counts jobs and updates global statistics.

    Based on the configuration (a JobCounterSchema instance), it increments a specific
    statistic each time it processes a message.
    """

    def __init__(self, config: BaseModel, stage_name: Optional[str] = None) -> None:
        # Ensure base attributes (e.g. self._running) are initialized.
        super().__init__(config, stage_name=stage_name)
        # The validated config should be a JobCounterSchema instance.
        self.validated_config: JobCounterSchema = config
        # Obtain the global stats' singleton.
        self.stats = GlobalStats.get_instance()

    @nv_ingest_node_failure_try_except()
    @traceable()
    @udf_intercept_hook()
    async def on_data(self, message: Any) -> Any:
        """
        Process an incoming IngestControlMessage by counting jobs.

        If the validated configuration name is "completed_jobs", then if the message metadata
        indicates a failure (cm_failed == True), increments "failed_jobs"; otherwise, increments "completed_jobs".
        For any other configuration name, it increments that statistic.

        Returns the original message.
        """
        logger.debug(f"Performing job counter: {self.validated_config.name}")
        try:
            if self.validated_config.name == "completed_jobs":
                if message.has_metadata("cm_failed") and message.get_metadata("cm_failed"):
                    self.stats.increment_stat("failed_jobs")
                else:
                    self.stats.increment_stat("completed_jobs")
                return message

            self.stats.increment_stat(self.validated_config.name)
            return message
        except Exception as e:
            new_message = f"on_data: Failed to run job counter. Original error: {str(e)}"
            logger.exception(new_message)
            raise type(e)(new_message) from e
