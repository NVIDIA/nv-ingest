# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline lifecycle management for declarative execution.

This module provides high-level lifecycle management for pipelines,
orchestrating configuration resolution, broker setup, and execution
using the configured strategy pattern.
"""

import logging
from typing import Optional

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.framework.orchestration.process.strategies import ProcessExecutionStrategy
from nv_ingest.framework.orchestration.process.dependent_services import start_simple_message_broker

logger = logging.getLogger(__name__)


class PipelineLifecycleManager:
    """
    High-level manager for pipeline lifecycle operations.

    This class orchestrates the complete pipeline lifecycle including
    broker setup, configuration validation, and execution using the
    configured execution strategy.

    Attributes
    ----------
    strategy : ProcessExecutionStrategy
        The execution strategy to use for running pipelines.
    """

    def __init__(self, strategy: ProcessExecutionStrategy):
        """
        Initialize the lifecycle manager with an execution strategy.

        Parameters
        ----------
        strategy : ProcessExecutionStrategy
            The strategy to use for pipeline execution.
        """
        self.strategy = strategy

    def start(self, config: PipelineConfigSchema, options: ExecutionOptions) -> ExecutionResult:
        """
        Start a pipeline using the configured execution strategy.

        This method handles the complete pipeline startup process:
        1. Validate configuration
        2. Start message broker if required
        3. Execute pipeline using the configured strategy

        Parameters
        ----------
        config : PipelineConfigSchema
            Validated pipeline configuration to execute.
        options : ExecutionOptions
            Execution options controlling blocking behavior and output.

        Returns
        -------
        ExecutionResult
            Result containing pipeline interface and/or timing information.

        Raises
        ------
        RuntimeError
            If pipeline startup fails.
        """
        logger.info("Starting pipeline lifecycle")

        try:
            # Start message broker if configured
            self._setup_message_broker(config)

            # Execute pipeline using the configured strategy
            result = self.strategy.execute(config, options)

            logger.info("Pipeline lifecycle started successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to start pipeline lifecycle: {e}")
            raise RuntimeError(f"Pipeline startup failed: {e}") from e

    def _setup_message_broker(self, config: PipelineConfigSchema) -> None:
        """
        Set up message broker if required by configuration.

        Parameters
        ----------
        config : PipelineConfigSchema
            Pipeline configuration containing broker settings.
        """
        if config.pipeline.launch_simple_broker:
            logger.info("Starting simple message broker")
            start_simple_message_broker({})
        else:
            logger.debug("Simple broker launch not required")

    def stop(self, pipeline_id: Optional[str] = None) -> None:
        """
        Stop a running pipeline.

        This method provides a hook for future pipeline stopping functionality.
        Currently, pipeline stopping is handled by the individual interfaces.

        Parameters
        ----------
        pipeline_id : Optional[str]
            Identifier of the pipeline to stop. Currently unused.
        """
        logger.info("Pipeline stop requested")
        # TODO: Implement pipeline stopping logic when needed
        # This would involve coordinating with the execution strategy
        # to gracefully shut down running pipelines
        pass
