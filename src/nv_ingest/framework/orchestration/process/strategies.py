# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Process execution strategies for pipeline deployment.

This module defines abstract and concrete strategies for executing pipelines
in different process contexts (in-process vs subprocess), implementing the
Strategy pattern for clean separation of execution concerns.
"""

import atexit
import logging
import multiprocessing
import time
from abc import ABC, abstractmethod

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipelineInterface,
    RayPipelineSubprocessInterface,
)
from nv_ingest.framework.orchestration.process.execution import (
    launch_pipeline,
    run_pipeline_process,
    kill_pipeline_process_group,
)

logger = logging.getLogger(__name__)


class ProcessExecutionStrategy(ABC):
    """
    Abstract base class for pipeline execution strategies.

    This class defines the interface for different ways of executing
    a pipeline (in-process, subprocess, etc.) using the Strategy pattern.
    """

    @abstractmethod
    def execute(self, config: PipelineConfigSchema, options: ExecutionOptions) -> ExecutionResult:
        """
        Execute a pipeline using this strategy.

        Parameters
        ----------
        config : PipelineConfigSchema
            Validated pipeline configuration to execute.
        options : ExecutionOptions
            Execution options controlling blocking behavior and output redirection.

        Returns
        -------
        ExecutionResult
            Result containing pipeline interface and/or timing information.
        """
        pass


class InProcessStrategy(ProcessExecutionStrategy):
    """
    Strategy for executing pipelines in the current process.

    This strategy runs the pipeline directly in the current Python process,
    providing the most direct execution path with minimal overhead.
    """

    def execute(self, config: PipelineConfigSchema, options: ExecutionOptions) -> ExecutionResult:
        """
        Execute pipeline in the current process.

        Parameters
        ----------
        config : PipelineConfigSchema
            Pipeline configuration to execute.
        options : ExecutionOptions
            Execution options. stdout/stderr are ignored for in-process execution.

        Returns
        -------
        ExecutionResult
            Result with pipeline interface (non-blocking) or elapsed time (blocking).
        """
        logger.info("Executing pipeline in current process")

        # Execute the pipeline using existing launch_pipeline function
        # launch_pipeline returns raw RayPipeline object (not wrapped in interface)
        pipeline, total_elapsed = launch_pipeline(
            config,
            block=options.block,
            disable_dynamic_scaling=None,  # Already applied in config
        )

        if options.block:
            logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
            return ExecutionResult(interface=None, elapsed_time=total_elapsed)
        else:
            # Wrap the raw RayPipeline in RayPipelineInterface
            interface = RayPipelineInterface(pipeline)
            return ExecutionResult(interface=interface, elapsed_time=None)


class SubprocessStrategy(ProcessExecutionStrategy):
    """
    Strategy for executing pipelines in a separate subprocess.

    This strategy launches the pipeline in a separate Python process using
    multiprocessing, providing process isolation and output redirection.
    """

    def execute(self, config: PipelineConfigSchema, options: ExecutionOptions) -> ExecutionResult:
        """
        Execute pipeline in a separate subprocess.

        Parameters
        ----------
        config : PipelineConfigSchema
            Pipeline configuration to execute.
        options : ExecutionOptions
            Execution options including output redirection streams.

        Returns
        -------
        ExecutionResult
            Result with subprocess interface (non-blocking) or elapsed time (blocking).
        """
        logger.info("Launching pipeline in Python subprocess using multiprocessing.")

        # Create subprocess using fork context
        ctx = multiprocessing.get_context("fork")
        process = ctx.Process(
            target=run_pipeline_process,
            args=(
                config,
                options.stdout,  # raw_stdout
                options.stderr,  # raw_stderr
            ),
            daemon=False,
        )

        process.start()
        interface = RayPipelineSubprocessInterface(process)

        if options.block:
            # Block until subprocess completes
            start_time = time.time()
            logger.info("Waiting for subprocess pipeline to complete...")
            process.join()
            logger.info("Pipeline subprocess completed.")
            elapsed_time = time.time() - start_time
            return ExecutionResult(interface=None, elapsed_time=elapsed_time)
        else:
            # Return interface for non-blocking execution
            logger.info(f"Pipeline subprocess started (PID={process.pid})")
            # Ensure we pass the Process object, not just the PID, to avoid AttributeError
            # kill_pipeline_process_group expects a multiprocessing.Process instance
            # Capture raw PID to avoid using multiprocessing APIs during interpreter shutdown
            pid = int(process.pid)
            atexit.register(kill_pipeline_process_group, pid)
            return ExecutionResult(interface=interface, elapsed_time=None)


def create_execution_strategy(run_in_subprocess: bool) -> ProcessExecutionStrategy:
    """
    Factory function to create the appropriate execution strategy.

    Parameters
    ----------
    run_in_subprocess : bool
        If True, creates SubprocessStrategy. If False, creates InProcessStrategy.

    Returns
    -------
    ProcessExecutionStrategy
        Configured execution strategy instance.
    """
    if run_in_subprocess:
        return SubprocessStrategy()
    else:
        return InProcessStrategy()
