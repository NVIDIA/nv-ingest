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
import atexit
import multiprocessing
import os
import signal
from typing import Optional

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.framework.orchestration.process.strategies import ProcessExecutionStrategy
from nv_ingest.framework.orchestration.process.strategies import SubprocessStrategy
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
        # Track broker process so we can terminate it during teardown
        self._broker_process: Optional[multiprocessing.Process] = None

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

        # If running pipeline in a subprocess and broker is enabled, ensure the broker
        # is launched in the child process group by signaling via environment variable
        prev_env = None
        set_env = False
        if getattr(config, "pipeline", None) and getattr(config.pipeline, "launch_simple_broker", False):
            if isinstance(self.strategy, SubprocessStrategy):
                prev_env = os.environ.get("NV_INGEST_BROKER_IN_SUBPROCESS")
                os.environ["NV_INGEST_BROKER_IN_SUBPROCESS"] = "1"
                set_env = True

        try:
            # Start message broker if configured (may defer to subprocess based on env)
            self._setup_message_broker(config)

            # Execute pipeline using the configured strategy
            result = self.strategy.execute(config, options)

            logger.info("Pipeline lifecycle started successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to start pipeline lifecycle: {e}")
            raise RuntimeError(f"Pipeline startup failed: {e}") from e
        finally:
            if set_env:
                if prev_env is None:
                    try:
                        del os.environ["NV_INGEST_BROKER_IN_SUBPROCESS"]
                    except KeyError:
                        pass
                else:
                    os.environ["NV_INGEST_BROKER_IN_SUBPROCESS"] = prev_env

    def _setup_message_broker(self, config: PipelineConfigSchema) -> None:
        """
        Set up message broker if required by configuration.

        Parameters
        ----------
        config : PipelineConfigSchema
            Pipeline configuration containing broker settings.
        """
        if config.pipeline.launch_simple_broker:
            # If requested to launch broker inside the subprocess, skip here
            if os.environ.get("NV_INGEST_BROKER_IN_SUBPROCESS") == "1":
                logger.info("Deferring SimpleMessageBroker launch to subprocess")
                return
            logger.info("Starting simple message broker")
            # Start the broker and retain a handle for cleanup.
            # Use defaults (host=0.0.0.0, port=7671) as set by the broker implementation.
            try:
                self._broker_process = start_simple_message_broker({})
                # Ensure cleanup at interpreter shutdown in case caller forgets
                atexit.register(self._terminate_broker_atexit)
                logger.info(f"SimpleMessageBroker started (pid={getattr(self._broker_process, 'pid', None)})")
            except Exception as e:
                logger.error(f"Failed to start SimpleMessageBroker: {e}")
                raise
        else:
            logger.debug("Simple broker launch not required")

    def stop(self, pipeline_id: Optional[str] = None) -> None:
        """
        Stop a running pipeline.

        This method provides a hook for future pipeline stopping functionality.
        Currently, pipeline stopping is handled by the individual interfaces.
        Additionally, it ensures any dependent services (like the simple
        message broker) are terminated to avoid lingering processes.

        Parameters
        ----------
        pipeline_id : Optional[str]
            Identifier of the pipeline to stop. Currently unused.
        """
        logger.info("Pipeline stop requested")
        # Best-effort termination of broker if we started one
        self._terminate_broker()

    # --- Internal helpers ---
    def _terminate_broker_atexit(self) -> None:
        """Atexit-safe broker termination.

        Avoids raising exceptions during interpreter shutdown.
        """
        try:
            self._terminate_broker()
        except Exception:
            # Swallow errors at atexit to avoid noisy shutdowns
            pass

    def _terminate_broker(self) -> None:
        """Terminate the SimpleMessageBroker process if running."""
        proc = self._broker_process
        if not proc:
            return
        try:
            if hasattr(proc, "is_alive") and not proc.is_alive():
                return
        except Exception:
            # If querying state fails, continue with termination attempt
            pass

        pid = getattr(proc, "pid", None)
        logger.info(f"Stopping SimpleMessageBroker (pid={pid})")
        try:
            # First, try graceful terminate
            proc.terminate()
            try:
                proc.join(timeout=3.0)
            except Exception:
                pass

            # If still alive, escalate to SIGKILL on the single process
            still_alive = False
            try:
                still_alive = hasattr(proc, "is_alive") and proc.is_alive()
            except Exception:
                still_alive = True
            if still_alive and pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
                try:
                    proc.join(timeout=2.0)
                except Exception:
                    pass
        finally:
            # Clear handle to avoid repeated attempts
            self._broker_process = None
