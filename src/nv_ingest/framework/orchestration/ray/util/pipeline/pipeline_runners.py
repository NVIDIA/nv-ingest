# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Union, Optional, TextIO


from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipelineSubprocessInterface,
    RayPipelineInterface,
)
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema

from nv_ingest.pipeline.config.loaders import resolve_pipeline_config, apply_runtime_overrides
from nv_ingest.framework.orchestration.process.lifecycle import PipelineLifecycleManager
from nv_ingest.framework.orchestration.execution.helpers import (
    create_runtime_overrides,
    create_execution_options,
    select_execution_strategy,
)

logger = logging.getLogger(__name__)


def _configure_quiet_mode():
    """
    Configure environment for quiet/production logging in library mode.

    Sets INGEST_RAY_LOG_LEVEL=PRODUCTION if not already set by user, which:
    - Sets Ray logging to ERROR level (suppresses INFO/WARNING)
    - Disables Ray usage stats collection
    - Disables Ray import warnings

    Also silences other common warnings that are noisy in library mode.
    """
    # Only set if user hasn't explicitly configured
    if "INGEST_RAY_LOG_LEVEL" not in os.environ:
        os.environ["INGEST_RAY_LOG_LEVEL"] = "PRODUCTION"

    # Silence Ray accelerator env var warning
    if "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO" not in os.environ:
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    # Disable OTEL tracing export errors (no collector expected in library mode)
    if "OTEL_SDK_DISABLED" not in os.environ:
        os.environ["OTEL_SDK_DISABLED"] = "true"

    # Set nv-ingest module loggers to WARNING to suppress INFO level startup messages
    logging.getLogger("nv_ingest").setLevel(logging.WARNING)
    logging.getLogger("nv_ingest_api").setLevel(logging.WARNING)


def run_pipeline(
    pipeline_config: Optional[PipelineConfigSchema] = None,
    block: bool = True,
    disable_dynamic_scaling: Optional[bool] = None,
    dynamic_memory_threshold: Optional[float] = None,
    run_in_subprocess: bool = False,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    libmode: bool = True,
    quiet: Optional[bool] = None,
) -> Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]:
    """
    Launch and manage a pipeline using configuration.

    This function is the primary entry point for executing a Ray pipeline,
    either within the current process or in a separate Python subprocess.
    It supports synchronous blocking execution or non-blocking lifecycle management,
    and allows redirection of output to specified file-like objects.

    Parameters
    ----------
    pipeline_config : Optional[PipelineConfigSchema], default=None
        The validated configuration object used to construct and launch the pipeline.
        If None and libmode is True, loads the default libmode pipeline.
    block : bool, default=True
        If True, blocks until the pipeline completes.
        If False, returns an interface to control the pipeline externally.
    disable_dynamic_scaling : Optional[bool], default=None
        If provided, overrides the `disable_dynamic_scaling` setting from the pipeline config.
    dynamic_memory_threshold : Optional[float], default=None
        If provided, overrides the `dynamic_memory_threshold` setting from the pipeline config.
    run_in_subprocess : bool, default=False
        If True, launches the pipeline in a separate Python subprocess using `multiprocessing.Process`.
        If False, runs the pipeline in the current process.
    stdout : Optional[TextIO], default=None
        Optional file-like stream to which subprocess stdout should be redirected.
        If None, stdout is redirected to /dev/null.
    stderr : Optional[TextIO], default=None
        Optional file-like stream to which subprocess stderr should be redirected.
        If None, stderr is redirected to /dev/null.
    libmode : bool, default=True
        If True and pipeline_config is None, loads the default libmode pipeline configuration.
        If False, requires pipeline_config to be provided.
    quiet : Optional[bool], default=None
        If True, configures logging for minimal output (PRODUCTION preset, suppresses
        INFO-level startup messages). If None, defaults to True when libmode=True.
        Set to False to see verbose startup logs even in library mode.

    Returns
    -------
    Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]
        - If run in-process with `block=True`: returns elapsed time in seconds (float).
        - If run in-process with `block=False`: returns a `RayPipelineInterface`.
        - If run in subprocess with `block=False`: returns a `RayPipelineSubprocessInterface`.
        - If run in subprocess with `block=True`: returns 0.0.

    Raises
    ------
    ValueError
        If pipeline_config is None and libmode is False.
    RuntimeError
        If the subprocess fails to start or exits with an error.
    Exception
        Any other exceptions raised during pipeline launch or configuration.
    """
    # Configure quiet mode for library mode by default (unless explicitly disabled)
    if quiet is None:
        quiet = libmode
    if quiet:
        _configure_quiet_mode()

    # Resolve configuration
    config = resolve_pipeline_config(pipeline_config, libmode)
    overrides = create_runtime_overrides(disable_dynamic_scaling, dynamic_memory_threshold)
    final_config = apply_runtime_overrides(config, overrides)

    # Select execution strategy
    strategy = select_execution_strategy(run_in_subprocess)
    options = create_execution_options(block, stdout, stderr)

    # Execute using lifecycle manager
    lifecycle_manager = PipelineLifecycleManager(strategy)
    result = lifecycle_manager.start(final_config, options)

    # Return in expected format
    return result.get_return_value()
