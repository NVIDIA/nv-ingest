# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for pipeline execution configuration.

This module contains generic helper functions for converting individual parameters
into structured configuration objects, supporting the declarative execution architecture.
"""

from typing import Optional, TextIO

from nv_ingest.framework.orchestration.execution.options import PipelineRuntimeOverrides, ExecutionOptions
from nv_ingest.framework.orchestration.process.strategies import ProcessExecutionStrategy, create_execution_strategy


def create_runtime_overrides(
    disable_dynamic_scaling: Optional[bool], dynamic_memory_threshold: Optional[float]
) -> PipelineRuntimeOverrides:
    """
    Create runtime override object from individual parameters.

    This function converts the individual override parameters into
    a structured PipelineRuntimeOverrides object for declarative processing.

    Parameters
    ----------
    disable_dynamic_scaling : Optional[bool]
        Dynamic scaling override value.
    dynamic_memory_threshold : Optional[float]
        Memory threshold override value.

    Returns
    -------
    PipelineRuntimeOverrides
        Structured override object containing the provided values.
    """
    return PipelineRuntimeOverrides(
        disable_dynamic_scaling=disable_dynamic_scaling, dynamic_memory_threshold=dynamic_memory_threshold
    )


def create_execution_options(block: bool, stdout: Optional[TextIO], stderr: Optional[TextIO]) -> ExecutionOptions:
    """
    Create execution options object from individual parameters.

    This function converts individual execution parameters into
    a structured ExecutionOptions object for declarative processing.

    Parameters
    ----------
    block : bool
        Whether to block until pipeline completion.
    stdout : Optional[TextIO]
        Output stream for subprocess redirection.
    stderr : Optional[TextIO]
        Error stream for subprocess redirection.

    Returns
    -------
    ExecutionOptions
        Structured options object containing the provided values.
    """
    return ExecutionOptions(block=block, stdout=stdout, stderr=stderr)


def select_execution_strategy(run_in_subprocess: bool) -> ProcessExecutionStrategy:
    """
    Select appropriate execution strategy based on parameters.

    This function encapsulates the logic for choosing between
    in-process and subprocess execution strategies.

    Parameters
    ----------
    run_in_subprocess : bool
        Whether to run in a subprocess.

    Returns
    -------
    ProcessExecutionStrategy
        Configured execution strategy instance.
    """
    return create_execution_strategy(run_in_subprocess=run_in_subprocess)
