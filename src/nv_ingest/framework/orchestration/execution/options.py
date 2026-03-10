# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data classes for pipeline execution configuration and options.

This module defines declarative data structures for configuring pipeline execution,
replacing imperative parameter passing with structured configuration objects.
"""

from dataclasses import dataclass
from typing import Optional, TextIO, Union

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipelineInterface,
    RayPipelineSubprocessInterface,
)


@dataclass
class PipelineRuntimeOverrides:
    """
    Runtime parameter overrides for pipeline configuration.

    These overrides are applied to the base pipeline configuration
    to customize runtime behavior without modifying the source config.

    Attributes
    ----------
    disable_dynamic_scaling : Optional[bool]
        Override for dynamic scaling behavior. If provided, overrides
        the pipeline config's disable_dynamic_scaling setting.
    dynamic_memory_threshold : Optional[float]
        Override for memory threshold used in dynamic scaling decisions.
        Must be between 0.0 and 1.0 if provided.
    """

    disable_dynamic_scaling: Optional[bool] = None
    dynamic_memory_threshold: Optional[float] = None

    def __post_init__(self):
        """Validate override values."""
        if self.dynamic_memory_threshold is not None:
            if not (0.0 <= self.dynamic_memory_threshold <= 1.0):
                raise ValueError(
                    f"dynamic_memory_threshold must be between 0.0 and 1.0, " f"got {self.dynamic_memory_threshold}"
                )


@dataclass
class ExecutionOptions:
    """
    Options controlling pipeline execution behavior.

    These options determine how the pipeline is executed (blocking vs non-blocking)
    and where output is directed for subprocess execution.

    Attributes
    ----------
    block : bool
        If True, blocks until pipeline completes. If False, returns
        immediately with a control interface.
    stdout : Optional[TextIO]
        Stream for subprocess stdout redirection. Only used when
        run_in_subprocess=True. If None, redirected to /dev/null.
    stderr : Optional[TextIO]
        Stream for subprocess stderr redirection. Only used when
        run_in_subprocess=True. If None, redirected to /dev/null.
    """

    block: bool = True
    stdout: Optional[TextIO] = None
    stderr: Optional[TextIO] = None


@dataclass
class ExecutionResult:
    """
    Result of pipeline execution containing interface and timing information.

    This class encapsulates the results of pipeline execution and provides
    methods to convert to the legacy return format for backward compatibility.

    Attributes
    ----------
    interface : Union[RayPipelineInterface, RayPipelineSubprocessInterface, None]
        Pipeline control interface. None for blocking subprocess execution.
    elapsed_time : Optional[float]
        Total execution time in seconds. Only set for blocking execution.
    """

    interface: Union[RayPipelineInterface, RayPipelineSubprocessInterface, None]
    elapsed_time: Optional[float] = None

    def get_return_value(self) -> Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]:
        """
        Convert to legacy return format for backward compatibility.

        Returns
        -------
        Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]
            - If blocking execution: returns elapsed time (float)
            - If non-blocking execution: returns pipeline interface
        """
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.interface is not None:
            return self.interface
        else:
            # This should not happen in normal execution
            raise RuntimeError("ExecutionResult has neither interface nor elapsed_time")
