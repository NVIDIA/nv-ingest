# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Any, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from nv_ingest_api.internal.enums.common import PipelinePhase


class StageType(str, Enum):
    """
    The type of a pipeline stage.
    """

    SOURCE = "source"
    STAGE = "stage"
    SINK = "sink"


class ReplicaCalculationStrategy(str, Enum):
    """
    Strategy for calculating replica counts at runtime.
    """

    STATIC = "static"  # Fixed number of replicas
    CPU_PERCENTAGE = "cpu_percentage"  # Percentage of available CPU cores
    MEMORY_THRESHOLDING = "memory_thresholding"  # Based on memory allocation per replica
    MEMORY_STATIC_GLOBAL_PERCENT = "memory_static_global_percent"  # Memory-safe calculation with budget limits


class ReplicaStrategyConfig(BaseModel):
    """
    Configuration for a specific replica calculation strategy.

    Attributes
    ----------
    strategy : ReplicaCalculationStrategy
        The calculation strategy to use.
    value : Optional[Union[int, float]]
        The primary value for the strategy (e.g., static count, CPU percentage).
    limit : Optional[int]
        Optional upper limit for calculated replicas.
    cpu_percent : Optional[float]
        CPU percentage for CPU_PERCENTAGE strategy (0.0 to 1.0).
    memory_per_replica_mb : Optional[int]
        Expected memory usage per replica in MB.
    memory_threshold_percent : Optional[float]
        Memory threshold percentage for MEMORY_THRESHOLDING strategy (0.0 to 1.0).
    max_memory_budget_mb : Optional[int]
        Maximum memory budget for MEMORY_STATIC_GLOBAL_PERCENT strategy in MB.
    """

    strategy: ReplicaCalculationStrategy = Field(..., description="The calculation strategy to use.")
    value: Optional[Union[int, float]] = Field(None, description="Primary value for the strategy.")
    limit: Optional[int] = Field(None, description="Optional upper limit for calculated replicas.", ge=1)
    cpu_percent: Optional[float] = Field(
        None, description="CPU percentage for CPU_PERCENTAGE strategy.", ge=0.0, le=1.0
    )
    memory_per_replica_mb: Optional[int] = Field(None, description="Expected memory usage per replica in MB.", gt=0)
    memory_threshold_percent: Optional[float] = Field(
        None, description="Memory threshold percentage for MEMORY_THRESHOLDING strategy.", ge=0.0, le=1.0
    )
    max_memory_budget_mb: Optional[int] = Field(
        None, description="Maximum memory budget for MEMORY_STATIC_GLOBAL_PERCENT strategy in MB.", gt=0
    )

    @model_validator(mode="after")
    def validate_strategy_config(self):
        """Validate that required fields are present for each strategy."""
        if self.strategy == ReplicaCalculationStrategy.STATIC:
            if self.value is None or not isinstance(self.value, int):
                raise ValueError("STATIC strategy requires 'value' as an integer")
        elif self.strategy == ReplicaCalculationStrategy.CPU_PERCENTAGE:
            if self.cpu_percent is None:
                if self.value is None or not isinstance(self.value, (int, float)):
                    raise ValueError("CPU_PERCENTAGE strategy requires 'cpu_percent' or 'value' as a float")
                self.cpu_percent = float(self.value)
        elif self.strategy == ReplicaCalculationStrategy.MEMORY_THRESHOLDING:
            if self.memory_per_replica_mb is None:
                raise ValueError("MEMORY_THRESHOLDING strategy requires 'memory_per_replica_mb'")
        elif self.strategy == ReplicaCalculationStrategy.MEMORY_STATIC_GLOBAL_PERCENT:
            if self.memory_per_replica_mb is None:
                raise ValueError("MEMORY_STATIC_GLOBAL_PERCENT strategy requires 'memory_per_replica_mb'")
            # max_memory_budget_mb is optional - uses global static_memory_threshold if not provided
        return self


class ReplicaConfig(BaseModel):
    """
    Configuration for stage replicas supporting both dynamic and static scaling modes.

    Defines the min/max number of replicas for a stage, either as absolute counts,
    percentages of total CPU cores, or resource-based calculations. Supports different
    configurations for dynamic vs static scaling modes.

    Attributes
    ----------
    cpu_count_min : Optional[int]
        Absolute minimum number of replicas. Must be >= 0. (Legacy support)
    cpu_count_max : Optional[int]
        Absolute maximum number of replicas. Must be >= 1. (Legacy support)
    cpu_percent_min : Optional[float]
        Minimum number of replicas as a percentage (0.0 to 1.0) of total cores. (Legacy support)
    cpu_percent_max : Optional[float]
        Maximum number of replicas as a percentage (0.0 to 1.0) of total cores. (Legacy support)
    min_replicas : Optional[int]
        Minimum number of replicas for both scaling modes. Must be >= 0.
    max_replicas : Optional[Union[int, ReplicaStrategyConfig]]
        Maximum replicas for dynamic scaling mode. Can be static int or strategy config.
    static_replicas : Optional[Union[int, ReplicaStrategyConfig]]
        Replica configuration for static scaling mode. Can be static int or strategy config.
    """

    # Legacy fields for backward compatibility
    cpu_count_min: Optional[int] = Field(None, description="Absolute minimum number of replicas.", ge=0)
    cpu_count_max: Optional[int] = Field(None, description="Absolute maximum number of replicas.", ge=1)
    cpu_percent_min: Optional[float] = Field(
        None, description="Minimum number of replicas as a percentage of total cores.", ge=0.0, le=1.0
    )
    cpu_percent_max: Optional[float] = Field(
        None, description="Maximum number of replicas as a percentage of total cores.", ge=0.0, le=1.0
    )

    # New flexible replica configuration
    min_replicas: Optional[int] = Field(None, description="Minimum number of replicas.", ge=0)
    max_replicas: Optional[Union[int, ReplicaStrategyConfig]] = Field(
        None, description="Maximum replicas for dynamic scaling mode."
    )
    static_replicas: Optional[Union[int, ReplicaStrategyConfig]] = Field(
        None, description="Replica configuration for static scaling mode."
    )

    @model_validator(mode="after")
    def check_exclusive_min_max(self) -> "ReplicaConfig":
        """
        Validates that replica configuration is consistent and complete.

        Ensures that:
        1. Legacy fields (cpu_count_*, cpu_percent_*) are not mixed with new fields
        2. At least one configuration method is specified
        3. Min/max relationships are valid
        """
        legacy_fields = [self.cpu_count_min, self.cpu_count_max, self.cpu_percent_min, self.cpu_percent_max]
        new_fields = [self.min_replicas, self.max_replicas, self.static_replicas]

        has_legacy = any(field is not None for field in legacy_fields)
        has_new = any(field is not None for field in new_fields)

        if has_legacy and has_new:
            raise ValueError(
                "Cannot mix legacy replica fields (cpu_count_*, cpu_percent_*) with new fields "
                "(min_replicas, max_replicas, static_replicas). Use one approach or the other."
            )

        if not has_legacy and not has_new:
            # Set sensible defaults for new configuration
            self.min_replicas = 0
            self.max_replicas = 1

        # Legacy validation (existing logic)
        if has_legacy:
            if self.cpu_count_min is not None and self.cpu_percent_min is not None:
                raise ValueError("Cannot specify both cpu_count_min and cpu_percent_min")
            if self.cpu_count_max is not None and self.cpu_percent_max is not None:
                raise ValueError("Cannot specify both cpu_count_max and cpu_percent_max")

            # Validate min <= max for legacy fields
            if self.cpu_count_min is not None and self.cpu_count_max is not None:
                if self.cpu_count_min > self.cpu_count_max:
                    raise ValueError("cpu_count_min cannot be greater than cpu_count_max")
            if self.cpu_percent_min is not None and self.cpu_percent_max is not None:
                if self.cpu_percent_min > self.cpu_percent_max:
                    raise ValueError("cpu_percent_min cannot be greater than cpu_percent_max")

        # New configuration validation
        if has_new:
            # Validate min_replicas against max_replicas if both are static integers
            if (
                self.min_replicas is not None
                and isinstance(self.max_replicas, int)
                and self.min_replicas > self.max_replicas
            ):
                raise ValueError("min_replicas cannot be greater than max_replicas")

            # Validate min_replicas against static_replicas if both are static integers
            if (
                self.min_replicas is not None
                and isinstance(self.static_replicas, int)
                and self.min_replicas > self.static_replicas
            ):
                raise ValueError("min_replicas cannot be greater than static_replicas")

        return self

    model_config = ConfigDict(extra="forbid")


class StageConfig(BaseModel):
    """
    Configuration for a single pipeline stage.

    Describes a single component in the ingestion pipeline, including its name,
    type, actor implementation, and specific configuration.

    Attributes
    ----------
    name : str
        A unique name to identify the stage within the pipeline.
    type : StageType
        The type of the stage, which determines how it's added to the RayPipeline.
    phase: PipelinePhase
        The logical phase of the stage in the pipeline.
    actor : Optional[str]
        The fully qualified import path to the actor class or function that
        implements the stage's logic. Mutually exclusive with 'callable'.
    callable : Optional[str]
        The fully qualified import path to a callable function that
        implements the stage's logic. Mutually exclusive with 'actor'.
    task_filters: Optional[List[Any]]
        List of task types this callable stage should filter for. Only applies to callable stages.
        Supports both simple strings (e.g., "udf") and complex filters (e.g., ["udf", {"phase": 5}]).
    enabled : bool
        A flag to indicate whether the stage should be included in the pipeline.
        If False, the stage and its connected edges are ignored.
    config : Dict[str, Any]
        A dictionary of configuration parameters passed to the stage's actor.
    replicas : ReplicaConfig
        The replica configuration for the stage.
    runs_after: List[str]
        A list of stage names that this stage must be downstream of.
    """

    name: str = Field(..., description="Unique name for the stage.")
    type: StageType = Field(StageType.STAGE, description="Type of the stage.")
    phase: PipelinePhase = Field(..., description="The logical phase of the stage.")
    actor: Optional[str] = Field(None, description="Full import path to the stage's actor class or function.")
    callable: Optional[str] = Field(None, description="Full import path to a callable function for the stage.")
    task_filters: Optional[List[Any]] = Field(
        None, description="List of task types this callable stage should filter for. Only applies to callable stages."
    )
    enabled: bool = Field(True, description="Whether the stage is enabled.")
    config: Dict[str, Any] = Field({}, description="Configuration dictionary for the stage.")
    replicas: ReplicaConfig = Field(default_factory=ReplicaConfig, description="Replica configuration.")
    runs_after: List[str] = Field(default_factory=list, description="List of stages this stage must run after.")

    @model_validator(mode="after")
    def check_actor_or_callable(self) -> "StageConfig":
        """
        Validates that exactly one of 'actor' or 'callable' is specified.
        """
        if self.actor is None and self.callable is None:
            raise ValueError("Either 'actor' or 'callable' must be specified")

        if self.actor is not None and self.callable is not None:
            raise ValueError("Cannot specify both 'actor' and 'callable' - they are mutually exclusive")

        return self

    model_config = ConfigDict(extra="forbid")


class EdgeConfig(BaseModel):
    """
    Configuration for an edge between two stages.

    Defines a connection from a source stage to a destination stage, including
    the size of the intermediate queue.

    Attributes
    ----------
    from_stage : str
        The name of the source stage for the edge.
    to_stage : str
        The name of the destination stage for the edge.
    queue_size : int
        The maximum number of items in the queue between the two stages.
    """

    from_stage: str = Field(..., alias="from", description="The name of the source stage.")
    to_stage: str = Field(..., alias="to", description="The name of the destination stage.")
    queue_size: int = Field(100, gt=0, description="The size of the queue between stages.")

    model_config = ConfigDict(extra="forbid")


class PIDControllerConfig(BaseModel):
    """
    Configuration for the PID controller used in dynamic scaling.

    Attributes
    ----------
    kp : float
        Proportional gain for the PID controller.
    ki : float
        Integral gain for the PID controller.
    ema_alpha : float
        Exponential moving average alpha for the PID controller.
    target_queue_depth : int
        Target queue depth for the PID controller.
    penalty_factor : float
        Penalty factor for the PID controller.
    error_boost_factor : float
        Error boost factor for the PID controller.
    rcm_memory_safety_buffer_fraction : float
        Resource constraint manager memory safety buffer fraction.
    """

    kp: float = Field(0.2, gt=0.0, description="Proportional gain for the PID controller.")
    ki: float = Field(0.01, ge=0.0, description="Integral gain for the PID controller.")
    ema_alpha: float = Field(
        0.1, ge=0.0, le=1.0, description="Exponential moving average alpha for the PID controller."
    )
    target_queue_depth: int = Field(0, ge=0, description="Target queue depth for the PID controller.")
    penalty_factor: float = Field(0.1, ge=0.0, description="Penalty factor for the PID controller.")
    error_boost_factor: float = Field(1.5, gt=0.0, description="Error boost factor for the PID controller.")
    rcm_memory_safety_buffer_fraction: float = Field(
        0.15, ge=0.0, le=1.0, description="Resource constraint manager memory safety buffer fraction."
    )

    model_config = ConfigDict(extra="forbid")


class PipelineRuntimeConfig(BaseModel):
    """
    Configuration for pipeline runtime behavior.

    Parameters
    ----------
    disable_dynamic_scaling : bool
        Whether to disable dynamic scaling of replicas (default: False).
    dynamic_memory_threshold : float
        The memory utilization threshold (0.0 to 1.0) for dynamic scaling decisions.
    static_memory_threshold : float
        Global memory threshold for static scaling mode (default: 0.75).
    pid_controller : PIDControllerConfig
        PID controller configuration for dynamic scaling.
    launch_simple_broker : bool
        If True, launches a simple message broker for the pipeline.
    """

    disable_dynamic_scaling: bool = Field(False, description="Disable dynamic scaling of stage replicas.")
    dynamic_memory_threshold: float = Field(
        0.75, ge=0.0, le=0.95, description="Memory utilization threshold for dynamic scaling."
    )
    static_memory_threshold: float = Field(
        0.75, ge=0.0, le=1.0, description="Global memory threshold for static scaling mode."
    )
    pid_controller: PIDControllerConfig = Field(
        default_factory=PIDControllerConfig, description="PID controller configuration for dynamic scaling."
    )
    launch_simple_broker: bool = Field(False, description="Launch a simple message broker for the pipeline.")

    model_config = ConfigDict(extra="forbid")


class PipelineConfigSchema(BaseModel):
    """
    Root configuration model for an ingestion pipeline.

    This model represents the entire declarative configuration for an ingestion
    pipeline, including all stages and the edges that connect them.

    Attributes
    ----------
    name : str
        The name of the pipeline.
    description : str
        A description of the pipeline.
    stages : List[StageConfig]
        A list of all stage configurations in the pipeline.
    edges : List[EdgeConfig]
        A list of all edge configurations that define the pipeline's topology.
    pipeline: Optional[PipelineRuntimeConfig] = Field(default_factory=PipelineRuntimeConfig,
        description="Runtime configuration for the pipeline.")
    """

    name: str = Field(..., description="The name of the pipeline.")
    description: str = Field(..., description="A description of the pipeline.")
    stages: List[StageConfig] = Field(..., description="List of all stages in the pipeline.")
    edges: List[EdgeConfig] = Field(..., description="List of all edges connecting the stages.")
    pipeline: Optional[PipelineRuntimeConfig] = Field(
        default_factory=PipelineRuntimeConfig, description="Runtime configuration for the pipeline."
    )

    @field_validator("stages", "edges")
    def check_not_empty(cls, v: list) -> list:
        """Validates that the list is not empty."""
        if not v:
            raise ValueError("must not be empty")
        return v

    def get_phases(self) -> Set[PipelinePhase]:
        """Returns a set of all unique phases in the pipeline."""
        return {stage.phase for stage in self.stages}

    model_config = ConfigDict(extra="forbid")
