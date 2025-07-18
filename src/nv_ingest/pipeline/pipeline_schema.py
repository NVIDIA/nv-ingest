# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field, Extra, root_validator, field_validator


class StageType(str, Enum):
    """
    The type of a pipeline stage.
    """

    SOURCE = "source"
    STAGE = "stage"
    SINK = "sink"


class PipelinePhase(int, Enum):
    """
    The logical phase of a pipeline stage.
    """

    PRE_PROCESSING = 0
    EXTRACTION = 1
    POST_PROCESSING = 2
    MUTATION = 3
    TRANSFORM = 4
    RESPONSE = 5


class ReplicaConfig(BaseModel):
    """
    Configuration for stage replicas.

    Defines the min/max number of replicas for a stage, either as an absolute
    count or a percentage of total CPU cores.

    Attributes
    ----------
    cpu_count_min : Optional[int]
        Absolute minimum number of replicas. Must be >= 0.
    cpu_count_max : Optional[int]
        Absolute maximum number of replicas. Must be >= 1.
    cpu_percent_min : Optional[float]
        Minimum number of replicas as a percentage (0.0 to 1.0) of total cores.
    cpu_percent_max : Optional[float]
        Maximum number of replicas as a percentage (0.0 to 1.0) of total cores.
    """

    cpu_count_min: Optional[int] = Field(None, description="Absolute minimum number of replicas.", ge=0)
    cpu_count_max: Optional[int] = Field(None, description="Absolute maximum number of replicas.", ge=1)
    cpu_percent_min: Optional[float] = Field(
        None, description="Minimum number of replicas as a percentage of total cores.", ge=0.0, le=1.0
    )
    cpu_percent_max: Optional[float] = Field(
        None, description="Maximum number of replicas as a percentage of total cores.", ge=0.0, le=1.0
    )

    @root_validator(skip_on_failure=True)
    def check_exclusive_min_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that count and percent are not specified for min or max at the same time.
        """
        if values.get("cpu_count_min") is not None and values.get("cpu_percent_min") is not None:
            raise ValueError("Cannot specify both 'cpu_count_min' and 'cpu_percent_min'")

        if values.get("cpu_count_max") is not None and values.get("cpu_percent_max") is not None:
            raise ValueError("Cannot specify both 'cpu_count_max' and 'cpu_percent_max'")

        return values

    class Config:
        extra = Extra.forbid


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
    actor : str
        The fully qualified import path to the actor class or function that
        implements the stage's logic.
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
    actor: str = Field(..., description="Full import path to the stage's actor class or function.")
    enabled: bool = Field(True, description="Whether the stage is enabled.")
    config: Dict[str, Any] = Field({}, description="Configuration dictionary for the stage.")
    replicas: ReplicaConfig = Field(default_factory=ReplicaConfig, description="Replica configuration.")
    runs_after: List[str] = Field(default_factory=list, description="List of stages this stage must run after.")

    class Config:
        extra = Extra.forbid


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

    class Config:
        extra = Extra.forbid


class PipelineRuntimeConfig(BaseModel):
    """
    Configuration for the pipeline's runtime behavior.

    Attributes
    ----------
    disable_dynamic_scaling : bool
        If True, disables the dynamic scaling of stage replicas.
    dynamic_memory_threshold : float
        The memory utilization threshold (0.0 to 1.0) for dynamic scaling decisions.
    """

    disable_dynamic_scaling: bool = Field(False, description="Disable dynamic scaling of stage replicas.")
    dynamic_memory_threshold: float = Field(
        0.75, ge=0.0, le=0.95, description="Memory utilization threshold for dynamic scaling."
    )
    launch_simple_broker: bool = Field(False, description="Launch a simple message broker for the pipeline.")

    class Config:
        extra = Extra.forbid


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

    class Config:
        extra = Extra.forbid
