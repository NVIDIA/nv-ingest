# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field, Extra, root_validator


class PipelinePhase(int, Enum):
    """
    The logical phase of a pipeline stage.
    """

    PRE_PROCESSING = 0
    EXTRACTION = 1
    MUTATION = 2
    TRANSFORM = 3
    RESPONSE = 4


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
    type : Literal["source", "stage", "sink"]
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
    type: Literal["source", "stage", "sink"] = Field("stage", description="Type of the stage.")
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


class PipelineConfig(BaseModel):
    """
    Root configuration model for an ingestion pipeline.

    This model represents the entire declarative configuration for an ingestion
    pipeline, including all stages and the edges that connect them.

    Attributes
    ----------
    stages : List[StageConfig]
        A list of all stage configurations in the pipeline.
    edges : List[EdgeConfig]
        A list of all edge configurations that define the pipeline's topology.
    """

    stages: List[StageConfig] = Field(..., description="List of all stages in the pipeline.")
    edges: List[EdgeConfig] = Field(..., description="List of all edges connecting the stages.")

    class Config:
        extra = Extra.forbid
