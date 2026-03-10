# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.pipeline.pipeline_schema import (
    EdgeConfig,
    PipelineConfigSchema,
    PipelinePhase,
    ReplicaConfig,
    StageConfig,
)

# Tests for ReplicaConfig


def test_replica_config_valid_counts():
    """Tests that a ReplicaConfig with absolute counts is validated correctly."""
    config = ReplicaConfig(cpu_count_min=1, cpu_count_max=2)
    assert config.cpu_count_min == 1
    assert config.cpu_count_max == 2


def test_replica_config_valid_percentages():
    """Tests that a ReplicaConfig with percentage-based values is validated correctly."""
    config = ReplicaConfig(cpu_percent_min=0.1, cpu_percent_max=0.5)
    assert config.cpu_percent_min == 0.1
    assert config.cpu_percent_max == 0.5


def test_replica_config_valid_mixed():
    """Tests that a ReplicaConfig with a mix of count and percentage is validated correctly."""
    config = ReplicaConfig(cpu_count_min=1, cpu_percent_max=0.5)
    assert config.cpu_count_min == 1
    assert config.cpu_percent_max == 0.5


def test_replica_config_exclusive_min_fails():
    """Tests that validation fails if both count_min and percent_min are set."""
    with pytest.raises(ValidationError, match="Cannot specify both cpu_count_min and cpu_percent_min"):
        ReplicaConfig(cpu_count_min=1, cpu_percent_min=0.1)


def test_replica_config_exclusive_max_fails():
    """Tests that validation fails if both count_max and percent_max are set."""
    with pytest.raises(ValidationError, match="Cannot specify both cpu_count_max and cpu_percent_max"):
        ReplicaConfig(cpu_count_max=2, cpu_percent_max=0.5)


def test_replica_config_out_of_bounds_fails():
    """Tests that validation fails if values are outside their allowed range."""
    with pytest.raises(ValidationError):
        ReplicaConfig(cpu_count_min=-1)
    with pytest.raises(ValidationError):
        ReplicaConfig(cpu_count_max=0)
    with pytest.raises(ValidationError):
        ReplicaConfig(cpu_percent_min=-0.1)
    with pytest.raises(ValidationError):
        ReplicaConfig(cpu_percent_max=1.1)


def test_replica_config_extra_fields_fails():
    """Tests that validation fails if extra fields are provided."""
    with pytest.raises(ValidationError):
        ReplicaConfig(extra_field="value")


# Tests for StageConfig


def test_stage_config_minimal_valid():
    """Tests that a minimal valid StageConfig is parsed correctly with defaults."""
    config = StageConfig(
        name="test_stage",
        actor="my.actor.path",
        phase=PipelinePhase.EXTRACTION,
    )
    assert config.name == "test_stage"
    assert config.actor == "my.actor.path"
    assert config.phase == PipelinePhase.EXTRACTION
    assert config.type == "stage"
    assert config.enabled is True
    assert config.config == {}
    assert config.replicas == ReplicaConfig()
    assert config.runs_after == []


def test_stage_config_full_valid():
    """Tests that a StageConfig with all fields specified is parsed correctly."""
    config = StageConfig(
        name="full_stage",
        type="sink",
        phase=PipelinePhase.RESPONSE,
        actor="my.actor.path.sink",
        enabled=False,
        config={"key": "value"},
        replicas={"cpu_count_max": 4},
        runs_after=["other_stage"],
    )
    assert config.name == "full_stage"
    assert config.type == "sink"
    assert config.phase == PipelinePhase.RESPONSE
    assert config.actor == "my.actor.path.sink"
    assert config.enabled is False
    assert config.config == {"key": "value"}
    assert config.replicas.cpu_count_max == 4
    assert config.runs_after == ["other_stage"]


def test_stage_config_invalid_type_fails():
    """Tests that validation fails for an invalid stage type."""
    with pytest.raises(ValidationError):
        StageConfig(name="s", actor="a", phase=0, type="invalid_type")


def test_stage_config_invalid_phase_fails():
    """Tests that validation fails for an invalid phase value."""
    with pytest.raises(ValidationError):
        StageConfig(name="s", actor="a", phase=99)


def test_stage_config_missing_required_fails():
    """Tests that validation fails if required fields are missing."""
    with pytest.raises(ValidationError):  # missing name
        StageConfig(actor="a", phase=0)
    with pytest.raises(ValidationError):  # missing actor
        StageConfig(name="s", phase=0)
    with pytest.raises(ValidationError):  # missing phase
        StageConfig(name="s", actor="a")


def test_stage_config_extra_fields_fails():
    """Tests that validation fails if extra fields are provided."""
    with pytest.raises(ValidationError):
        StageConfig(name="s", actor="a", phase=0, extra_field="value")


# Tests for EdgeConfig


def test_edge_config_valid():
    """Tests that a valid EdgeConfig is parsed correctly."""
    # Since 'from' is a reserved keyword, we must parse from a dict.
    config = EdgeConfig.parse_obj({"from": "a", "to": "b", "queue_size": 50})
    assert config.from_stage == "a"
    assert config.to_stage == "b"
    assert config.queue_size == 50


def test_edge_config_valid_with_aliases():
    """Tests that a valid EdgeConfig is parsed correctly using aliases."""
    config = EdgeConfig.parse_obj({"from": "a", "to": "b"})
    assert config.from_stage == "a"
    assert config.to_stage == "b"
    assert config.queue_size == 100  # default


def test_edge_config_default_queue_size():
    """Tests that the default queue_size is applied correctly."""
    config = EdgeConfig.parse_obj({"from": "a", "to": "b"})
    assert config.queue_size == 100


def test_edge_config_invalid_queue_size_fails():
    """Tests that validation fails for an invalid queue_size."""
    with pytest.raises(ValidationError):
        EdgeConfig.parse_obj({"from": "a", "to": "b", "queue_size": 0})
    with pytest.raises(ValidationError):
        EdgeConfig.parse_obj({"from": "a", "to": "b", "queue_size": -1})


def test_edge_config_extra_fields_fails():
    """Tests that validation fails if extra fields are provided."""
    with pytest.raises(ValidationError):
        EdgeConfig.parse_obj({"from": "a", "to": "b", "extra_field": "value"})


# Tests for PipelineConfig


def test_pipeline_config_valid():
    """Tests that a valid PipelineConfig is parsed correctly."""
    config_data = {
        "name": "test_pipeline",
        "description": "A test pipeline",
        "stages": [
            {"name": "a", "actor": "actor.a", "phase": 0},
            {"name": "b", "actor": "actor.b", "phase": 1},
        ],
        "edges": [{"from": "a", "to": "b"}],
    }
    config = PipelineConfigSchema(**config_data)
    assert len(config.stages) == 2
    assert isinstance(config.stages[0], StageConfig)
    assert len(config.edges) == 1
    assert isinstance(config.edges[0], EdgeConfig)


def test_pipeline_config_missing_stages_fails():
    """Tests that validation fails if 'stages' field is missing."""
    with pytest.raises(ValidationError):
        PipelineConfigSchema(edges=[{"from": "a", "to": "b"}])


def test_pipeline_config_missing_edges_fails():
    """Tests that validation fails if 'edges' field is missing."""
    with pytest.raises(ValidationError):
        PipelineConfigSchema(stages=[{"name": "a", "actor": "actor.a", "phase": 0}])


def test_pipeline_config_empty_stages_fails():
    """Tests that validation fails if 'stages' list is empty."""
    with pytest.raises(ValidationError):
        PipelineConfigSchema(stages=[], edges=[{"from": "a", "to": "b"}])


def test_pipeline_config_empty_edges_fails():
    """Tests that validation fails if 'edges' list is empty."""
    with pytest.raises(ValidationError):
        PipelineConfigSchema(stages=[{"name": "a", "actor": "actor.a", "phase": 0}], edges=[])


def test_pipeline_config_extra_fields_fails():
    """Tests that validation fails if extra fields are provided."""
    with pytest.raises(ValidationError):
        PipelineConfigSchema(stages=[], edges=[], extra_field="value")
