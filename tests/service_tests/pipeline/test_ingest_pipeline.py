# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest.pipeline.ingest_pipeline import IngestPipelineBuilder
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema, StageConfig, EdgeConfig


def test_dependency_on_non_existent_stage_raises_error():
    """Verify that a pipeline with a dependency on a non-existent stage fails validation."""
    config = PipelineConfigSchema(
        name="test_pipeline",
        description="A test pipeline",
        stages=[
            StageConfig(name="stage_a", phase=0, actor="some.module:actor", runs_after=["stage_b"]),
            StageConfig(name="stage_c", phase=0, actor="some.module:actor"),  # Dummy stage for a valid edge
        ],
        edges=[EdgeConfig(**{"from": "stage_c", "to": "stage_a"})],
    )
    pipeline = IngestPipelineBuilder(config)
    with pytest.raises(ValueError, match="'stage_b' is not a defined stage"):
        pipeline.build()


def test_direct_circular_dependency_raises_error():
    """Verify that a direct circular dependency (A -> B -> A) is detected."""
    config = PipelineConfigSchema(
        name="test_pipeline",
        description="A test pipeline",
        stages=[
            StageConfig(name="stage_a", phase=0, actor="some.module:actor", runs_after=["stage_b"]),
            StageConfig(name="stage_b", phase=0, actor="some.module:actor", runs_after=["stage_a"]),
        ],
        edges=[EdgeConfig(**{"from": "stage_a", "to": "stage_b"})],
    )
    pipeline = IngestPipelineBuilder(config)
    with pytest.raises(ValueError, match="Circular dependency detected"):
        pipeline.build()


def test_indirect_circular_dependency_raises_error():
    """Verify that an indirect circular dependency (A -> B -> C -> A) is detected."""
    config = PipelineConfigSchema(
        name="test_pipeline",
        description="A test pipeline",
        stages=[
            StageConfig(name="stage_a", phase=0, actor="some.module:actor", runs_after=["stage_b"]),
            StageConfig(name="stage_b", phase=0, actor="some.module:actor", runs_after=["stage_c"]),
            StageConfig(name="stage_c", phase=0, actor="some.module:actor", runs_after=["stage_a"]),
        ],
        edges=[EdgeConfig(**{"from": "stage_a", "to": "stage_b"})],
    )
    pipeline = IngestPipelineBuilder(config)
    with pytest.raises(ValueError, match="Circular dependency detected"):
        pipeline.build()


def test_valid_diamond_dependency_passes(monkeypatch):
    """Verify that a valid diamond-shaped dependency (A -> B, A -> C, B -> D, C -> D) passes."""
    # Mock the actor resolver to avoid ImportError for dummy actor paths
    monkeypatch.setattr(
        "nv_ingest.pipeline.ingest_pipeline.resolve_actor_class_from_path", lambda *args, **kwargs: object
    )

    config = PipelineConfigSchema(
        name="test_pipeline",
        description="A test pipeline",
        stages=[
            StageConfig(name="stage_a", phase=0, actor="some.module:actor_a"),
            StageConfig(name="stage_b", phase=1, actor="some.module:actor_b", runs_after=["stage_a"]),
            StageConfig(name="stage_c", phase=1, actor="some.module:actor_c", runs_after=["stage_a"]),
            StageConfig(name="stage_d", phase=2, actor="some.module:actor_d", runs_after=["stage_b", "stage_c"]),
        ],
        edges=[
            EdgeConfig(**{"from": "stage_a", "to": "stage_b"}),
            EdgeConfig(**{"from": "stage_a", "to": "stage_c"}),
            EdgeConfig(**{"from": "stage_b", "to": "stage_d"}),
            EdgeConfig(**{"from": "stage_c", "to": "stage_d"}),
        ],
    )
    # This should build without raising an exception
    pipeline = IngestPipelineBuilder(config)
    try:
        pipeline.build()
    except ValueError as e:
        pytest.fail(f"Valid diamond dependency test failed unexpectedly: {e}")
