# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import ray

from nv_ingest.pipeline.config_loaders import load_pipeline_config
from nv_ingest.pipeline.ingest_pipeline import IngestPipeline
from nv_ingest.pipeline.pipeline_schema import PipelineConfig, StageType

# Correctly import the utility from the top-level tests directory
from tests.utilities_for_test import get_git_root


@pytest.fixture(scope="module")
def default_pipeline_config() -> PipelineConfig:
    """Fixture to load the default pipeline configuration from YAML."""
    project_root = get_git_root(__file__)
    assert project_root is not None, "Could not find the git repository root."
    config_path = os.path.join(project_root, "config/default_pipeline.yaml")
    config = load_pipeline_config(config_path)
    assert isinstance(config, PipelineConfig)
    return config


@pytest.fixture(scope="module")
def ray_instance():
    """Fixture to initialize and shut down a Ray instance for the tests."""
    ray.init(num_cpus=4, ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_load_and_build_default_pipeline(default_pipeline_config: PipelineConfig, ray_instance):
    """Tests that the default pipeline YAML can be loaded and built successfully."""
    # 1. Attempt to build an IngestPipeline from the loaded configuration
    ingest_pipeline = IngestPipeline(default_pipeline_config)
    ingest_pipeline.build()

    # 2. Verify the pipeline was built correctly
    enabled_stages = [s for s in default_pipeline_config.stages if s.enabled]
    built_stages_info = ingest_pipeline._pipeline.get_stages_info()
    assert len(built_stages_info) == len(enabled_stages)
    assert len(ingest_pipeline._pipeline.get_edge_queues()) == len(default_pipeline_config.edges)

    # 3. Spot-check a few key stages to ensure they are configured correctly
    # Check the source stage
    source_stage_info = next((s for s in built_stages_info if s.is_source), None)
    assert source_stage_info is not None
    assert source_stage_info.name == "source_stage"

    # Check the sink stage
    sink_stage_info = next((s for s in built_stages_info if s.is_sink), None)
    assert sink_stage_info is not None
    assert sink_stage_info.name == "drain"
