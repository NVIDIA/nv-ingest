# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from nv_ingest.pipeline.config_loaders import load_pipeline_config
from nv_ingest.pipeline.ingest_pipeline import IngestPipeline
from nv_ingest.pipeline.pipeline_schema import PipelineConfig

# Correctly import the utility from the top-level tests directory
from tests.utilities_for_test import get_git_root


def test_load_and_build_default_pipeline():
    """Tests that the default pipeline YAML can be loaded and built successfully."""
    # Get the project root and build the absolute path to the YAML file
    project_root = get_git_root(__file__)
    assert project_root is not None, "Could not find the git repository root."
    config_path = os.path.join(project_root, "config/default_pipeline.yaml")

    # 1. Load the pipeline configuration from the YAML file
    pipeline_config = load_pipeline_config(config_path)

    # 2. Validate that the loaded configuration is a valid PipelineConfig object
    assert isinstance(pipeline_config, PipelineConfig)

    # 3. Attempt to build an IngestPipeline from the loaded configuration
    ingest_pipeline = IngestPipeline(pipeline_config)
    ingest_pipeline.build()

    # 4. Verify the pipeline has been built
    assert ingest_pipeline.is_built()
    assert ingest_pipeline.pipeline is not None
