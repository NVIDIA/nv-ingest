# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from nv_ingest.pipeline.config.loaders import load_pipeline_config, load_default_libmode_config
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema

from ...utilities_for_test import get_project_root


def _load_default_yaml_config() -> PipelineConfigSchema:
    project_root = get_project_root(__file__)
    assert project_root is not None, "Could not find the project root."
    config_path = os.path.join(project_root, "config/default_pipeline.yaml")
    return load_pipeline_config(config_path)


def test_load_default_yaml_config():
    """Test that the default YAML config loads successfully."""
    config = _load_default_yaml_config()
    assert isinstance(config, PipelineConfigSchema)


def test_load_default_libmode_config():
    """Test that the default libmode config loads successfully."""
    config = load_default_libmode_config()
    assert isinstance(config, PipelineConfigSchema)
