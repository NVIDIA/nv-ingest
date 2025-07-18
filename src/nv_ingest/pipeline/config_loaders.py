# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import yaml
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest_api.util.string_processing.yaml import substitute_env_vars_in_yaml_content


def load_pipeline_config(config_path: str) -> PipelineConfigSchema:
    """
    Loads a pipeline configuration file, substituting environment variables.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A validated PipelineConfig object.
    """
    # 1. Read the raw YAML file content
    with open(config_path, "r") as f:
        raw_content = f.read()

    # 2. Substitute all environment variable placeholders using the utility function
    substituted_content = substitute_env_vars_in_yaml_content(raw_content)

    # 3. Parse the substituted content with PyYAML
    processed_config = yaml.safe_load(substituted_content)

    # Pydantic validates the clean, substituted data against the schema
    return PipelineConfigSchema(**processed_config)
