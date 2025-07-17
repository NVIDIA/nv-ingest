# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from envyaml import EnvYAML

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema


def load_pipeline_config(config_path: str) -> PipelineConfigSchema:
    """
    Loads a pipeline configuration file, substituting environment variables.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A validated PipelineConfig object.
    """
    # EnvYAML loads the file and substitutes environment variables.
    # We set include_environment=False to prevent it from adding all environment
    # variables to the dictionary, which would cause Pydantic validation to fail.
    # We set flatten=False to preserve the nested structure of the YAML.
    raw_config: EnvYAML = EnvYAML(yaml_file=config_path, include_environment=False, flatten=False)

    # Pydantic validates the loaded data against the schema
    return PipelineConfigSchema(**raw_config)
