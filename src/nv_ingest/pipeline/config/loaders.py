# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration loading and management functions for pipeline execution.

This module provides declarative functions for loading, validating, and applying
runtime overrides to pipeline configurations, replacing imperative inline logic.
"""

import logging
import yaml
from typing import Optional

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.pipeline.default_libmode_pipeline_impl import DEFAULT_LIBMODE_PIPELINE_YAML
from nv_ingest.pipeline.default_pipeline_impl import DEFAULT_PIPELINE_YAML
from nv_ingest.framework.orchestration.execution.options import PipelineRuntimeOverrides
from nv_ingest_api.util.string_processing.yaml import substitute_env_vars_in_yaml_content

logger = logging.getLogger(__name__)


def load_pipeline_config(config_path: str) -> PipelineConfigSchema:
    """
    Load a pipeline configuration file, substituting environment variables.

    Parameters
    ----------
    config_path : str
        The path to the YAML configuration file.

    Returns
    -------
    PipelineConfigSchema
        A validated PipelineConfigSchema object.

    Raises
    ------
    ValueError
        If the YAML file cannot be parsed after environment variable substitution.
    """
    logger.info(f"Loading pipeline configuration from: {config_path}")

    # Read the raw YAML file content
    with open(config_path, "r") as f:
        raw_content = f.read()

    # Substitute all environment variable placeholders using the utility function
    substituted_content = substitute_env_vars_in_yaml_content(raw_content)

    # Parse the substituted content with PyYAML, with error handling
    try:
        processed_config = yaml.safe_load(substituted_content)
    except yaml.YAMLError as e:
        error_message = (
            f"Failed to parse YAML after environment variable substitution. "
            f"Error: {e}\n\n"
            f"--- Substituted Content ---\n{substituted_content}\n---------------------------"
        )
        raise ValueError(error_message) from e

    # Pydantic validates the clean, substituted data against the schema
    return PipelineConfigSchema(**processed_config)


def load_default_pipeline_config() -> PipelineConfigSchema:
    """
    Load and validate the embedded default (non-libmode) pipeline configuration.

    Returns
    -------
    PipelineConfigSchema
        Validated default pipeline configuration.

    Raises
    ------
    ValueError
        If the default YAML cannot be parsed or validated.
    """
    logger.info("Loading embedded default pipeline configuration")

    substituted_content = substitute_env_vars_in_yaml_content(DEFAULT_PIPELINE_YAML)

    try:
        processed_config = yaml.safe_load(substituted_content)
    except yaml.YAMLError as e:
        error_message = (
            f"Failed to parse embedded default pipeline YAML after environment variable substitution. Error: {e}"
        )
        raise ValueError(error_message) from e

    return PipelineConfigSchema(**processed_config)


def load_default_libmode_config() -> PipelineConfigSchema:
    """
    Load and validate the default libmode pipeline configuration.

    This function loads the embedded default libmode pipeline YAML,
    performs environment variable substitution, and returns a validated
    configuration object.

    Returns
    -------
    PipelineConfigSchema
        Validated default libmode pipeline configuration.

    Raises
    ------
    ValueError
        If the default YAML cannot be parsed or validated.
    """
    logger.info("Loading default libmode pipeline configuration")

    # Substitute environment variables in the YAML content
    substituted_content = substitute_env_vars_in_yaml_content(DEFAULT_LIBMODE_PIPELINE_YAML)

    # Parse the substituted content with PyYAML
    try:
        processed_config = yaml.safe_load(substituted_content)
    except yaml.YAMLError as e:
        error_message = (
            f"Failed to parse default libmode pipeline YAML after environment variable substitution. " f"Error: {e}"
        )
        raise ValueError(error_message) from e

    # Create and return validated PipelineConfigSchema
    return PipelineConfigSchema(**processed_config)


def apply_runtime_overrides(config: PipelineConfigSchema, overrides: PipelineRuntimeOverrides) -> PipelineConfigSchema:
    """
    Apply runtime parameter overrides to a pipeline configuration.

    This function creates a copy of the provided configuration and applies
    any non-None override values to the pipeline runtime settings.

    Parameters
    ----------
    config : PipelineConfigSchema
        Base pipeline configuration to modify.
    overrides : PipelineRuntimeOverrides
        Runtime overrides to apply. Only non-None values are applied.

    Returns
    -------
    PipelineConfigSchema
        Modified configuration with overrides applied.
    """
    # Create a copy to avoid modifying the original
    modified_config = config.model_copy(deep=True)

    # Apply overrides if provided
    if overrides.disable_dynamic_scaling is not None:
        modified_config.pipeline.disable_dynamic_scaling = overrides.disable_dynamic_scaling
        logger.debug(f"Applied dynamic scaling override: {overrides.disable_dynamic_scaling}")

    if overrides.dynamic_memory_threshold is not None:
        modified_config.pipeline.dynamic_memory_threshold = overrides.dynamic_memory_threshold
        logger.debug(f"Applied memory threshold override: {overrides.dynamic_memory_threshold}")

    return modified_config


def validate_pipeline_config(config: Optional[PipelineConfigSchema]) -> PipelineConfigSchema:
    """
    Validate and ensure a pipeline configuration is available.

    This function ensures that a valid pipeline configuration is available,
    either from the provided config or by loading the default libmode config.

    Parameters
    ----------
    config : Optional[PipelineConfigSchema]
        Pipeline configuration to validate, or None to load default.

    Returns
    -------
    PipelineConfigSchema
        Validated pipeline configuration.

    Raises
    ------
    ValueError
        If config is None and default config cannot be loaded.
    """
    if config is None:
        return load_default_libmode_config()

    # Config is already validated by Pydantic, just return it
    return config


def resolve_pipeline_config(provided_config: Optional[PipelineConfigSchema], libmode: bool) -> PipelineConfigSchema:
    """
    Resolve the final pipeline configuration from inputs.

    This function implements the configuration resolution logic:
    - If config provided: use it
    - If libmode=True and no config: load default libmode config
    - If libmode=False and no config: raise error

    Parameters
    ----------
    provided_config : Optional[PipelineConfigSchema]
        User-provided pipeline configuration, or None.
    libmode : bool
        Whether to allow loading default libmode configuration.

    Returns
    -------
    PipelineConfigSchema
        Resolved and validated pipeline configuration.

    Raises
    ------
    ValueError
        If no config provided and libmode=False.
    """
    if provided_config is not None:
        return provided_config

    if libmode:
        return load_default_libmode_config()
    else:
        # For non-libmode, fall back to embedded default pipeline implementation
        return load_default_pipeline_config()
