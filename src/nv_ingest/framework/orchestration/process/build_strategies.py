# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic pipeline build strategies.

This module defines an abstract PipelineBuildStrategy and concrete implementations
for Ray and the lightweight Python execution frameworks. It encapsulates the
framework-specific environment preparation, pipeline construction, and lifecycle
operations (start/stop), allowing launch sites to be framework-agnostic.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
import json

import ray
from ray import LoggingConfig

from nv_ingest.pipeline.config.replica_resolver import resolve_static_replicas
from nv_ingest.pipeline.ingest_pipeline import IngestPipelineBuilder
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest_api.util.string_processing.configuration import (
    pretty_print_pipeline_config,
)

# Python framework (lightweight) imports
from nv_ingest.framework.orchestration.python.pipeline import PythonPipeline
from nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
    PythonMessageBrokerTaskSourceConfig,
)
from nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
    PythonMessageBrokerTaskSinkConfig,
)

logger = logging.getLogger(__name__)


class PipelineBuildStrategy(ABC):
    """
    Framework-agnostic strategy for building and running pipelines.

    Implementations encapsulate all framework-specific setup (e.g., Ray
    initialization), pipeline construction and lifecycle operations, so that
    callers do not need to be framework-aware.
    """

    def __init__(self, pipeline_config: PipelineConfigSchema) -> None:
        self._config = pipeline_config

    @abstractmethod
    def prepare_environment(self) -> None:
        """Perform any framework-specific environment initialization."""

    @abstractmethod
    def build(self) -> Any:
        """Build the pipeline and return a framework-specific pipeline handle."""

    @abstractmethod
    def start(self, pipeline_handle: Any) -> None:
        """Start the pipeline for processing."""

    @abstractmethod
    def stop(self, pipeline_handle: Any) -> None:
        """Stop the pipeline and perform best-effort cleanup."""


class RayPipelineBuildStrategy(PipelineBuildStrategy):
    """
    Strategy for Ray-based pipeline execution.

    Responsibilities:
    - Initialize Ray runtime
    - Resolve replicas and pretty print final config
    - Build pipeline via IngestPipelineBuilder
    - Start/stop the pipeline and manage Ray shutdown
    """

    def prepare_environment(self) -> None:
        if not ray.is_initialized():
            logging_config = _build_logging_config_from_env()

            # Clear existing handlers from root logger before Ray adds its handler
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            logger.info("Cleared existing root logger handlers to prevent Ray logging duplicates")

            ray.init(
                namespace="nv_ingest_ray",
                ignore_reinit_error=True,
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
                logging_config=logging_config,  # Ray will add its own StreamHandler
                _system_config={
                    "local_fs_capacity_threshold": 0.9,
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {
                                "directory_path": [
                                    "/tmp/ray_spill_testing_0",
                                    "/tmp/ray_spill_testing_1",
                                    "/tmp/ray_spill_testing_2",
                                    "/tmp/ray_spill_testing_3",
                                ],
                                "buffer_size": 100_000_000,
                            },
                        },
                    ),
                },
            )

    def build(self) -> Any:
        # Resolve static replicas and pretty print the final configuration
        config = resolve_static_replicas(self._config)
        try:
            pretty_output = pretty_print_pipeline_config(config, config_path=None)
            logger.info("\n" + pretty_output)
        except Exception:
            pass

        # Build via IngestPipelineBuilder
        ingest_pipeline = IngestPipelineBuilder(config)
        ingest_pipeline.build()
        return ingest_pipeline

    def start(self, pipeline_handle: Any) -> None:
        # pipeline_handle is IngestPipelineBuilder
        pipeline_handle.start()

    def stop(self, pipeline_handle: Any) -> None:
        try:
            if pipeline_handle is not None:
                pipeline_handle.stop()
        finally:
            if ray.is_initialized():
                ray.shutdown()


class PythonPipelineBuildStrategy(PipelineBuildStrategy):
    """
    Strategy for Python (lightweight) pipeline execution.

    Current implementation wires a message-broker source to a message-broker sink
    and allows for future insertion of in-process processing functions.
    """

    def prepare_environment(self) -> None:
        # No special environment initialization required for pure Python mode
        return None

    def build(self) -> Any:
        # Pretty print config (parity with Ray)
        try:
            pretty_output = pretty_print_pipeline_config(self._config, config_path=None)
            logger.info("\n" + pretty_output)
        except Exception:
            pass

        # Resolve endpoints by conventional names (initial implementation)
        source_stage_cfg = _find_stage_by_name_local(self._config, "source_stage")
        sink_stage_cfg = _find_stage_by_name_local(self._config, "broker_response")
        if source_stage_cfg is None or sink_stage_cfg is None:
            raise ValueError(
                "Python framework requires stages named 'source_stage' and 'broker_response' "
                "in this initial implementation"
            )

        # Build Python stage configs
        src_cfg = PythonMessageBrokerTaskSourceConfig(**(source_stage_cfg.config or {}))
        snk_cfg = PythonMessageBrokerTaskSinkConfig(**(sink_stage_cfg.config or {}))

        # Instantiate stages (propagate YAML-driven names)
        source = PythonMessageBrokerTaskSource(config=src_cfg, stage_name=source_stage_cfg.name)
        sink = PythonMessageBrokerTaskSink(config=snk_cfg, stage_name=sink_stage_cfg.name)

        # Placeholder for future processing functions
        processing_functions = []

        pipeline = PythonPipeline(source=source, sink=sink, processing_functions=processing_functions)
        return pipeline

    def start(self, pipeline_handle: Any) -> None:
        # pipeline_handle is PythonPipeline
        pipeline_handle.start()

    def stop(self, pipeline_handle: Any) -> None:
        try:
            if pipeline_handle is not None:
                pipeline_handle.stop()
        except Exception:
            pass


def _build_logging_config_from_env() -> LoggingConfig:
    """
    Build Ray LoggingConfig from environment variables.
    Mirrors the configuration used in execution.launch_pipeline.
    """

    # Apply package-level preset defaults first
    preset_level = os.environ.get("INGEST_RAY_LOG_LEVEL", "DEVELOPMENT").upper()

    # Define preset configurations
    presets = {
        "PRODUCTION": {
            "RAY_LOGGING_LEVEL": "ERROR",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "9",  # 10GB total
            "RAY_DISABLE_IMPORT_WARNING": "1",
            "RAY_USAGE_STATS_ENABLED": "0",
        },
        "DEVELOPMENT": {
            "RAY_LOGGING_LEVEL": "INFO",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "19",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
        "DEBUG": {
            "RAY_LOGGING_LEVEL": "DEBUG",
            "RAY_LOGGING_ENCODING": "JSON",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "name,funcName,lineno",
            "RAY_DEDUP_LOGS": "0",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "536870912",  # 512MB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "39",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
    }

    # Validate preset level
    if preset_level not in presets:
        logger.warning(
            f"Invalid INGEST_RAY_LOG_LEVEL '{preset_level}', using DEVELOPMENT. "
            f"Valid presets: {list(presets.keys())}"
        )
        preset_level = "DEVELOPMENT"

    # Apply preset defaults (only if env var not already set)
    preset_config = presets[preset_level]
    for key, default_value in preset_config.items():
        if key not in os.environ:
            os.environ[key] = default_value

    logger.info(f"Applied Ray logging preset: {preset_level}")

    # Get log level from environment, default to INFO
    log_level = os.environ.get("RAY_LOGGING_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        logger.warning(f"Invalid RAY_LOGGING_LEVEL '{log_level}', using INFO. Valid levels: {valid_levels}")
        log_level = "INFO"

    # Get encoding format from environment, default to TEXT
    encoding = os.environ.get("RAY_LOGGING_ENCODING", "TEXT").upper()

    # Validate encoding
    valid_encodings = ["TEXT", "JSON"]
    if encoding not in valid_encodings:
        logger.warning(f"Invalid RAY_LOGGING_ENCODING '{encoding}', using TEXT. Valid encodings: {valid_encodings}")
        encoding = "TEXT"

    # Get additional standard logger attributes
    additional_attrs_str = os.environ.get("RAY_LOGGING_ADDITIONAL_ATTRS", "")
    additional_log_standard_attrs = []
    if additional_attrs_str:
        additional_log_standard_attrs = [attr.strip() for attr in additional_attrs_str.split(",") if attr.strip()]

    # Set log deduplication environment variable if specified
    dedup_logs = os.environ.get("RAY_DEDUP_LOGS", "1")
    if dedup_logs is not None:
        os.environ["RAY_DEDUP_LOGS"] = str(dedup_logs)

    # Set log to driver environment variable if specified
    log_to_driver = os.environ.get("RAY_LOG_TO_DRIVER", "0")
    if log_to_driver is not None:
        os.environ["RAY_LOG_TO_DRIVER"] = str(log_to_driver)

    # Configure log rotation settings
    rotate_bytes = os.environ.get("RAY_LOGGING_ROTATE_BYTES", "1073741824")  # Default: 1GB per file
    if rotate_bytes is not None:
        try:
            rotate_bytes_int = int(rotate_bytes)
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = str(rotate_bytes_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BYTES '{rotate_bytes}', using default (1GB)")
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = "1073741824"

    rotate_backup_count = os.environ.get("RAY_LOGGING_ROTATE_BACKUP_COUNT", "19")  # Default: 19 backups (20GB Max)
    if rotate_backup_count is not None:
        try:
            backup_count_int = int(rotate_backup_count)
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = str(backup_count_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BACKUP_COUNT '{rotate_backup_count}', using default (19)")
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = "19"

    # Configure Ray internal logging verbosity
    disable_import_warning = os.environ.get("RAY_DISABLE_IMPORT_WARNING", "0")
    if disable_import_warning is not None:
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = str(disable_import_warning)

    # Configure usage stats collection
    usage_stats_enabled = os.environ.get("RAY_USAGE_STATS_ENABLED", "1")
    if usage_stats_enabled is not None:
        os.environ["RAY_USAGE_STATS_ENABLED"] = str(usage_stats_enabled)

    # Create LoggingConfig with validated parameters
    logging_config = LoggingConfig(
        encoding=encoding,
        log_level=log_level,
        additional_log_standard_attrs=additional_log_standard_attrs,
    )

    logger.info(
        f"Ray logging configured: preset={preset_level}, level={log_level}, encoding={encoding}, "
        f"additional_attrs={additional_log_standard_attrs}, "
        f"dedup_logs={os.environ.get('RAY_DEDUP_LOGS', '1')}, "
        f"log_to_driver={os.environ.get('RAY_LOG_TO_DRIVER', '0')}, "
        f"rotate_bytes={os.environ.get('RAY_LOGGING_ROTATE_BYTES', '1073741824')}, "
        f"rotate_backup_count={os.environ.get('RAY_LOGGING_ROTATE_BACKUP_COUNT', '19')}"
    )

    return logging_config


def _find_stage_by_name_local(config: PipelineConfigSchema, name: str) -> Optional[Any]:
    for st in getattr(config, "stages", []) or []:
        if getattr(st, "name", None) == name:
            return st
    return None
