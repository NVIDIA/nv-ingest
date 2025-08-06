# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import os
from typing import Dict, Any

import ray
from ray import LoggingConfig
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.stage_builders import (
    add_source_stage,
    add_metadata_injector_stage,
    add_pdf_extractor_stage,
    add_image_extractor_stage,
    add_docx_extractor_stage,
    add_audio_extractor_stage,
    add_html_extractor_stage,
    add_image_dedup_stage,
    add_image_filter_stage,
    add_table_extractor_stage,
    add_chart_extractor_stage,
    add_image_caption_stage,
    add_text_splitter_stage,
    add_text_embedding_stage,
    add_embedding_storage_stage,
    add_image_storage_stage,
    add_message_broker_response_stage,
    add_pptx_extractor_stage,
    add_infographic_extractor_stage,
    add_otel_tracer_stage,
    add_default_drain_stage,
)
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger("uvicorn")


def export_config_to_env(ingest_config: Any) -> None:
    if isinstance(ingest_config, BaseModel):
        ingest_config = ingest_config.model_dump()

    os.environ.update({key.upper(): val for key, val in ingest_config.items()})


def build_logging_config_from_env() -> LoggingConfig:
    """
    Build Ray LoggingConfig from environment variables.

    Package-level preset (sets all defaults):
    - INGEST_RAY_LOG_LEVEL: PRODUCTION, DEVELOPMENT, DEBUG. Default: DEVELOPMENT

    Individual environment variables (override preset defaults):
    - RAY_LOGGING_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
    - RAY_LOGGING_ENCODING: Log encoding format (TEXT, JSON). Default: TEXT
    - RAY_LOGGING_ADDITIONAL_ATTRS: Comma-separated list of additional standard logger attributes
    - RAY_DEDUP_LOGS: Enable/disable log deduplication (0/1). Default: 1 (enabled)
    - RAY_LOG_TO_DRIVER: Enable/disable logging to driver (true/false). Default: true
    - RAY_LOGGING_ROTATE_BYTES: Maximum log file size before rotation (bytes). Default: 1GB
    - RAY_LOGGING_ROTATE_BACKUP_COUNT: Number of backup log files to keep. Default: 19
    - RAY_DISABLE_IMPORT_WARNING: Disable Ray import warnings (0/1). Default: 0
    - RAY_USAGE_STATS_ENABLED: Enable/disable usage stats collection (0/1). Default: 1
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
            "RAY_LOG_TO_DRIVER": "1",  # true
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
            "RAY_LOG_TO_DRIVER": "1",  # true
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
    log_to_driver = os.environ.get("RAY_LOG_TO_DRIVER", "1")
    if log_to_driver is not None:
        os.environ["RAY_LOG_TO_DRIVER"] = str(log_to_driver).lower()

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
        f"log_to_driver={os.environ.get('RAY_LOG_TO_DRIVER', 'true')}, "
        f"rotate_bytes={os.environ.get('RAY_LOGGING_ROTATE_BYTES', '1073741824')}, "
        f"rotate_backup_count={os.environ.get('RAY_LOGGING_ROTATE_BACKUP_COUNT', '19')}"
    )

    return logging_config


def setup_ingestion_pipeline(pipeline: RayPipeline, ingest_config: Dict[str, Any] = None):
    # Initialize the pipeline with the configuration
    if ingest_config:
        # Export the config to environment variables
        export_config_to_env(ingest_config)

    _ = logging.getLogger().getEffectiveLevel()
    logging_config = build_logging_config_from_env()
    ray_context = ray.init(
        namespace="nv_ingest_ray",
        logging_config=logging_config,
        ignore_reinit_error=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
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
    system_resource_probe = SystemResourceProbe()

    effective_cpu_core_count = system_resource_probe.get_effective_cores()
    default_cpu_count = int(os.environ.get("NV_INGEST_MAX_UTIL", int(max(1, math.floor(effective_cpu_core_count)))))

    add_meter_stage = os.environ.get("MESSAGE_CLIENT_TYPE") != "simple"
    _ = add_meter_stage  # TODO(Devin)

    ########################################################################################################
    ## Insertion and Pre-processing stages
    ########################################################################################################
    logger.debug("Setting up ingestion pipeline")
    source_stage_id = add_source_stage(pipeline, default_cpu_count)
    # TODO(Devin): Job counter used a global stats object that isn't ray compatible, need to update.
    # submitted_job_counter_stage = add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    metadata_injector_stage_id = add_metadata_injector_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Primitive extraction
    ########################################################################################################
    pdf_extractor_stage_id = add_pdf_extractor_stage(pipeline, default_cpu_count)
    image_extractor_stage_id = add_image_extractor_stage(pipeline, default_cpu_count)
    docx_extractor_stage_id = add_docx_extractor_stage(pipeline, default_cpu_count)
    pptx_extractor_stage_id = add_pptx_extractor_stage(pipeline, default_cpu_count)
    audio_extractor_stage_id = add_audio_extractor_stage(pipeline, default_cpu_count)
    html_extractor_stage_id = add_html_extractor_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Post-processing
    ########################################################################################################
    image_dedup_stage_id = add_image_dedup_stage(pipeline, default_cpu_count)
    image_filter_stage_id = add_image_filter_stage(pipeline, default_cpu_count)
    table_extraction_stage_id = add_table_extractor_stage(pipeline, default_cpu_count)
    chart_extraction_stage_id = add_chart_extractor_stage(pipeline, default_cpu_count)
    infographic_extraction_stage_id = add_infographic_extractor_stage(pipeline, default_cpu_count)
    image_caption_stage_id = add_image_caption_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Transforms and data synthesis
    ########################################################################################################
    text_splitter_stage_id = add_text_splitter_stage(pipeline, default_cpu_count)
    embed_extractions_stage_id = add_text_embedding_stage(pipeline, default_cpu_count)

    ########################################################################################################
    ## Storage and output
    ########################################################################################################
    embedding_storage_stage_id = add_embedding_storage_stage(pipeline, default_cpu_count)
    image_storage_stage_id = add_image_storage_stage(pipeline, default_cpu_count)
    # vdb_task_sink_stage = add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config)
    broker_response_stage_id = add_message_broker_response_stage(pipeline, default_cpu_count)
    ########################################################################################################

    #######################################################################################################
    ## Telemetry (Note: everything after the sync stage is out of the hot path, please keep it that way) ##
    #######################################################################################################
    otel_tracer_stage_id = add_otel_tracer_stage(pipeline, default_cpu_count)

    # TODO(devin)
    # if add_meter_stage:
    #    otel_meter_stage = add_otel_meter_stage(pipe, morpheus_pipeline_config, ingest_config)
    # else:
    #    otel_meter_stage = None
    # completed_job_counter_stage = add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    # Add a drain stage to the pipeline -- flushes and deletes control messages
    drain_id = add_default_drain_stage(pipeline, default_cpu_count)

    ingest_edge_buffer_size = int(os.environ.get("INGEST_EDGE_BUFFER_SIZE", 32))

    # Add edges
    ###### Intake Stages ########
    pipeline.make_edge(source_stage_id, metadata_injector_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(metadata_injector_stage_id, pdf_extractor_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Document Extractors ########
    pipeline.make_edge(pdf_extractor_stage_id, audio_extractor_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(audio_extractor_stage_id, docx_extractor_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(docx_extractor_stage_id, pptx_extractor_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(pptx_extractor_stage_id, image_extractor_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(image_extractor_stage_id, html_extractor_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(html_extractor_stage_id, infographic_extraction_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Primitive Extractors ########
    pipeline.make_edge(infographic_extraction_stage_id, table_extraction_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(table_extraction_stage_id, chart_extraction_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(chart_extraction_stage_id, image_filter_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Primitive Mutators ########
    pipeline.make_edge(image_filter_stage_id, image_dedup_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(image_dedup_stage_id, text_splitter_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Primitive Transforms ########
    pipeline.make_edge(text_splitter_stage_id, image_caption_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(image_caption_stage_id, embed_extractions_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(embed_extractions_stage_id, image_storage_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Primitive Storage ########
    pipeline.make_edge(image_storage_stage_id, embedding_storage_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(embedding_storage_stage_id, broker_response_stage_id, queue_size=ingest_edge_buffer_size)

    ###### Response and Telemetry ########
    pipeline.make_edge(broker_response_stage_id, otel_tracer_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(otel_tracer_stage_id, drain_id, queue_size=ingest_edge_buffer_size)

    pipeline.build()

    # TODO(devin)
    # if add_meter_stage:
    #    pipe.add_edge(sink_stage, otel_meter_stage)
    #    pipe.add_edge(otel_meter_stage, otel_tracer_stage)
    # else:
    #    pipe.add_edge(sink_stage, otel_tracer_stage)

    # pipe.add_edge(otel_tracer_stage, completed_job_counter_stage)

    return ray_context
