# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import os
from typing import Dict, Any

import ray
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


def setup_ingestion_pipeline(pipeline: RayPipeline, ingest_config: Dict[str, Any] = None):
    # Initialize the pipeline with the configuration
    if ingest_config:
        # Export the config to environment variables
        export_config_to_env(ingest_config)

    current_level = logging.getLogger().getEffectiveLevel()
    ray_context = ray.init(
        namespace="nv_ingest_ray",
        logging_level=current_level,
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
    pipeline.make_edge(text_splitter_stage_id, embed_extractions_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(embed_extractions_stage_id, image_caption_stage_id, queue_size=ingest_edge_buffer_size)
    pipeline.make_edge(image_caption_stage_id, image_storage_stage_id, queue_size=ingest_edge_buffer_size)

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
