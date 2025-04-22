# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(Devin)
# flake8: noqa

import json
import math
import os
from typing import Dict, Any

import ray

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.stage_builders import (
    add_source_stage,
    add_metadata_injector_stage,
    add_pdf_extractor_stage,
    add_image_extractor_stage,
    add_docx_extractor_stage,
    add_audio_extractor_stage,
    add_image_dedup_stage,
    add_image_filter_stage,
    add_table_extractor_stage,
    add_chart_extractor_stage,
    add_image_caption_stage,
    add_text_splitter_stage,
    add_text_embedding_stage,
    add_embedding_storage_stage,
    add_image_storage_stage,
    add_sink_stage,
    add_pptx_extractor_stage,
    add_infographic_extractor_stage,
)


def setup_ingestion_pipeline(pipeline: RayPipeline, ingest_config: Dict[str, Any]):
    ray.init(
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

    default_cpu_count = os.environ.get("NV_INGEST_MAX_UTIL", int(max(1, math.floor(len(os.sched_getaffinity(0))))))
    add_meter_stage = os.environ.get("MESSAGE_CLIENT_TYPE") != "simple"

    ########################################################################################################
    ## Insertion and Pre-processing stages
    ########################################################################################################
    source_stage = add_source_stage(pipeline, default_cpu_count)
    # TODO(Devin)
    # submitted_job_counter_stage = add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    metadata_injector_stage = add_metadata_injector_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Primitive extraction
    ########################################################################################################
    pdf_extractor_stage = add_pdf_extractor_stage(pipeline, default_cpu_count)
    image_extractor_stage = add_image_extractor_stage(pipeline, default_cpu_count)
    docx_extractor_stage = add_docx_extractor_stage(pipeline, default_cpu_count)
    pptx_extractor_stage = add_pptx_extractor_stage(pipeline, default_cpu_count)
    audio_extractor_stage = add_audio_extractor_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Post-processing
    ########################################################################################################
    image_dedup_stage = add_image_dedup_stage(pipeline, default_cpu_count)
    image_filter_stage = add_image_filter_stage(pipeline, default_cpu_count)
    table_extraction_stage = add_table_extractor_stage(pipeline, default_cpu_count)
    chart_extraction_stage = add_chart_extractor_stage(pipeline, default_cpu_count)
    infographic_extraction_stage = add_infographic_extractor_stage(pipeline, default_cpu_count)
    image_caption_stage = add_image_caption_stage(pipeline, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Transforms and data synthesis
    ########################################################################################################
    text_splitter_stage = add_text_splitter_stage(pipeline, default_cpu_count)
    embed_extractions_stage = add_text_embedding_stage(pipeline, default_cpu_count)
    ########################################################################################################
    ## Storage and output
    ########################################################################################################
    embedding_storage_stage = add_embedding_storage_stage(pipeline, default_cpu_count)
    image_storage_stage = add_image_storage_stage(pipeline, default_cpu_count)
    # vdb_task_sink_stage = add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config)
    sink_stage = add_sink_stage(pipeline, default_cpu_count)
    ########################################################################################################

    #######################################################################################################
    ## Telemetry (Note: everything after the sync stage is out of the hot path, please keep it that way) ##
    #######################################################################################################
    # TODO(Devin)
    # otel_tracer_stage = add_otel_tracer_stage(pipe, morpheus_pipeline_config, ingest_config)
    # if add_meter_stage:
    #    otel_meter_stage = add_otel_meter_stage(pipe, morpheus_pipeline_config, ingest_config)
    # else:
    #    otel_meter_stage = None
    # completed_job_counter_stage = add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    # Add edges
    ###### INTAKE STAGES ########
    pipeline.make_edge("source", "metadata_injection", queue_size=16)
    # pipeline.make_edge("job_counter", "metadata_injection", queue_size=16)
    pipeline.make_edge("metadata_injection", "pdf_extractor", queue_size=128)  # to limit memory pressure

    ###### Document Extractors ########
    pipeline.make_edge("pdf_extractor", "audio_extractor", queue_size=16)
    pipeline.make_edge("audio_extractor", "docx_extractor", queue_size=16)
    pipeline.make_edge("docx_extractor", "pptx_extractor", queue_size=16)
    pipeline.make_edge("pptx_extractor", "image_extractor", queue_size=16)
    pipeline.make_edge("image_extractor", "infographic_extractor", queue_size=16)
    pipeline.make_edge("infographic_extractor", "table_extractor", queue_size=16)

    ###### Primitive Extractors ########
    pipeline.make_edge("table_extractor", "chart_extractor", queue_size=16)
    pipeline.make_edge("chart_extractor", "image_filter", queue_size=16)

    ###### Primitive Mutators ########
    pipeline.make_edge("image_filter", "image_dedup", queue_size=16)
    pipeline.make_edge("image_dedup", "text_splitter", queue_size=16)

    ###### Primitive Transforms ########
    pipeline.make_edge("text_splitter", "text_embedding", queue_size=16)
    pipeline.make_edge("text_embedding", "image_caption", queue_size=16)
    pipeline.make_edge("image_caption", "image_storage", queue_size=16)

    ###### Primitive Storage ########
    pipeline.make_edge("image_storage", "embedding_storage", queue_size=16)
    pipeline.make_edge("embedding_storage", "sink", queue_size=16)

    pipeline.build()

    # if add_meter_stage:
    #    pipe.add_edge(sink_stage, otel_meter_stage)
    #    pipe.add_edge(otel_meter_stage, otel_tracer_stage)
    # else:
    #    pipe.add_edge(sink_stage, otel_tracer_stage)

    # pipe.add_edge(otel_tracer_stage, completed_job_counter_stage)
