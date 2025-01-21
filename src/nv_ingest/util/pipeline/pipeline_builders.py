# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

from morpheus.config import Config
from morpheus.pipeline.pipeline import Pipeline

from nv_ingest.util.pipeline.stage_builders import *

logger = logging.getLogger(__name__)


def setup_ingestion_pipeline(
    pipe: Pipeline, morpheus_pipeline_config: Config, ingest_config: typing.Dict[str, typing.Any]
):
    default_cpu_count = get_default_cpu_count()
    add_meter_stage = os.environ.get("MESSAGE_CLIENT_TYPE") != "simple"

    ########################################################################################################
    ## Insertion and Pre-processing stages
    ########################################################################################################
    source_stage = add_source_stage(pipe, morpheus_pipeline_config, ingest_config)
    submitted_job_counter_stage = add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    metadata_injector_stage = add_metadata_injector_stage(pipe, morpheus_pipeline_config)
    ########################################################################################################

    ########################################################################################################
    ## Primitive extraction
    ########################################################################################################
    pdf_extractor_stage = add_pdf_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_extractor_stage = add_image_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    docx_extractor_stage = add_docx_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    pptx_extractor_stage = add_pptx_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Post-processing
    ########################################################################################################
    image_dedup_stage = add_image_dedup_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_filter_stage = add_image_filter_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    table_extraction_stage = add_table_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    chart_extraction_stage = add_chart_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_caption_stage = add_image_caption_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Transforms and data synthesis
    ########################################################################################################
    nemo_splitter_stage = add_nemo_splitter_stage(pipe, morpheus_pipeline_config, ingest_config)
    embed_extractions_stage = add_embed_extractions_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################
    ## Storage and output
    ########################################################################################################
    embedding_storage_stage = add_embedding_storage_stage(pipe, morpheus_pipeline_config)
    image_storage_stage = add_image_storage_stage(pipe, morpheus_pipeline_config)
    vdb_task_sink_stage = add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config)
    sink_stage = add_sink_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    #######################################################################################################
    ## Telemetry (Note: everything after the sync stage is out of the hot path, please keep it that way) ##
    #######################################################################################################
    otel_tracer_stage = add_otel_tracer_stage(pipe, morpheus_pipeline_config, ingest_config)
    if add_meter_stage:
        otel_meter_stage = add_otel_meter_stage(pipe, morpheus_pipeline_config, ingest_config)
    else:
        otel_meter_stage = None
    completed_job_counter_stage = add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    # Add edges
    pipe.add_edge(source_stage, submitted_job_counter_stage)
    pipe.add_edge(submitted_job_counter_stage, metadata_injector_stage)
    pipe.add_edge(metadata_injector_stage, pdf_extractor_stage)
    pipe.add_edge(pdf_extractor_stage, image_extractor_stage)
    pipe.add_edge(image_extractor_stage, docx_extractor_stage)
    pipe.add_edge(docx_extractor_stage, pptx_extractor_stage)
    pipe.add_edge(pptx_extractor_stage, image_dedup_stage)
    pipe.add_edge(image_dedup_stage, image_filter_stage)
    pipe.add_edge(image_filter_stage, table_extraction_stage)
    pipe.add_edge(table_extraction_stage, chart_extraction_stage)
    pipe.add_edge(chart_extraction_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, image_caption_stage)
    pipe.add_edge(image_caption_stage, embed_extractions_stage)
    pipe.add_edge(embed_extractions_stage, image_storage_stage)
    pipe.add_edge(image_storage_stage, embedding_storage_stage)
    pipe.add_edge(embedding_storage_stage, vdb_task_sink_stage)
    pipe.add_edge(vdb_task_sink_stage, sink_stage)
    if add_meter_stage:
        pipe.add_edge(sink_stage, otel_meter_stage)
        pipe.add_edge(otel_meter_stage, otel_tracer_stage)
    else:
        pipe.add_edge(sink_stage, otel_tracer_stage)
    pipe.add_edge(otel_tracer_stage, completed_job_counter_stage)
