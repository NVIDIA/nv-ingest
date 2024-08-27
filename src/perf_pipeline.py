# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import os
import time
import typing

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage

from nv_ingest.modules.filters.image_dedup import ImageDedupLoaderFactory
from nv_ingest.modules.filters.image_filter import ImageFilterLoaderFactory
from nv_ingest.stages.pdf_extractor_stage import generate_pdf_extractor_stage
from nv_ingest.stages.pdf_memory_source_stage import PdfMemoryFileSource

logger = logging.getLogger(__name__)


@stage
def no_op_stage(message: typing.Any) -> typing.Any:
    # Return the message for the next stage
    return ["msg"]


def validate_source_config(source_info: typing.Dict[str, any]) -> None:
    """
    Validates the configuration of a source.

    This function checks whether the given source configuration dictionary
    contains all required keys: 'type', 'name', and 'config'.

    Parameters
    ----------
    source_info : typing.Dict[str, any]
        The source configuration dictionary to validate.

    Raises
    ------
    ValueError
        If any of the required keys ('type', 'name', 'config') are missing
        in the source configuration.
    """
    if "type" not in source_info or "name" not in source_info or "config" not in source_info:
        raise ValueError(f"Each source must have 'type', 'name', and 'config':\n {source_info}")


def setup_pdf_ingest_pipe(pipe: Pipeline, config: Config):
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    logger.info(f"REDIS_HOST: {redis_host}")
    logger.info(f"REDIS_PORT: {redis_port}")

    n_pe_workers = 23
    dataset_json = "/workspace/src/.tmp/new_test_output_100MB.json"
    delayed_start = True
    repeat_count = 1

    with open(dataset_json, "r") as f:
        source_config = json.load(f)

    source_stage = pipe.add_stage(PdfMemoryFileSource(config, source_config, repeat=repeat_count))
    source_monitor = pipe.add_stage(MonitorStage(config, description="Source Throughput", unit="msgs"))

    trigger_stage = pipe.add_stage(TriggerStage(config))

    extractor_stage = pipe.add_stage(
        generate_pdf_extractor_stage(config, pe_count=n_pe_workers, task="extract", task_desc="pdf_content_extractor")
    )

    extractor_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Extractor Throughput",
            unit="extractions",
            delayed_start=delayed_start,
        )
    )

    image_dedup_loader = ImageDedupLoaderFactory.get_instance(module_name="dedup_images", module_config={})

    image_dedup_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            image_dedup_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    image_dedup_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Image Dedup Throughput",
            unit="extractions",
            delayed_start=delayed_start,
        )
    )

    image_filter_loader = ImageFilterLoaderFactory.get_instance(module_name="filter_images", module_config={})

    image_filter_stage = pipe.add_stage(
        LinearModulesStage(
            config,
            image_filter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    image_filter_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Image Filter Throughput",
            unit="extractions",
            delayed_start=delayed_start,
        )
    )

    no_op = pipe.add_stage(no_op_stage(config))

    pipeline_monitor = pipe.add_stage(
        MonitorStage(
            config,
            description="Pipeline Throughput",
            unit="files",
            delayed_start=delayed_start,
        )
    )

    pipe.add_edge(source_stage, source_monitor)
    pipe.add_edge(source_monitor, trigger_stage)
    pipe.add_edge(trigger_stage, extractor_stage)
    pipe.add_edge(extractor_stage, extractor_monitor)
    pipe.add_edge(extractor_monitor, image_dedup_stage)
    pipe.add_edge(image_dedup_stage, image_dedup_monitor)
    pipe.add_edge(image_dedup_monitor, image_filter_stage)
    pipe.add_edge(image_filter_stage, image_filter_monitor)
    pipe.add_edge(image_filter_monitor, no_op)
    pipe.add_edge(no_op, pipeline_monitor)

    return source_stage


def pipeline(pipeline_config: Config) -> float:
    logging.info("Starting pipeline setup")

    pipe = Pipeline(pipeline_config)
    start_abs = time.time_ns()

    setup_pdf_ingest_pipe(pipe, pipeline_config)

    end_setup = start_run = time.time_ns()
    setup_elapsed = (end_setup - start_abs) / 1e9
    logging.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    logging.info("Running pipeline")
    pipe.run()

    end_run = time.time_ns()
    run_elapsed = (end_run - start_run) / 1e9
    total_elapsed = (end_run - start_abs) / 1e9

    logging.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logging.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from morpheus.config import CppConfig

    CppConfig.set_should_use_cpp(False)

    config = Config()
    config.pipeline_batch_size = 256
    config.enable_monitor = True
    config.feature_length = 512
    config.num_threads = os.cpu_count()
    config.model_max_batch_size = 256
    config.mode = PipelineModes.NLP

    pipeline(config)
