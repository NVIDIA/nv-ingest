# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import mrc
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.schemas.job_counter_schema import JobCounterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.telemetry.global_stats import GlobalStats
from nv_ingest.util.tracing import traceable
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage

logger = logging.getLogger(__name__)

MODULE_NAME = "job_counter"
MODULE_NAMESPACE = "nv_ingest"

JobCounterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def _count_jobs_impl(message: IngestControlMessage, validated_config, stats) -> IngestControlMessage:
    """
    Private helper function to count jobs.

    Parameters
    ----------
    message : IngestControlMessage
        The IngestControlMessage instance to process.
    validated_config : JobCounterSchema
        The validated configuration for the job counter.
    stats : GlobalStats
        The global statistics instance.

    Returns
    -------
    IngestControlMessage
        The updated IngestControlMessage.
    """
    logger.debug(f"Performing job counter: {validated_config.name}")

    if validated_config.name == "completed_jobs":
        if message.has_metadata("cm_failed") and message.get_metadata("cm_failed"):
            stats.increment_stat("failed_jobs")
        else:
            stats.increment_stat("completed_jobs")
        return message

    stats.increment_stat(validated_config.name)
    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _job_counter(builder: mrc.Builder) -> None:
    """
    Module for counting submitted jobs and updating the global statistics.

    This module sets up a job counter that increments a specified statistic in the global
    statistics structure each time a message is processed.

    Parameters
    ----------
    builder : mrc.Builder
        The module configuration builder.

    Returns
    -------
    None
    """
    validated_config = fetch_and_validate_module_config(builder, JobCounterSchema)
    stats = GlobalStats.get_instance()

    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
        skip_processing_if_failed=False,
    )
    def count_jobs(message: IngestControlMessage) -> IngestControlMessage:
        try:
            return _count_jobs_impl(message, validated_config, stats)
        except Exception as e:
            logger.exception("count_jobs: Failed to run job counter")
            raise type(e)("count_jobs: Failed to run job counter") from e

    job_counter_node = builder.make_node(f"{validated_config.name}_counter", ops.map(count_jobs))
    # Register the input and output of the module
    builder.register_module_input("input", job_counter_node)
    builder.register_module_output("output", job_counter_node)
