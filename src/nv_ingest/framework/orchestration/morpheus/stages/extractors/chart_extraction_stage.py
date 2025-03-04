# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any
from typing import Dict

from morpheus.config import Config

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema

logger = logging.getLogger(f"morpheus.{__name__}")


def generate_chart_extractor_stage(
    c: Config,
    extractor_config: Dict[str, Any],
    task: str = "chart_data_extract",
    task_desc: str = "chart_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform chart data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    extractor_config : Dict[str, Any]
        Configuration parameters for the chart content extractor, passed as a dictionary
        validated against the `ChartExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific chart extraction process.
        Default is "chart_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during chart extraction.
        Default is "chart_data_extraction".

    pe_count : int, optional
        The number of process engines to use for chart data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles chart data extraction
        from PDF content.
    """
    try:
        validated_config = ChartExtractorSchema(**extractor_config)

        _wrapped_process_fn = functools.partial(
            extract_chart_data_from_image_internal, extraction_config=validated_config
        )

        return MultiProcessingBaseStage(
            c=c,
            pe_count=pe_count,
            task=task,
            task_desc=task_desc,
            process_fn=_wrapped_process_fn,
        )

    except Exception as e:
        err_msg = f"generate_chart_extractor_stage: Error generating table extractor stage. Original error: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
