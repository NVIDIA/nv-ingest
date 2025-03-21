# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
from typing import Any
from typing import Dict

from morpheus.config import Config

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema

logger = logging.getLogger(__name__)


def generate_infographic_extractor_stage(
    c: Config,
    extraction_config: Dict[str, Any],
    task: str = "infographic_data_extract",
    task_desc: str = "infographic_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform infographic data extraction from PDF content.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    extraction_config : Dict[str, Any]
        Configuration parameters for the infographic content extractor, passed as a dictionary
        validated against the `TableExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific infographic extraction process.
        Default is "infographic_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during infographic extraction.
        Default is "infographic_data_extraction".

    pe_count : int, optional
        The number of process engines to use for infographic data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles infographic data extraction
        from PDF content.
    """

    validated_config = InfographicExtractorSchema(**extraction_config)
    _wrapped_process_fn = functools.partial(
        extract_infographic_data_from_image_internal, extraction_config=validated_config
    )

    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
