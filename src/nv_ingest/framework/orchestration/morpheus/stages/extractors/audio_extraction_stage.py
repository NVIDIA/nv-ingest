# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Dict, Any

from morpheus.config import Config
from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest_api.internal.extract.audio.audio_extraction import extract_text_from_audio_internal
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema


def generate_audio_extractor_stage(
    c: Config,
    stage_config: Dict[str, Any],
    task: str = "audio_data_extract",
    task_desc: str = "audio_data_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform audio data extraction.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.

    stage_config : Dict[str, Any]
        Configuration parameters for the audio content extractor, passed as a dictionary
        validated against the `AudioExtractorSchema`.

    task : str, optional
        The task name for the stage worker function, defining the specific audio extraction process.
        Default is "audio_data_extract".

    task_desc : str, optional
        A descriptor used for latency tracing and logging during audio extraction.
        Default is "audio_data_extraction".

    pe_count : int, optional
        The number of process engines to use for audio data extraction. This value controls
        how many worker processes will run concurrently. Default is 1.

    Returns
    -------
    MultiProcessingBaseStage
        A configured Morpheus stage with an applied worker function that handles audio data extraction
        from PDF content.
    """

    validated_config = AudioExtractorSchema(**stage_config)
    _wrapped_process_fn = functools.partial(extract_text_from_audio_internal, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c,
        pe_count=pe_count,
        task=task,
        task_desc=task_desc,
        process_fn=_wrapped_process_fn,
        document_type="regex:^(wav|mp4)$",
    )
