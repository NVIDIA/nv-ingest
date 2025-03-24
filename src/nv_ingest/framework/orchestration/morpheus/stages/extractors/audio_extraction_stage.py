# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from morpheus.config import Config
from nv_ingest_api.internal.extract.audio.audio_extraction import extract_text_from_audio_internal
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema

from nv_ingest.framework.orchestration.morpheus.stages.meta.multiprocessing_stage import MultiProcessingBaseStage


def _inject_validated_config(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict,
    execution_trace_log: Optional[List[Any]] = None,
    validated_config: Any = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Helper function that injects the validated_config into the config dictionary and
    calls extract_text_from_audio_internal.

    Parameters
    ----------
    df_payload : pd.DataFrame
        A DataFrame containing PDF documents.
    task_config : dict
        A dictionary of configuration parameters. Expected to include 'task_props'.
    execution_trace_log : list, optional
        Optional list for trace information.
    validated_config : Any, optional
        The validated configuration to be injected.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        The result from extract_primitives_from_pdf.
    """

    return extract_text_from_audio_internal(
        df_extraction_ledger=df_extraction_ledger,
        task_config=task_config,
        extraction_config=validated_config,
        execution_trace_log=execution_trace_log,
    )


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
    _wrapped_process_fn = functools.partial(_inject_validated_config, validated_config=validated_config)

    return MultiProcessingBaseStage(
        c=c,
        pe_count=pe_count,
        task=task,
        task_desc=task_desc,
        process_fn=_wrapped_process_fn,
        document_type="regex:^(mp3|wav)$",
    )
