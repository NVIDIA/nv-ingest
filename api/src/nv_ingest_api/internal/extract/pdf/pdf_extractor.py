# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, NVIDIA CORPORATION.

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import logging

from nv_ingest_api.internal.extract.pdf.engines.pdf_helpers import _orchestrate_row_extraction

logger = logging.getLogger(__name__)


def extract_primitives_from_pdf_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    extractor_config: Any,
    execution_trace_log: Optional[List[Any]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Process a DataFrame of PDF documents by orchestrating extraction for each row.

    This function applies the row-level orchestration function to every row in the
    DataFrame, aggregates the results, and returns a new DataFrame with the extracted
    data along with any trace information.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        A pandas DataFrame containing PDF documents. Must include a 'content' column
        with base64-encoded PDF data.
    task_config: dict
        A dictionary of configuration parameters. Expected to include 'task_properties'
        and 'validated_config' keys.
    extractor_config: Any
        A dictionary of configuration parameters for the extraction process.
    execution_trace_log : list, optional
        A list for accumulating trace information during extraction. Defaults to None.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        A tuple where the first element is a DataFrame with the extracted data (with
        columns: document_type, metadata, uuid) and the second element is a dictionary
        containing trace information.

    Raises
    ------
    Exception
        If an error occurs during the extraction process on any row.
    """
    try:
        task_config = task_config
        extractor_config = extractor_config

        # Apply the orchestration function to each row.
        extraction_series = df_extraction_ledger.apply(
            lambda row: _orchestrate_row_extraction(row, task_config, extractor_config, execution_trace_log), axis=1
        )
        # Explode the results if the extraction returns lists.
        extraction_series = extraction_series.explode().dropna()

        # Convert the extracted results into a DataFrame.
        if not extraction_series.empty:
            extracted_df = pd.DataFrame(extraction_series.to_list(), columns=["document_type", "metadata", "uuid"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

        return extracted_df, {"execution_trace_log": execution_trace_log}
    except Exception as e:
        err_msg = f"extract_primitives_from_pdf: Error processing PDF bytes: {e}"
        logger.error(err_msg, exc_info=True)
        raise type(e)(err_msg) from e
