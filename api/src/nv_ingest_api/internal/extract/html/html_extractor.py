# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import uuid
from typing import Optional, Dict, Any, Union, Tuple, List

import pandas as pd

from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import MetadataSchema
from nv_ingest_api.internal.schemas.extract.extract_html_schema import HtmlExtractorSchema
from nv_ingest_api.util.schema.schema_validator import validate_schema
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler

from markitdown.converters import HtmlConverter

logger = logging.getLogger(__name__)


@unified_exception_handler
def _convert_html(row: pd.Series, execution_trace_log: Optional[List[Any]] = None):
    metadata = row.get("metadata")
    html_content = row.get("content")

    if html_content:
        html_converter = HtmlConverter()
        md_content = html_converter.convert_string(html_content=html_content).text_content
        metadata["content"] = md_content

    return [[ContentTypeEnum.TEXT, validate_schema(metadata, MetadataSchema).model_dump(), str(uuid.uuid4())]]


def extract_markdown_from_html_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Dict[str, Any],
    extraction_config: HtmlExtractorSchema,
    execution_trace_log: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Union[Dict, None]]:
    """
    Processes a pandas DataFrame containing HTML file content, extracting html as text from
    each document and converting it to markdown.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        The input DataFrame containing html files as raw text. Expected columns include
        'source_id' and 'content'.
    task_config : Union[Dict[str, Any], BaseModel]
        Configuration instructions for the document processing task. This can be provided as a
        dictionary or a Pydantic model.
    extraction_config : Any
        A configuration object for document extraction that guides the extraction process.
    execution_trace_log : Optional[Dict[str, Any]], default=None
        An optional dictionary containing trace information for debugging or logging.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the original html content converted to markdown. The resulting
        DataFrame contains the columns "document_type", "metadata", and "uuid".

    Raises
    ------
    Exception
        If an error occurs during the document extraction process, the exception is logged and
        re-raised.
    """

    # Apply the decode_and_extract function to each row in the DataFrame.
    sr_extraction = df_extraction_ledger.apply(lambda row: _convert_html(row, execution_trace_log), axis=1)

    # Explode any list results and drop missing values.
    sr_extraction = sr_extraction.explode().dropna()

    # Convert the extraction results to a DataFrame if available.
    if not sr_extraction.empty:
        extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata", "uuid"])
    else:
        extracted_df = pd.DataFrame({"document_type": [], "metadata": [], "uuid": []})

    return extracted_df, {}
