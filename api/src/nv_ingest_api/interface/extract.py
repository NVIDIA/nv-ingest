# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple, Optional, Any, List

import pandas as pd
from pandas import DataFrame

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorConfigSchema
from nv_ingest.schemas.ingest_job_schema import IngestTaskChartExtraction, IngestTaskTableExtraction
from nv_ingest.schemas.table_extractor_schema import TableExtractorConfigSchema
from . import extraction_interface_relay_constructor
from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.extract.image.chart import extract_chart_data_from_image_internal
from ..internal.extract.image.table import extract_table_data_from_image_internal
from ..util.exception_handlers.decorators import unified_exception_handler

logger = logging.getLogger(__name__)


@unified_exception_handler
@extraction_interface_relay_constructor(
    api_fn=extract_primitives_from_pdf_internal,
    task_keys=["extract_text", "extract_images", "extract_tables", "extract_charts"],
)
def extract_primitives_from_pdf(
    *,
    df_extraction_ledger,  # Ledger (e.g., a pandas DataFrame)
    extract_method: str = "pdfium",  # Determines which extraction schema to use
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    text_depth: str = "page",
    # Adobe-specific parameters:
    adobe_client_id: Optional[str] = None,
    adobe_client_secret: Optional[str] = None,
    # LLama
    llama_api_key: Optional[str] = None,
    # PDFium-specific parameters:
    yolox_auth_token: Optional[str] = None,
    yolox_endpoints: Optional[Tuple[Optional[str], Optional[str]]] = None,
    yolox_infer_protocol: str = "http",
    # Tika-specific parameter:
    tika_server_url: Optional[str] = None,
    execution_trace_log: Optional[list[Any]] = None,
):
    """
    High-level extraction function for PDF primitives.

    This function provides a simplified interface for extracting primitives (such as text,
    images, tables, and charts) from PDF documents. It dynamically assembles both common
    task-level configuration and method-specific configuration based on the specified extraction
    method. The method-specific configuration is validated using a corresponding Pydantic schema
    (registered in the global CONFIG_SCHEMAS dictionary). The assembled configuration is then
    passed to the backend extraction function `extract_primitives_from_pdf_internal` via a decorator.

    Parameters
    ----------
    df_extraction_ledger : pandas.DataFrame
        The extraction ledger containing the PDF documents to process.
        # TODO(Devin): Add more details about the expected structure of the DataFrame.
    extract_method : str, default "pdfium"
        **Required.** Specifies which extraction method to use. This value determines which Pydantic
        schema is used to validate method-specific configuration options (e.g., "pdfium" or "tika").
    extract_text : bool, default True
        Flag indicating whether to extract text from the PDFs.
    extract_images : bool, default True
        Flag indicating whether to extract images from the PDFs.
    extract_tables : bool, default True
        Flag indicating whether to extract tables from the PDFs.
    extract_charts : bool, default True
        Flag indicating whether to extract charts from the PDFs.
    yolox_auth_token : str, optional
        Authentication token required for PDFium extraction (if applicable).
    yolox_endpoints : tuple of (Optional[str], Optional[str]), optional
        A tuple specifying the endpoints for PDFium extraction. Typically, the first element is for gRPC
        and the second for HTTP. At least one endpoint must be provided.
    yolox_infer_protocol : str, default "http"
        The inference protocol to use for PDFium extraction.
    tika_server_url : str, optional
        URL for Tika extraction (if the "tika" extraction method is used).
    execution_trace_log : list, optional
        A list for recording execution trace information during extraction. If None, an empty dictionary
        is substituted.

    Returns
    -------
    tuple
        A tuple containing:
            - The original extraction ledger.
            - A task configuration dictionary containing common task parameters.
            - An extractor configuration dictionary containing method-specific configuration (nested
              under a key corresponding to the extraction method).
            - The execution trace log (an empty dictionary if None was provided).

    Raises
    ------
    ValueError
        If the required 'extract_method' parameter is missing, if an unsupported extraction method is
        specified, or if the backend API function does not conform to the expected signature.

    Notes
    -----
    This function is intended to provide a user-friendly API that abstracts the complexity of
    configuration assembly. It leverages the `extraction_interface_relay_constructor` decorator to
    dynamically build and validate configurations, passing the final composite configuration to the
    backend function `extract_primitives_from_pdf_internal`.
    """
    pass


@unified_exception_handler
def extract_primitives_from_pptx():
    pass


@unified_exception_handler
def extract_primitives_from_docx():
    pass


@unified_exception_handler
def extract_primitives_from_image():
    pass


@unified_exception_handler
def extract_structured_data_from_chart():
    pass


@unified_exception_handler
def extract_structured_data_from_table():
    pass


@unified_exception_handler
def extract_chart_data_from_image(
    *,
    df_ledger: pd.DataFrame,
    yolox_endpoints: Tuple[str, str],
    paddle_endpoints: Tuple[str, str],
    yolox_protocol: str = "grpc",
    paddle_protocol: str = "grpc",
    auth_token: str = "",
    execution_trace_log: Optional[List[Any]] = None,
) -> tuple[DataFrame, dict]:
    """
    Public interface to extract chart data from ledger DataFrame.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing metadata required for chart extraction.
    yolox_endpoints : Tuple[str, str]
        YOLOX inference server endpoints.
    paddle_endpoints : Tuple[str, str]
        PaddleOCR inference server endpoints.
    yolox_protocol : str, optional
        Protocol for YOLOX inference (default "grpc").
    paddle_protocol : str, optional
        Protocol for PaddleOCR inference (default "grpc").
    auth_token : str, optional
        Authentication token for inference services.
    execution_trace_log : list, optional
        Execution trace logs.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame after chart extraction.

    Raises
    ------
    Exception
        If an error occurs during extraction.
    """
    task_config = IngestTaskChartExtraction()
    extraction_config = ChartExtractorConfigSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": yolox_endpoints,
                "paddle_endpoints": paddle_endpoints,
                "yolox_infer_protocol": yolox_protocol,
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": auth_token,
            }
        }
    )

    return extract_chart_data_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=execution_trace_log,
    )


@unified_exception_handler
def extract_table_data_from_image(
    *,
    df_ledger: pd.DataFrame,
    yolox_endpoints: Tuple[str, str],
    paddle_endpoints: Tuple[str, str],
    yolox_protocol: str = "grpc",
    paddle_protocol: str = "grpc",
    auth_token: str = "",
    execution_trace_log: Optional[List[Any]] = None,
) -> tuple[DataFrame, dict]:
    """
    Public interface to extract chart data from ledger DataFrame.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing metadata required for chart extraction.
    yolox_endpoints : Tuple[str, str]
        YOLOX inference server endpoints.
    paddle_endpoints : Tuple[str, str]
        PaddleOCR inference server endpoints.
    yolox_protocol : str, optional
        Protocol for YOLOX inference (default "grpc").
    paddle_protocol : str, optional
        Protocol for PaddleOCR inference (default "grpc").
    auth_token : str, optional
        Authentication token for inference services.
    execution_trace_log : list, optional
        Execution trace logs.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame after chart extraction.

    Raises
    ------
    Exception
        If an error occurs during extraction.
    """

    task_config = IngestTaskTableExtraction()
    extraction_config = TableExtractorConfigSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": yolox_endpoints,
                "paddle_endpoints": paddle_endpoints,
                "yolox_infer_protocol": yolox_protocol,
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": auth_token,
            }
        }
    )

    return extract_table_data_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=execution_trace_log,
    )
