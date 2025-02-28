# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple, Optional

import pandas as pd
from pandas import DataFrame

from nv_ingest.schemas.infographic_extractor_schema import InfographicExtractorConfigSchema
from nv_ingest.schemas.table_extractor_schema import TableExtractorConfigSchema
from . import extraction_interface_relay_constructor

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorConfigSchema
from nv_ingest.schemas.ingest_job_schema import (
    IngestTaskChartExtraction,
    IngestTaskTableExtraction,
    IngestTaskInfographicExtraction,
)
from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.internal.extract.image.chart import extract_chart_data_from_image_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from ..internal.extract.image.table import extract_table_data_from_image_internal

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
) -> DataFrame:
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

    result, _ = extract_chart_data_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def extract_table_data_from_image(
    *,
    df_ledger: pd.DataFrame,
    yolox_endpoints: Optional[Tuple[str, str]] = None,
    paddle_endpoints: Optional[Tuple[str, str]] = None,
    yolox_protocol: Optional[str] = None,
    paddle_protocol: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Public interface to extract chart data from a ledger DataFrame.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing metadata required for chart extraction.
    yolox_endpoints : Optional[Tuple[str, str]], default=None
        YOLOX inference server endpoints. If None, the default defined in ChartExtractorConfigSchema is used.
    paddle_endpoints : Optional[Tuple[str, str]], default=None
        PaddleOCR inference server endpoints. If None, the default defined in ChartExtractorConfigSchema is used.
    yolox_protocol : Optional[str], default=None
        Protocol for YOLOX inference. If None, the default defined in ChartExtractorConfigSchema is used.
    paddle_protocol : Optional[str], default=None
        Protocol for PaddleOCR inference. If None, the default defined in ChartExtractorConfigSchema is used.
    auth_token : Optional[str], default=None
        Authentication token for inference services. If None, the default defined in ChartExtractorConfigSchema is used.

    Returns
    -------
    pd.DataFrame
        - The updated DataFrame after chart extraction.

    Raises
    ------
    Exception
        If an error occurs during extraction.
    """
    task_config = IngestTaskTableExtraction()

    config_kwargs = {
        "yolox_endpoints": yolox_endpoints,
        "paddle_endpoints": paddle_endpoints,
        "yolox_infer_protocol": yolox_protocol,
        "paddle_infer_protocol": paddle_protocol,
        "auth_token": auth_token,
    }
    # Remove keys with None values so that ChartExtractorConfigSchema's defaults are used.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    extraction_config = TableExtractorConfigSchema(**config_kwargs)

    result, _ = extract_table_data_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def extract_infographic_data_from_image(
    *,
    df_ledger: pd.DataFrame,
    paddle_endpoints: Optional[Tuple[str, str]] = None,
    paddle_protocol: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract infographic data from a DataFrame using the configured infographic extraction pipeline.

    This function creates a task configuration for infographic extraction, builds the extraction
    configuration from the provided PaddleOCR endpoints, protocol, and authentication token (or uses
    the default values from InfographicExtractorConfigSchema if None), and then calls the internal
    extraction function to process the DataFrame. The unified exception handler decorator ensures
    that any errors are appropriately logged and managed.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        DataFrame containing the images and associated metadata from which infographic data is to be extracted.
    paddle_endpoints : Optional[Tuple[str, str]], default=None
        A tuple of PaddleOCR endpoint addresses (e.g., (gRPC_endpoint, HTTP_endpoint)) used for inference.
        If None, the default endpoints from InfographicExtractorConfigSchema are used.
    paddle_protocol : Optional[str], default=None
        The protocol (e.g., "grpc" or "http") for PaddleOCR inference.
        If None, the default protocol from InfographicExtractorConfigSchema is used.
    auth_token : Optional[str], default=None
        The authentication token required for secure access to PaddleOCR inference services.
        If None, the default value from InfographicExtractorConfigSchema is used.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame after infographic extraction has been performed.

    Raises
    ------
    Exception
        Propagates any exception raised during the extraction process, after being handled by the
        unified exception handler.
    """
    task_config = IngestTaskInfographicExtraction()

    config_kwargs = {
        "paddle_endpoints": paddle_endpoints,
        "paddle_infer_protocol": paddle_protocol,
        "auth_token": auth_token,
    }
    # Remove keys with None values so that InfographicExtractorConfigSchema's defaults are used.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    extraction_config = InfographicExtractorConfigSchema(**config_kwargs)

    result, _ = extract_infographic_data_from_image(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result
