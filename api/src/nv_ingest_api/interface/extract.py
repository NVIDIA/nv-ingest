# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Tuple, Optional, Dict, Any

import pandas as pd
from pandas import DataFrame

from . import extraction_interface_relay_constructor

from nv_ingest_api.internal.extract.pdf.pdf_extractor import extract_primitives_from_pdf_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler
from nv_ingest_api.internal.extract.docx.docx_extractor import extract_primitives_from_docx_internal
from nv_ingest_api.internal.extract.pptx.pptx_extractor import extract_primitives_from_pptx_internal
from nv_ingest_api.internal.extract.image.chart_extractor import extract_chart_data_from_image_internal
from nv_ingest_api.internal.extract.image.image_extractor import extract_primitives_from_image_internal
from nv_ingest_api.internal.extract.image.table_extractor import extract_table_data_from_image_internal
from ..internal.extract.audio.audio_extraction import extract_text_from_audio_internal
from ..internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from ..internal.schemas.extract.extract_audio_schema import AudioConfigSchema
from ..internal.schemas.extract.extract_chart_schema import ChartExtractorConfigSchema
from ..internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from ..internal.schemas.extract.extract_infographic_schema import (
    InfographicExtractorConfigSchema,
    InfographicExtractorSchema,
)
from ..internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from ..internal.schemas.extract.extract_table_schema import TableExtractorSchema
from ..internal.schemas.meta.ingest_job_schema import (
    IngestTaskChartExtraction,
    IngestTaskTableExtraction,
)

logger = logging.getLogger(__name__)


# TODO(Devin) - Alternate impl that directly takes data type and returns the dataframe


@unified_exception_handler
@extraction_interface_relay_constructor(
    api_fn=extract_primitives_from_pdf_internal,
    task_keys=["extract_text", "extract_images", "extract_tables", "extract_charts", "extract_infographics"],
)
def extract_primitives_from_pdf(
    *,
    df_extraction_ledger,  # Ledger (e.g., a pandas DataFrame)
    extract_method: str = "pdfium",  # Determines which extraction schema to use
    extract_text: bool = True,
    extract_images: bool = True,
    extract_infographics: bool = True,
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
    # UnstructuredIO parameters:
    unstructured_io_api_key: Optional[str] = None,
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
def extract_primitives_from_audio(
    *,
    df_ledger: pd.DataFrame,
    audio_endpoints: Tuple[str, str],
    audio_infer_protocol: str = "grpc",
    auth_token: str = "",
    use_ssl: bool = False,
    ssl_cert: str = "",
) -> Any:
    """
    Extract audio primitives from a ledger DataFrame using the specified audio configuration.

    This function builds an extraction configuration based on the provided audio endpoints,
    inference protocol, authentication token, and SSL settings. It then delegates the extraction
    work to the internal function ``extract_text_from_audio_internal`` using the constructed
    configuration and ledger DataFrame.

    Parameters
    ----------
    df_ledger : pandas.DataFrame
        A DataFrame containing the ledger information required for audio extraction.
    audio_endpoints : Tuple[str, str]
        A tuple of two strings representing the audio service endpoints gRPC and HTTP services.
    audio_infer_protocol : str, optional
        The protocol to use for audio inference (e.g., "grpc"). Default is "grpc".
    auth_token : str, optional
        Authentication token for the audio inference service. Default is an empty string.
    use_ssl : bool, optional
        Flag indicating whether to use SSL for secure connections. Default is False.
    ssl_cert : str, optional
        Path to the SSL certificate file to use if ``use_ssl`` is True. Default is an empty string.

    Returns
    -------
    Any
        The result of the audio extraction as returned by
        ``extract_text_from_audio_internal``. The specific type depends on the internal implementation.

    Raises
    ------
    Exception
        Any exceptions raised during the extraction process will be handled by the
        ``@unified_exception_handler`` decorator.

    Examples
    --------
    >>> import pandas as pd
    >>> # Create a sample DataFrame with ledger data
    >>> df = pd.DataFrame({"audio_data": ["file1.wav", "file2.wav"]})
    >>> result = extract_primitives_from_audio(
    ...     df_ledger=df,
    ...     audio_endpoints=("http://primary.endpoint", "http://secondary.endpoint"),
    ...     audio_infer_protocol="grpc",
    ...     auth_token="secret-token",
    ...     use_ssl=True,
    ...     ssl_cert="/path/to/cert.pem"
    ... )
    """
    task_config: Dict[str, Any] = {"params": {"extract_audio_params": {}}}

    extraction_config = AudioConfigSchema(
        **{
            "audio_endpoints": audio_endpoints,
            "audio_infer_protocol": audio_infer_protocol,
            "auth_token": auth_token,
            "use_ssl": use_ssl,
            "ssl_cert": ssl_cert,
        }
    )

    result, _ = extract_text_from_audio_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def extract_primitives_from_pptx(
    *,
    df_ledger: pd.DataFrame,
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    yolox_endpoints: Optional[Tuple[str, str]] = None,
    yolox_infer_protocol: str = "grpc",
    auth_token: str = "",
) -> pd.DataFrame:
    """
    Extract primitives from PPTX files provided in a DataFrame.

    This function configures the PPTX extraction task by assembling a task configuration
    dictionary using the provided parameters. It then creates an extraction configuration
    object (e.g., an instance of PPTXExtractorSchema) and delegates the actual extraction
    process to the internal function `extract_primitives_from_pptx_internal`.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        A DataFrame containing base64-encoded PPTX files. The DataFrame is expected to include
        columns such as "content" (with the base64-encoded PPTX) and "source_id".
    extract_text : bool, default=True
        Flag indicating whether text should be extracted from the PPTX files.
    extract_images : bool, default=True
        Flag indicating whether images should be extracted.
    extract_tables : bool, default=True
        Flag indicating whether tables should be extracted.
    extract_charts : bool, default=True
        Flag indicating whether charts should be extracted.
    extract_infographics : bool, default=True
        Flag indicating whether infographics should be extracted.
    yolox_endpoints : Optional[Tuple[str, str]], default=None
        Optional tuple containing endpoints for YOLOX inference, if needed for image analysis.
    yolox_infer_protocol : str, default="grpc"
        The protocol to use for YOLOX inference.
    auth_token : str, default=""
        Authentication token to be used with the PPTX extraction configuration.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted primitives from the PPTX files. Expected columns include
        "document_type", "metadata", and "uuid".

    Notes
    -----
    This function is decorated with `@unified_exception_handler` to handle exceptions uniformly.
    The task configuration is assembled with two main keys:
      - "params": Contains boolean flags for controlling which primitives to extract.
      - "pptx_extraction_config": Contains additional settings for PPTX extraction (e.g., YOLOX endpoints,
        inference protocol, and auth token).
    It then calls `extract_primitives_from_pptx_internal` with the DataFrame, the task configuration,
    and the extraction configuration.
    """
    task_config: Dict[str, Any] = {
        "params": {
            "extract_text": extract_text,
            "extract_images": extract_images,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
        },
        "pptx_extraction_config": {
            "yolox_endpoints": yolox_endpoints,
            "yolox_infer_protocol": yolox_infer_protocol,
            "auth_token": auth_token,
        },
    }

    extraction_config = PPTXExtractorSchema()  # Assuming PPTXExtractorSchema is defined and imported

    return extract_primitives_from_pptx_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )


@unified_exception_handler
def extract_primitives_from_docx(
    *,
    df_ledger: pd.DataFrame,
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    yolox_endpoints: Optional[Tuple[str, str]] = None,
    yolox_infer_protocol: str = "grpc",
    auth_token: str = "",
) -> pd.DataFrame:
    """
    Extract primitives from DOCX documents in a DataFrame.

    This function configures and invokes the DOCX extraction process. It builds a task configuration
    using the provided extraction flags (for text, images, tables, charts, and infographics) and additional
    settings for YOLOX endpoints, inference protocol, and authentication. It then creates a DOCX extraction
    configuration (an instance of DocxExtractorSchema) and delegates the extraction to an internal function.

    Parameters
    ----------
    df_ledger : pd.DataFrame
        The input DataFrame containing DOCX documents in base64 encoding. The DataFrame is expected to
        include required columns such as "content" (with the base64-encoded DOCX) and optionally "source_id".
    extract_text : bool, optional
        Flag indicating whether to extract text content from the DOCX documents (default is True).
    extract_images : bool, optional
        Flag indicating whether to extract images from the DOCX documents (default is True).
    extract_tables : bool, optional
        Flag indicating whether to extract tables from the DOCX documents (default is True).
    extract_charts : bool, optional
        Flag indicating whether to extract charts from the DOCX documents (default is True).
    extract_infographics : bool, optional
        Flag indicating whether to extract infographics from the DOCX documents (default is True).
    yolox_endpoints : Optional[Tuple[str, str]], optional
        A tuple containing YOLOX inference endpoints. If None, the default endpoints defined in the
        DOCX extraction configuration will be used.
    yolox_infer_protocol : str, optional
        The inference protocol to use with the YOLOX endpoints (default is "grpc").
    auth_token : str, optional
        The authentication token for accessing the YOLOX inference service (default is an empty string).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted DOCX primitives. Typically, the resulting DataFrame contains
        columns such as "document_type", "metadata", and "uuid".

    Raises
    ------
    Exception
        If an error occurs during the DOCX extraction process, the exception is logged and re-raised.
    """
    # Build the task configuration with parameters and DOCX-specific extraction settings.
    task_config: Dict[str, Any] = {
        "params": {
            "extract_text": extract_text,
            "extract_images": extract_images,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
        },
        "docx_extraction_config": {
            "yolox_endpoints": yolox_endpoints,
            "yolox_infer_protocol": yolox_infer_protocol,
            "auth_token": auth_token,
        },
    }

    # Create the extraction configuration object (instance of DocxExtractorSchema).
    extraction_config = DocxExtractorSchema()

    # Delegate the actual extraction to the internal function.
    return extract_primitives_from_docx_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )


@unified_exception_handler
def extract_primitives_from_image(
    *,
    df_ledger: pd.DataFrame,
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    yolox_endpoints: Optional[Tuple[str, str]] = None,
    yolox_infer_protocol: str = "grpc",
    auth_token: str = "",
) -> pd.DataFrame:
    task_config: Dict[str, Any] = {
        "params": {
            "extract_text": extract_text,
            "extract_images": extract_images,
            "extract_tables": extract_tables,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
        },
        "image_extraction_config": {
            "yolox_endpoints": yolox_endpoints,
            "yolox_infer_protocol": yolox_infer_protocol,
            "auth_token": auth_token,
        },
    }

    result, _ = extract_primitives_from_image_internal(
        df_extraction_ledger=df_ledger, task_config=task_config, extraction_config=None, execution_trace_log=None
    )

    return result


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
        "endpoint_config": {
            "yolox_endpoints": yolox_endpoints,
            "paddle_endpoints": paddle_endpoints,
            "yolox_infer_protocol": yolox_protocol,
            "paddle_infer_protocol": paddle_protocol,
            "auth_token": auth_token,
        }
    }
    # Remove keys with None values so that ChartExtractorConfigSchema's defaults are used.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    extraction_config = TableExtractorSchema(**config_kwargs)

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
    df_extraction_ledger : pd.DataFrame
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

    task_config = {}

    extractor_config_kwargs = {
        "endpoint_config": InfographicExtractorConfigSchema(
            **{
                "paddle_endpoints": paddle_endpoints,
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": auth_token,
            }
        )
    }
    # Remove keys with None values so that InfographicExtractorConfigSchema's defaults are used.
    extractor_config_kwargs = {k: v for k, v in extractor_config_kwargs.items() if v is not None}

    extraction_config = InfographicExtractorSchema(**extractor_config_kwargs)

    result, _ = extract_infographic_data_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result
