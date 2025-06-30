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
from nv_ingest_api.internal.extract.image.infographic_extractor import extract_infographic_data_from_image_internal
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import (
    InfographicExtractorConfigSchema,
    InfographicExtractorSchema,
)
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import (
    IngestTaskChartExtraction,
    IngestTaskTableExtraction,
)
from nv_ingest_api.internal.extract.audio.audio_extraction import extract_text_from_audio_internal
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema

logger = logging.getLogger(__name__)


@unified_exception_handler
@extraction_interface_relay_constructor(
    api_fn=extract_primitives_from_pdf_internal,
    task_keys=["extract_text", "extract_images", "extract_tables", "extract_charts", "extract_infographics"],
)
def extract_primitives_from_pdf(
    *,
    df_extraction_ledger: pd.DataFrame,  # Ledger (e.g., a pandas DataFrame)
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
    # Nemoretriver Parse parameters:
    nemoretriever_parse_endpoints: Optional[Tuple[str, str]] = None,
    nemoretriever_parse_protocol: str = "http",
    nemoretriever_parse_model_name: str = None,
    # UnstructuredIO parameters:
    unstructured_io_api_key: Optional[str] = None,
    # Tika-specific parameter:
    tika_server_url: Optional[str] = None,
):
    """
    Extract text, images, tables, charts, and infographics from PDF documents.

    This function serves as a unified interface for PDF primitive extraction, supporting multiple
    extraction engines (pdfium, adobe, llama, nemoretriever_parse, unstructured_io, and tika).
    It processes a DataFrame containing base64-encoded PDF data and returns a new DataFrame
    with structured information about the extracted elements.

    The function uses a decorator pattern to dynamically validate configuration parameters
    and invoke the appropriate extraction pipeline. This design allows for flexible
    engine-specific configuration while maintaining a consistent interface.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing PDF documents to process. Must include the following columns:
        - "content" : str
            Base64-encoded PDF data
        - "source_id" : str
            Unique identifier for the document
        - "source_name" : str
            Name of the document (filename or descriptive name)
        - "document_type" : str or enum
            Document type identifier (should be "pdf" or related enum value)
        - "metadata" : Dict[str, Any]
            Dictionary containing additional metadata about the document

    extract_method : str, default "pdfium"
        The extraction engine to use. Valid options:
        - "pdfium" : PDFium-based extraction (default)
        - "adobe" : Adobe PDF Services API
        - "llama" : LlamaParse extraction
        - "nemoretriever_parse" : NVIDIA NemoRetriever Parse
        - "unstructured_io" : Unstructured.io extraction
        - "tika" : Apache Tika extraction

    extract_text : bool, default True
        Whether to extract text content from the PDFs.

    extract_images : bool, default True
        Whether to extract embedded images from the PDFs.

    extract_infographics : bool, default True
        Whether to extract infographics from the PDFs.

    extract_tables : bool, default True
        Whether to extract tables from the PDFs.

    extract_charts : bool, default True
        Whether to extract charts and graphs from the PDFs.

    text_depth : str, default "page"
        Level of text granularity to extract. Options:
        - "page" : Text extracted at page level
        - "block" : Text extracted at block level
        - "paragraph" : Text extracted at paragraph level
        - "line" : Text extracted at line level

    adobe_client_id : str, optional
        Client ID for Adobe PDF Services API. Required when extract_method="adobe".

    adobe_client_secret : str, optional
        Client secret for Adobe PDF Services API. Required when extract_method="adobe".

    llama_api_key : str, optional
        API key for LlamaParse service. Required when extract_method="llama".

    yolox_auth_token : str, optional
        Authentication token for YOLOX inference services.

    yolox_endpoints : tuple of (str, str), optional
        A tuple containing (gRPC endpoint, HTTP endpoint) for YOLOX services.
        At least one endpoint must be non-empty.

    yolox_infer_protocol : str, default "http"
        Protocol to use for YOLOX inference. Options: "http" or "grpc".

    nemoretriever_parse_endpoints : tuple of (str, str), optional
        A tuple containing (gRPC endpoint, HTTP endpoint) for NemoRetriever Parse.
        Required when extract_method="nemoretriever_parse".

    nemoretriever_parse_protocol : str, default "http"
        Protocol to use for NemoRetriever Parse. Options: "http" or "grpc".

    nemoretriever_parse_model_name : str, optional
        Model name for NemoRetriever Parse. Default is "nvidia/nemoretriever-parse".

    unstructured_io_api_key : str, optional
        API key for Unstructured.io services. Required when extract_method="unstructured_io".

    tika_server_url : str, optional
        URL for Apache Tika server. Required when extract_method="tika".

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the extracted primitives with the following columns:
        - "document_type" : Type of the extracted element (e.g., "text", "image", "table")
        - "metadata" : Dictionary containing detailed information about the extracted element
        - "uuid" : Unique identifier for the extracted element

    Raises
    ------
    ValueError
        If an unsupported extraction method is specified.
        If required parameters for the specified extraction method are missing.
        If the input DataFrame does not have the required structure.

    KeyError
        If required columns are missing from the input DataFrame.

    RuntimeError
        If extraction fails due to processing errors.

    Notes
    -----
    The function uses a decorator pattern through `extraction_interface_relay_constructor`
    which dynamically processes the parameters and validates them against the appropriate
    configuration schema. The actual extraction work is delegated to the
    `extract_primitives_from_pdf_internal` function.

    For each extraction method, specific parameters are required:
    - pdfium: yolox_endpoints
    - adobe: adobe_client_id, adobe_client_secret
    - llama: llama_api_key
    - nemoretriever_parse: nemoretriever_parse_endpoints
    - unstructured_io: unstructured_io_api_key
    - tika: tika_server_url

    Examples
    --------
    >>> import pandas as pd
    >>> import base64
    >>>
    >>> # Read a PDF file and encode it as base64
    >>> with open("document.pdf", "rb") as f:
    >>>     pdf_content = base64.b64encode(f.read()).decode("utf-8")
    >>>
    >>> # Create a DataFrame with the PDF content
    >>> df = pd.DataFrame({
    >>>     "source_id": ["doc1"],
    >>>     "source_name": ["document.pdf"],
    >>>     "content": [pdf_content],
    >>>     "document_type": ["pdf"],
    >>>     "metadata": [{"content_metadata": {"type": "document"}}]
    >>> })
    >>>
    >>> # Extract primitives using PDFium
    >>> result_df = extract_primitives_from_pdf(
    >>>     df_extraction_ledger=df,
    >>>     extract_method="pdfium",
    >>>     yolox_endpoints=(None, "http://localhost:8000/v1/infer")
    >>> )
    >>>
    >>> # Display the types of extracted elements
    >>> print(result_df["document_type"].value_counts())
    """
    pass


def extract_primitives_from_pdf_pdfium(
    df_extraction_ledger: pd.DataFrame,
    *,
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    text_depth: str = "page",
    yolox_auth_token: Optional[str] = None,
    yolox_endpoints: Optional[Tuple[Optional[str], Optional[str]]] = None,
    yolox_infer_protocol: str = "http",
) -> pd.DataFrame:
    """
    Extract primitives from PDF documents using the PDFium extraction method.

    A simplified wrapper around the general extract_primitives_from_pdf function
    that defaults to using the PDFium extraction engine.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing PDF documents to process. Must include the following columns:
        - "content" : str
            Base64-encoded PDF data
        - "source_id" : str
            Unique identifier for the document
        - "source_name" : str
            Name of the document (filename or descriptive name)
        - "document_type" : str or enum
            Document type identifier (should be "pdf" or related enum value)
        - "metadata" : Dict[str, Any]
            Dictionary containing additional metadata about the document
    extract_text : bool, default True
        Whether to extract text content
    extract_images : bool, default True
        Whether to extract embedded images
    extract_tables : bool, default True
        Whether to extract tables
    extract_charts : bool, default True
        Whether to extract charts
    extract_infographics : bool, default True
        Whether to extract infographics
    text_depth : str, default "page"
        Level of text granularity (page, block, paragraph, line)
    yolox_auth_token : str, optional
        Authentication token for YOLOX inference services
    yolox_endpoints : tuple of (str, str), optional
        Tuple containing (gRPC endpoint, HTTP endpoint) for YOLOX services
    yolox_infer_protocol : str, default "http"
        Protocol to use for YOLOX inference ("http" or "grpc")

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted primitives
    """
    return extract_primitives_from_pdf(
        df_extraction_ledger=df_extraction_ledger,
        extract_method="pdfium",
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        text_depth=text_depth,
        yolox_auth_token=yolox_auth_token,
        yolox_endpoints=yolox_endpoints,
        yolox_infer_protocol=yolox_infer_protocol,
    )


def extract_primitives_from_pdf_nemoretriever_parse(
    df_extraction_ledger: pd.DataFrame,
    *,
    extract_text: bool = True,
    extract_images: bool = True,
    extract_tables: bool = True,
    extract_charts: bool = True,
    extract_infographics: bool = True,
    text_depth: str = "page",
    yolox_auth_token: Optional[str] = None,
    yolox_endpoints: Optional[Tuple[Optional[str], Optional[str]]] = None,
    yolox_infer_protocol: str = "http",
    nemoretriever_parse_endpoints: Optional[Tuple[str, str]] = None,
    nemoretriever_parse_protocol: str = "http",
    nemoretriever_parse_model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract primitives from PDF documents using the NemoRetriever Parse extraction method.

    This function serves as a specialized wrapper around the general extract_primitives_from_pdf
    function, pre-configured to use NemoRetriever Parse as the extraction engine. It processes
    PDF documents to extract various content types including text, images, tables, charts, and
    infographics, returning the results in a structured DataFrame.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing PDF documents to process. Must include the following columns:
        - "content" : str
            Base64-encoded PDF data
        - "source_id" : str
            Unique identifier for the document
        - "source_name" : str
            Name of the document (filename or descriptive name)
        - "document_type" : str or enum
            Document type identifier (should be "pdf" or related enum value)
        - "metadata" : Dict[str, Any]
            Dictionary containing additional metadata about the document

    extract_text : bool, default True
        Whether to extract text content from the PDFs. When True, the function will
        attempt to extract and structure all textual content according to the
        granularity specified by `text_depth`.

    extract_images : bool, default True
        Whether to extract embedded images from the PDFs. When True, the function
        will identify, extract, and process images embedded within the document.

    extract_tables : bool, default True
        Whether to extract tables from the PDFs. When True, the function will
        detect tabular structures and convert them into structured data.

    extract_charts : bool, default True
        Whether to extract charts and graphs from the PDFs. When True, the function
        will detect and extract visual data representations.

    extract_infographics : bool, default True
        Whether to extract infographics from the PDFs. When True, the function will
        identify and extract complex visual information displays.

    text_depth : str, default "page"
        Level of text granularity to extract. Options:
        - "page" : Text extracted at page level (coarsest granularity)
        - "block" : Text extracted at block level (groups of paragraphs)
        - "paragraph" : Text extracted at paragraph level (semantic units)
        - "line" : Text extracted at line level (finest granularity)

    yolox_auth_token : Optional[str], default None
        Authentication token for YOLOX inference services used for image processing.
        Required if the YOLOX services need authentication.

    yolox_endpoints : Optional[Tuple[Optional[str], Optional[str]]], default None
        A tuple containing (gRPC endpoint, HTTP endpoint) for YOLOX services.
        Used for image processing capabilities within the extraction pipeline.
        Format: (grpc_endpoint, http_endpoint)
        Example: (None, "http://localhost:8000/v1/infer")

    yolox_infer_protocol : str, default "http"
        Protocol to use for YOLOX inference. Options:
        - "http" : Use HTTP protocol for YOLOX inference services
        - "grpc" : Use gRPC protocol for YOLOX inference services

    nemoretriever_parse_endpoints : Optional[Tuple[str, str]], default None
        A tuple containing (gRPC endpoint, HTTP endpoint) for NemoRetriever Parse.
        Format: (grpc_endpoint, http_endpoint)
        Example: (None, "http://localhost:8015/v1/chat/completions")
        Required for this extraction method.

    nemoretriever_parse_protocol : str, default "http"
        Protocol to use for NemoRetriever Parse. Options:
        - "http" : Use HTTP protocol for NemoRetriever Parse services
        - "grpc" : Use gRPC protocol for NemoRetriever Parse services

    nemoretriever_parse_model_name : Optional[str], default None
        Model name for NemoRetriever Parse.
        Default is typically "nvidia/nemoretriever-parse" if None is provided.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the extracted primitives with the following columns:
        - "document_type" : str
            Type of the extracted element (e.g., "text", "image", "structured")
        - "metadata" : Dict[str, Any]
            Dictionary containing detailed information about the extracted element
            including position, content, confidence scores, etc.
        - "uuid" : str
            Unique identifier for the extracted element

    Raises
    ------
    ValueError
        If `nemoretriever_parse_endpoints` is None or empty
        If the input DataFrame does not have the required structure

    KeyError
        If required columns are missing from the input DataFrame

    RuntimeError
        If extraction fails due to service unavailability or processing errors

    Examples
    --------
    >>> import pandas as pd
    >>> import base64
    >>>
    >>> # Read a PDF file and encode it as base64
    >>> with open("document.pdf", "rb") as f:
    >>>     pdf_content = base64.b64encode(f.read()).decode("utf-8")
    >>>
    >>> # Create a DataFrame with the PDF content
    >>> df = pd.DataFrame({
    >>>     "source_id": ["doc1"],
    >>>     "source_name": ["document.pdf"],
    >>>     "content": [pdf_content],
    >>>     "document_type": ["pdf"],
    >>>     "metadata": [{"content_metadata": {"type": "document"}}]
    >>> })
    >>>
    >>> # Extract primitives using NemoRetriever Parse
    >>> result_df = extract_primitives_from_pdf_nemoretriever_parse(
    >>>     df_extraction_ledger=df,
    >>>     nemoretriever_parse_endpoints=(None, "http://localhost:8015/v1/chat/completions")
    >>> )
    >>>
    >>> # Display the types of extracted elements
    >>> print(result_df["document_type"].value_counts())

    Notes
    -----
    - NemoRetriever Parse excels at extracting structured data like tables from PDFs
    - For optimal results, ensure both NemoRetriever Parse and YOLOX services are
      properly configured and accessible
    - The extraction quality may vary depending on the complexity and quality of the input PDF
    - This function wraps the more general `extract_primitives_from_pdf` function with
      pre-configured parameters for NemoRetriever Parse extraction
    """
    return extract_primitives_from_pdf(
        df_extraction_ledger=df_extraction_ledger,
        extract_method="nemoretriever_parse",
        extract_text=extract_text,
        extract_images=extract_images,
        extract_tables=extract_tables,
        extract_charts=extract_charts,
        extract_infographics=extract_infographics,
        text_depth=text_depth,
        yolox_endpoints=yolox_endpoints,
        yolox_auth_token=yolox_auth_token,
        yolox_infer_protocol=yolox_infer_protocol,
        nemoretriever_parse_endpoints=nemoretriever_parse_endpoints,
        nemoretriever_parse_protocol=nemoretriever_parse_protocol,
        nemoretriever_parse_model_name=nemoretriever_parse_model_name,
    )


@unified_exception_handler
def extract_primitives_from_audio(
    *,
    df_ledger: pd.DataFrame,
    audio_endpoints: Tuple[str, str],
    audio_infer_protocol: str = "grpc",
    auth_token: str = None,
    use_ssl: bool = False,
    ssl_cert: str = None,
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

    extraction_config = AudioExtractorSchema(
        **{
            "audio_extraction_config": {
                "audio_endpoints": audio_endpoints,
                "audio_infer_protocol": audio_infer_protocol,
                "auth_token": auth_token,
                "ssl_cert": ssl_cert,
                "use_ssl": use_ssl,
            }
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
    }

    extraction_config = PPTXExtractorSchema(
        **{
            "pptx_extraction_config": {
                "yolox_endpoints": yolox_endpoints,
                "yolox_infer_protocol": yolox_infer_protocol,
                "auth_token": auth_token,
            },
        }
    )  # Assuming PPTXExtractorSchema is defined and imported

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
    }

    # Create the extraction configuration object (instance of DocxExtractorSchema).
    extraction_config = DocxExtractorSchema(
        **{
            "docx_extraction_config": {
                "yolox_endpoints": yolox_endpoints,
                "yolox_infer_protocol": yolox_infer_protocol,
                "auth_token": auth_token,
            },
        }
    )

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
    }

    extraction_config = ImageExtractorSchema(
        **{
            "image_extraction_config": {
                "yolox_endpoints": yolox_endpoints,
                "yolox_infer_protocol": yolox_infer_protocol,
                "auth_token": auth_token,
            },
        }
    )

    result, _ = extract_primitives_from_image_internal(
        df_extraction_ledger=df_ledger,
        task_config=task_config,
        extraction_config=extraction_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def extract_chart_data_from_image(
    *,
    df_ledger: pd.DataFrame,
    yolox_endpoints: Tuple[str, str],
    ocr_endpoints: Tuple[str, str],
    yolox_protocol: str = "grpc",
    ocr_protocol: str = "grpc",
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
    ocr_endpoints : Tuple[str, str]
        PaddleOCR inference server endpoints.
    yolox_protocol : str, optional
        Protocol for YOLOX inference (default "grpc").
    ocr_protocol : str, optional
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
    extraction_config = ChartExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": yolox_endpoints,
                "ocr_endpoints": ocr_endpoints,
                "yolox_infer_protocol": yolox_protocol,
                "ocr_infer_protocol": ocr_protocol,
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
    ocr_endpoints: Optional[Tuple[str, str]] = None,
    yolox_protocol: Optional[str] = None,
    ocr_protocol: Optional[str] = None,
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
    ocr_endpoints : Optional[Tuple[str, str]], default=None
        PaddleOCR inference server endpoints. If None, the default defined in ChartExtractorConfigSchema is used.
    yolox_protocol : Optional[str], default=None
        Protocol for YOLOX inference. If None, the default defined in ChartExtractorConfigSchema is used.
    ocr_protocol : Optional[str], default=None
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
            "ocr_endpoints": ocr_endpoints,
            "yolox_infer_protocol": yolox_protocol,
            "ocr_infer_protocol": ocr_protocol,
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
    ocr_endpoints: Optional[Tuple[str, str]] = None,
    ocr_protocol: Optional[str] = None,
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
    ocr_endpoints : Optional[Tuple[str, str]], default=None
        A tuple of PaddleOCR endpoint addresses (e.g., (gRPC_endpoint, HTTP_endpoint)) used for inference.
        If None, the default endpoints from InfographicExtractorConfigSchema are used.
    ocr_protocol : Optional[str], default=None
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
                "ocr_endpoints": ocr_endpoints,
                "ocr_infer_protocol": ocr_protocol,
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
