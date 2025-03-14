# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
from typing import Optional, Dict, List, Union

import pandas as pd

from nv_ingest_api.interface.utility import (
    build_dataframe_from_files,
)
from nv_ingest_api.internal.enums.common import DocumentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.transform.caption_image import transform_image_create_vlm_caption_internal
from nv_ingest_api.internal.transform.embed_text import transform_create_text_embeddings_internal
from nv_ingest_api.internal.transform.split_text import transform_text_split_and_tokenize_internal
from nv_ingest_api.util.exception_handlers.decorators import unified_exception_handler


@unified_exception_handler
def transform_text_create_embeddings(
    *,
    inputs: pd.DataFrame,
    api_key: str,
    batch_size: Optional[int] = 8192,
    embedding_model: Optional[str] = None,
    embedding_nim_endpoint: Optional[str] = None,
    encoding_format: Optional[str] = None,
    input_type: Optional[str] = None,
    truncate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Creates text embeddings using the provided configuration.
    Parameters provided as None will use the default values from EmbedExtractionsSchema.
    """
    task_config = {}

    # Build configuration parameters only if provided; defaults come from EmbedExtractionsSchema.
    config_kwargs = {
        "batch_size": batch_size,
        "embedding_model": embedding_model,
        "embedding_nim_endpoint": embedding_nim_endpoint,
        "encoding_format": encoding_format,
        "input_type": input_type,
        "truncate": truncate,
    }
    # Remove any keys with a None value.
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}
    config_kwargs["api_key"] = api_key

    transform_config = TextEmbeddingSchema(**config_kwargs)

    result, _ = transform_create_text_embeddings_internal(
        df_transform_ledger=inputs,
        task_config=task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def transform_image_create_vlm_caption(
    *,
    inputs: Union[pd.DataFrame, tuple, List[tuple]],
    api_key: Optional[str] = None,
    prompt: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract captions for image content using the VLM model API.

    This function processes image content for caption generation. It accepts input in one
    of three forms:

    1. A pandas DataFrame with the following required structure:
       - Columns:
         - ``source_name`` (str): Identifier for the source file.
         - ``source_id`` (str): Unique identifier for the file.
         - ``content`` (str): Base64-encoded string representing the file content.
         - ``document_type`` (str): A string representing the document type (e.g., DocumentTypeEnum.PNG).
         - ``metadata`` (dict): A dictionary containing at least:
             - ``content``: Same as the base64-encoded file content.
             - ``source_metadata``: Dictionary created via :func:`create_source_metadata`.
             - ``content_metadata``: Dictionary created via :func:`create_content_metadata`.
             - ``image_metadata``: For image files, initialized as an empty dict ({}); other metadata fields
               (audio_metadata, text_metadata, etc.) are typically None or empty.
             - ``raise_on_failure``: Boolean flag (typically False).

    2. A single tuple of the form ``(file_source, document_type)``.
       - ``file_source``: Either a file path (str) or a file-like object (e.g., BytesIO).
       - ``document_type``: A string representing the document type (e.g., DocumentTypeEnum.PNG).

    3. A list of such tuples.

    For non-DataFrame inputs, a DataFrame is constructed using the helper function
    :func:`build_dataframe_from_files`. When the file_source is a file-like object, its content
    is converted to a base64-encoded string using :func:`read_bytesio_as_base64`; if it is a file
    path (str), :func:`read_file_as_base64` is used.

    Parameters
    ----------
    inputs : Union[pd.DataFrame, tuple, List[tuple]]
        Input data representing image content. Accepted formats:
          - A pandas DataFrame with the required structure as described above.
          - A single tuple ``(file_source, document_type)``.
          - A list of tuples of the form ``(file_source, document_type)``.
        In the tuples, ``file_source`` is either a file path (str) or a file-like object (e.g., BytesIO),
        and ``document_type`` is a string (typically one of the DocumentTypeEnum values).

    api_key : Optional[str], default=None
        API key for authentication with the VLM endpoint. If not provided, defaults are used.

    prompt : Optional[str], default=None
        Text prompt to guide caption generation.

    endpoint_url : Optional[str], default=None
        URL of the VLM model HTTP endpoint.

    model_name : Optional[str], default=None
        Name of the model to be used for caption generation.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with generated captions inserted into the
        ``metadata.image_metadata.caption`` field for each image row.

    Raises
    ------
    ValueError
        If the input is not a DataFrame, tuple, or list of tuples, or if any tuple is not of length 2.
    Exception
        Propagates any exception encountered during processing or caption extraction.

    Examples
    --------
    >>> # Example using a DataFrame:
    >>> df = pd.DataFrame({
    ...     "source_name": ["image.png"],
    ...     "source_id": ["image.png"],
    ...     "content": ["<base64-string>"],
    ...     "document_type": ["png"],
    ...     "metadata": [{
    ...         "content": "<base64-string>",
    ...         "source_metadata": {...},
    ...         "content_metadata": {...},
    ...         "image_metadata": {},
    ...         "raise_on_failure": False,
    ...     }],
    ... })
    >>> transform_image_create_vlm_caption(inputs=df, api_key="key", prompt="Caption the image:")

    >>> # Example using a tuple:
    >>> transform_image_create_vlm_caption(inputs=("image.png", DocumentTypeEnum.PNG), api_key="key",
        prompt="Caption the image:")

    >>> # Example using a list of tuples with file paths:
    >>> transform_image_create_vlm_caption(inputs=[("image.png", DocumentTypeEnum.PNG),
        ("image2.png", DocumentTypeEnum.PNG)], api_key="key", prompt="Caption the image:")

    >>> # Example using a list of tuples with BytesIO objects:
    >>> from io import BytesIO
    >>> with open("image.png", "rb") as f:
    ...     bytes_io = BytesIO(f.read())
    >>> transform_image_create_vlm_caption(inputs=[(bytes_io, DocumentTypeEnum.PNG)],
        api_key="key", prompt="Caption the image:")
    """
    if not isinstance(inputs, pd.DataFrame):
        # Normalize a single tuple to a list.
        if isinstance(inputs, tuple):
            file_items = [inputs]
        elif isinstance(inputs, list):
            file_items = inputs
        else:
            raise ValueError(
                "df_ledger must be a DataFrame, a tuple (file_source, document_type), or a list of such tuples."
            )

        file_sources: List[Union[str, BytesIO]] = []
        source_names: List[str] = []
        source_ids: List[str] = []
        doc_types: List[str] = []

        for item in file_items:
            if not (isinstance(item, tuple) and len(item) == 2):
                raise ValueError("Each item must be a tuple of (file_source, document_type).")
            file_source, doc_type = item
            file_sources.append(file_source)
            # Use the file_source string as the identifier if available; else construct one.
            if isinstance(file_source, str):
                identifier = file_source
            else:
                identifier = f"bytesio_{doc_type}"
            source_names.append(identifier)
            source_ids.append(identifier)
            doc_types.append(doc_type)

        inputs = build_dataframe_from_files(file_sources, source_names, source_ids, doc_types)

    task_config: Dict[str, Optional[str]] = {
        "api_key": api_key,
        "prompt": prompt,
        "endpoint_url": endpoint_url,
        "model_name": model_name,
    }
    filtered_task_config: Dict[str, str] = {k: v for k, v in task_config.items() if v is not None}

    transform_config = ImageCaptionExtractionSchema(**filtered_task_config)

    result = transform_image_create_vlm_caption_internal(
        df_transform_ledger=inputs,
        task_config=filtered_task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )

    return result


@unified_exception_handler
def transform_text_split_and_tokenize(
    *,
    inputs: Union[pd.DataFrame, str, List[str]],
    tokenizer: str,
    chunk_size: int,
    chunk_overlap: int,
    split_source_types: Optional[List[str]] = None,
    hugging_face_access_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Transform and tokenize text documents by splitting them into smaller chunks.

    This function prepares the configuration parameters for text splitting and tokenization,
    and then delegates the splitting and asynchronous tokenization to an internal function.

    The function accepts input in one of two forms:

    1. A pandas DataFrame that already follows the required structure:

       Required DataFrame Structure:
           - source_name (str): Identifier for the source document.
           - source_id (str): Unique identifier for the document.
           - content (str): The document content (typically as a base64-encoded string).
           - document_type (str): For plain text, set to DocumentTypeEnum.TXT.
           - metadata (dict): Must contain:
               * content: The original text content.
               * content_metadata: A dictionary with a key "type" (e.g., "text").
               * source_metadata: A dictionary with source-specific metadata (e.g., file path, timestamps).
               * Other keys (audio_metadata, image_metadata, etc.) set to None or empty as appropriate.
               * raise_on_failure: Boolean (typically False).

    2. A plain text string or a list of plain text strings.
       In this case, the function converts each text into a BytesIO object (encoding it as UTF-8)
       and then uses the helper function `build_dataframe_from_files` to construct a DataFrame where:
           - source_name and source_id are generated as "text_0", "text_1", etc.
           - content is the base64-encoded representation of the UTF-8 encoded text.
           - document_type is set to DocumentTypeEnum.TXT.
           - metadata is constructed using helper functions (for source and content metadata),
             with content_metadata's "type" set to "text".

    Parameters
    ----------
    inputs : Union[pd.DataFrame, str, List[str]]
        Either a DataFrame following the required structure, a single plain text string,
        or a list of plain text strings.
    tokenizer : str
        Identifier or path of the tokenizer to be used (e.g., "bert-base-uncased").
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of tokens to overlap between consecutive chunks.
    split_source_types : Optional[List[str]], default=["text"]
        List of source types to filter for text splitting. If None or empty, defaults to ["text"].
    hugging_face_access_token : Optional[str], default=None
        Access token for Hugging Face authentication, if required.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the processed documents, where text content has been split into smaller chunks.
        The returned DataFrame retains the original columns and updates the "metadata" field with
        generated tokenized segments and embedding information.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during text splitting and tokenization, with additional
        context provided by the unified exception handler.

    Examples
    --------
    >>> # Using a DataFrame:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "source_name": ["doc1.txt"],
    ...     "source_id": ["doc1.txt"],
    ...     "content": ["<base64-encoded text>"],
    ...     "document_type": ["text"],
    ...     "metadata": [{
    ...         "content": "This is a document.",
    ...         "content_metadata": {"type": "text"},
    ...         "source_metadata": {"source_id": "doc1.txt", "source_name": "doc1.txt", "source_type": "txt"},
    ...         "audio_metadata": None,
    ...         "image_metadata": None,
    ...         "text_metadata": None,
    ...         "raise_on_failure": False,
    ...     }],
    ... })
    >>> transform_text_split_and_tokenize(
    ...     inputs=df,
    ...     tokenizer="bert-base-uncased",
    ...     chunk_size=512,
    ...     chunk_overlap=50
    ... )

    >>> # Using a single plain text string:
    >>> transform_text_split_and_tokenize(
    ...     inputs="This is a plain text document.",
    ...     tokenizer="bert-base-uncased",
    ...     chunk_size=512,
    ...     chunk_overlap=50
    ... )

    >>> # Using a list of plain text strings:
    >>> texts = ["Document one text.", "Document two text."]
    >>> transform_text_split_and_tokenize(
    ...     inputs=texts,
    ...     tokenizer="bert-base-uncased",
    ...     chunk_size=512,
    ...     chunk_overlap=50
    ... )
    """
    # If input is not a DataFrame, assume it is a string or list of strings and construct a DataFrame.
    if not isinstance(inputs, pd.DataFrame):
        if isinstance(inputs, str):
            texts = [inputs]
        elif isinstance(inputs, list) and all(isinstance(t, str) for t in inputs):
            texts = inputs
        else:
            raise ValueError("df_ledger must be a DataFrame, a string, or a list of strings.")
        # Convert each text string to a BytesIO object with UTF-8 encoding.
        file_sources = [BytesIO(text.encode("utf-8")) for text in texts]
        # Generate unique identifiers for source_name and source_id.
        source_names = [f"text_{i}" for i in range(len(texts))]
        source_ids = source_names.copy()
        # For plain text, document type is set to DocumentTypeEnum.TXT.
        doc_types = [DocumentTypeEnum.TXT for _ in texts]
        inputs = build_dataframe_from_files(file_sources, source_names, source_ids, doc_types)

    if not split_source_types:
        split_source_types = ["text"]

    task_config: Dict[str, any] = {
        "chunk_overlap": chunk_overlap,
        "chunk_size": chunk_size,
        "params": {
            "hf_access_token": hugging_face_access_token,
            "split_source_types": split_source_types,
        },
        "tokenizer": tokenizer,
    }

    transform_config: TextSplitterSchema = TextSplitterSchema(
        chunk_overlap=chunk_overlap,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
    )

    result = transform_text_split_and_tokenize_internal(
        df_transform_ledger=inputs,
        task_config=task_config,
        transform_config=transform_config,
        execution_trace_log=None,
    )

    return result
