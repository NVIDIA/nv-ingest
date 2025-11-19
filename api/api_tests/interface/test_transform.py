# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import logging
from io import BytesIO

import pandas as pd
import pytest

from .. import get_project_root, find_root_by_pattern
from nv_ingest_api.interface.transform import (
    transform_text_create_embeddings,
    transform_text_split_and_tokenize,
    transform_image_create_vlm_caption,
)
from nv_ingest_api.internal.enums.common import DocumentTypeEnum, ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema

logger = logging.getLogger(__name__)


@pytest.mark.integration
def test_transform_text_create_embeddings_integration():
    # Build a sample ledger DataFrame with the required columns.
    df_ledger = pd.DataFrame(
        {
            "source_name": ["./data/multimodal_test.pdf"],
            "source_id": ["./data/multimodal_test.pdf"],
            "content": ["ZmFrZV9jb250ZW50"],  # dummy base64 encoded string
            "document_type": [DocumentTypeEnum.TXT],
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": "sample text content",
                    "content_metadata": {"type": ContentTypeEnum.STRUCTURED},
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/multimodal_test.pdf",
                        "source_name": "./data/multimodal_test.pdf",
                        "source_type": "pdf",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Build the default input arguments from the EmbedExtractionsSchema.
    default_schema = TextEmbeddingSchema()
    default_args = default_schema.model_dump() if hasattr(default_schema, "model_dump") else default_schema.dict()

    # Pull configuration values from the environment.
    _EMBEDDING_API_KEY = os.getenv("INGEST_EMBEDDING_API_KEY", default_args["api_key"])
    _EMBEDDING_ENDPOINT = os.getenv("INGEST_EMBEDDING_ENDPOINT", "http://127.0.0.1:8012/v1")
    _EMBEDDING_FORMAT = os.getenv("INGEST_EMBEDDING_FORMAT", default_args["encoding_format"])
    _EMBEDDING_INPUT_TYPE = os.getenv("INGEST_EMBEDDING_INPUT_TYPE", default_args["input_type"])
    _EMBEDDING_MODEL = os.getenv("INGEST_EMBEDDING_MODEL", default_args["embedding_model"])
    _EMBEDDING_TRUNCATE = os.getenv("INGEST_EMBEDDING_TRUNCATE", default_args["truncate"])
    _EMBEDDING_VECTOR_LENGTH = int(os.getenv("INGEST_EMBEDDING_VECTOR_LENGTH", 2048))

    # Explicitly map the schema values to the function's expected arguments.
    integration_args = {
        "api_key": _EMBEDDING_API_KEY,
        "embedding_model": _EMBEDDING_MODEL,
        "embedding_nim_endpoint": _EMBEDDING_ENDPOINT,
        "encoding_format": _EMBEDDING_FORMAT,
        "input_type": _EMBEDDING_INPUT_TYPE,
        "truncate": _EMBEDDING_TRUNCATE,
    }

    # Call the function under test with the constructed parameters.
    result = transform_text_create_embeddings(inputs=df_ledger, **integration_args)

    # Unpack the result, which should be a tuple: (updated DataFrame, trace_info dict)
    df_result = result

    # Assert that the returned DataFrame is not empty.
    assert not df_result.empty, "Resulting DataFrame should not be empty."

    # Check that the DataFrame has been augmented with the embeddings column.
    assert "_contains_embeddings" in df_result.columns, "Missing '_contains_embeddings' column in the result."

    # Verify that each row's metadata.embedding field is a list of the expected length.
    for idx, row in df_result.iterrows():
        metadata = row.get("metadata", {})
        assert "embedding" in metadata, f"Row {idx} missing 'embedding' in metadata."
        embedding = metadata["embedding"]
        assert isinstance(embedding, list), f"Row {idx} metadata.embedding is not a list."
        assert (
            len(embedding) == _EMBEDDING_VECTOR_LENGTH
        ), f"Row {idx} metadata.embedding length is not {_EMBEDDING_VECTOR_LENGTH}, got {len(embedding)} instead."


@pytest.mark.integration
def test_transform_text_create_embeddings_all_primitive_types_integration():
    # Build a sample ledger DataFrame with three rows—one per primitive type.
    # Note: The transform pipeline uses the document_type field to select an extractor:
    #   - TEXT rows: document_type must equal "text" and _get_pandas_text_content returns row["content"].
    #   - STRUCTURED rows: document_type must equal "structured" and _get_pandas_table_content returns
    #   row["table_metadata"]["table_content"].
    #   - IMAGE rows: document_type must equal "image" and _get_pandas_image_content returns
    #   row["image_metadata"]["caption"].
    df_ledger = pd.DataFrame(
        {
            "source_name": [
                "./data/text_file.txt",
                "./data/structured_file.pdf",
                "./data/image_file.png",
                "./data/audio_file.mp3",
            ],
            "source_id": [
                "./data/text_file.txt",
                "./data/structured_file.pdf",
                "./data/image_file.png",
                "./data/audio_file.mp3",
            ],
            "content": [
                "ZmFrZV9jb250ZW50",  # dummy base64 string for text
                "ZmFrZV9jb250ZW50",  # dummy base64 string for structured
                "ZmFrZV9jb250ZW50",  # dummy base64 string for image
                "ZmFrZV9jb250ZW50",  # dummy base64 string for audio
            ],
            # IMPORTANT: Use the document_type values expected by the extractors.
            "document_type": [
                ContentTypeEnum.TEXT.value,  # "text"
                ContentTypeEnum.STRUCTURED.value,  # "structured"
                ContentTypeEnum.IMAGE.value,  # "image"
                ContentTypeEnum.AUDIO.value,  # "audio"
            ],
            "metadata": [
                # Row 0: Text content.
                {
                    "audio_metadata": None,
                    "content": "sample text content",  # used by _get_pandas_text_content
                    "content_metadata": {"type": ContentTypeEnum.TEXT.value},
                    "error_metadata": None,
                    "image_metadata": None,
                    "table_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/text_file.txt",
                        "source_name": "./data/text_file.txt",
                        "source_type": "txt",
                    },
                    "text_metadata": None,
                },
                # Row 1: Structured content.
                {
                    "audio_metadata": None,
                    "content": "",  # not used
                    "content_metadata": {"type": ContentTypeEnum.STRUCTURED.value},
                    "error_metadata": None,
                    "image_metadata": None,
                    # For structured content, the extractor expects a non-empty "table_content".
                    "table_metadata": {"table_content": "sample table content"},
                    "source_metadata": {
                        "source_id": "./data/structured_file.pdf",
                        "source_name": "./data/structured_file.pdf",
                        "source_type": "pdf",
                    },
                    "text_metadata": None,
                },
                # Row 2: Image content.
                {
                    "audio_metadata": None,
                    "content": "",  # not used
                    "content_metadata": {"type": ContentTypeEnum.IMAGE.value},
                    "error_metadata": None,
                    # For image content, the extractor expects a non-empty caption.
                    "image_metadata": {"caption": "sample image caption"},
                    "table_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/image_file.png",
                        "source_name": "./data/image_file.png",
                        "source_type": "png",
                    },
                    "text_metadata": None,
                },
                # Row 3: Audio row.
                {
                    # For audio extraction, we supply an audio transcript.
                    "audio_metadata": {"audio_transcript": "sample audio transcript"},
                    "content": "",  # not used for audio extraction
                    "content_metadata": {"type": ContentTypeEnum.AUDIO.value},
                    "error_metadata": None,
                    "image_metadata": None,
                    "table_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/audio_file.mp3",
                        "source_name": "./data/audio_file.mp3",
                        "source_type": "mp3",
                    },
                    "text_metadata": None,
                },
            ],
        }
    )

    # Build the default input arguments from the EmbedExtractionsSchema.
    default_schema = TextEmbeddingSchema()
    default_args = default_schema.model_dump() if hasattr(default_schema, "model_dump") else default_schema.dict()

    # Pull configuration values from the environment, with fallbacks to schema defaults.
    _EMBEDDING_API_KEY = os.getenv("INGEST_EMBEDDING_API_KEY", default_args["api_key"])
    _EMBEDDING_ENDPOINT = os.getenv("INGEST_EMBEDDING_ENDPOINT", "http://127.0.0.1:8012/v1")
    _EMBEDDING_FORMAT = os.getenv("INGEST_EMBEDDING_FORMAT", default_args["encoding_format"])
    _EMBEDDING_INPUT_TYPE = os.getenv("INGEST_EMBEDDING_INPUT_TYPE", default_args["input_type"])
    _EMBEDDING_MODEL = os.getenv("INGEST_EMBEDDING_MODEL", default_args["embedding_model"])
    _EMBEDDING_TRUNCATE = os.getenv("INGEST_EMBEDDING_TRUNCATE", default_args["truncate"])
    _EMBEDDING_VECTOR_LENGTH = int(os.getenv("INGEST_EMBEDDING_VECTOR_LENGTH", 2048))

    # Map only the parameters expected by transform_text_create_embeddings.
    integration_args = {
        "api_key": _EMBEDDING_API_KEY,
        "embedding_model": _EMBEDDING_MODEL,
        "embedding_nim_endpoint": _EMBEDDING_ENDPOINT,
        "encoding_format": _EMBEDDING_FORMAT,
        "input_type": _EMBEDDING_INPUT_TYPE,
        "truncate": _EMBEDDING_TRUNCATE,
    }

    # Call the function under test.
    result = transform_text_create_embeddings(inputs=df_ledger, **integration_args)
    df_result = result

    # Assert that the returned DataFrame is not empty.
    assert not df_result.empty, "Resulting DataFrame should not be empty."
    # Check that the DataFrame includes the embeddings column.
    assert "_contains_embeddings" in df_result.columns, "Missing '_contains_embeddings' column in the result."

    # For each row, verify that metadata.embedding is a list of the expected length
    # and that the _contains_embeddings flag is True.
    for idx, row in df_result.iterrows():
        metadata = row.get("metadata", {})
        assert "embedding" in metadata, f"Row {idx} missing 'embedding' in metadata."
        embedding = metadata["embedding"]
        assert isinstance(embedding, list), f"Row {idx} metadata.embedding is not a list."
        assert (
            len(embedding) == _EMBEDDING_VECTOR_LENGTH
        ), f"Row {idx} metadata.embedding length is not {_EMBEDDING_VECTOR_LENGTH}, got {len(embedding)} instead."
        assert row.get("_contains_embeddings") is True, f"Row {idx} _contains_embeddings flag is not True."


# ----------------------------
# Successful splitting integration test
# ----------------------------
@pytest.mark.integration
@pytest.mark.parametrize("input_mode", ["dataframe", "string", "list"])
def test_transform_text_split_and_tokenize_integration_success(input_mode, monkeypatch):
    """
    Parameterized integration test for transform_text_split_and_tokenize.

    This test verifies that text documents can be split and tokenized correctly when provided as:

      - A preconstructed pandas DataFrame.
      - A single plain text string.
      - A list of plain text strings.

    When a plain text string or list of strings is provided, the function constructs a DataFrame
    where each row represents a TEXT document with the following structure:

      - source_name, source_id: Generated as "text_0", "text_1", etc.
      - content: Base64-encoded UTF-8 text.
      - document_type: Set to DocumentTypeEnum.TXT.
      - metadata: A dict containing:
          * content: The original text.
          * content_metadata: A dict with key "type" set to "text".
          * source_metadata: A dict with source info (including ISO8601 timestamps).
          * Other fields (audio_metadata, image_metadata, etc.) set to None or empty.
          * raise_on_failure: False.

    The function then delegates to an internal routine for splitting and tokenization.

    The resulting DataFrame is expected to be nonempty. If the text is split into multiple chunks,
    then a 'uuid' column is expected, and at least one chunk should differ from the original text.

    Parameters
    ----------
    input_mode : str
        One of "dataframe", "string", or "list", indicating the form of the input.
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture for setting environment variables.

    Returns
    -------
    None
    """
    monkeypatch.setenv("MODEL_PREDOWNLOAD_PATH", "/tmp")

    # Create a moderate-length text that will be split into chunks.
    text = "This is a test sentence. " * 50  # roughly 1400 characters

    # Integration configuration.
    _TOKENIZER = os.getenv("INGEST_SPLIT_TOKENIZER", "bert-base-uncased")
    _CHUNK_SIZE = int(os.getenv("INGEST_SPLIT_CHUNK_SIZE", 50))
    _CHUNK_OVERLAP = int(os.getenv("INGEST_SPLIT_CHUNK_OVERLAP", 10))
    _HF_ACCESS_TOKEN = os.getenv("INGEST_HF_ACCESS_TOKEN", None)
    split_source_types = ["text"]

    # Prepare input based on the parameter.
    if input_mode == "dataframe":
        # Build a DataFrame with the required structure.
        df_input = pd.DataFrame(
            {
                "source_name": ["doc1.txt"],
                "source_id": ["doc1.txt"],
                "content": ["dummy_base64"],  # Not used; actual text is in metadata.content.
                "document_type": [DocumentTypeEnum.TXT],
                "metadata": [
                    {
                        "audio_metadata": None,
                        "content": text,
                        "content_metadata": {"type": "text"},
                        "error_metadata": None,
                        "image_metadata": None,
                        "source_metadata": {
                            "source_id": "doc1.txt",
                            "source_name": "doc1.txt",
                            "source_type": "txt",
                        },
                        "text_metadata": None,
                        "raise_on_failure": False,
                    }
                ],
            }
        )
        input_arg = df_input
    elif input_mode == "string":
        # A single plain text string.
        input_arg = text
    elif input_mode == "list":
        # A list of plain text strings.
        input_arg = [text, text]
    else:
        pytest.skip("Invalid input mode.")

    # Call the splitting function.
    result_df = transform_text_split_and_tokenize(
        inputs=input_arg,
        tokenizer=_TOKENIZER,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        split_source_types=split_source_types,
        hugging_face_access_token=_HF_ACCESS_TOKEN,
    )

    # Validate that the result is nonempty.
    assert not result_df.empty, "Resulting DataFrame should not be empty."

    # If splitting occurred, a 'uuid' column is expected.
    if len(result_df) > 1:
        assert "uuid" in result_df.columns, "Missing 'uuid' column in the split results."

    # For each row, verify that document_type is 'text' and metadata.content is nonempty.
    for idx, row in result_df.iterrows():
        assert row["document_type"] == DocumentTypeEnum.TXT, f"Row {idx} document_type is not 'text'."
        metadata = row.get("metadata", {})
        assert "content" in metadata, f"Row {idx} missing 'content' in metadata."
        chunk_text = metadata["content"]
        assert isinstance(chunk_text, str) and chunk_text.strip() != "", f"Row {idx} has empty chunk content."

    # If multiple chunks are produced, at least one chunk should differ from the original text.
    if len(result_df) > 1:
        chunks = result_df["metadata"].apply(lambda m: m["content"]).tolist()
        assert any(
            chunk != text for chunk in chunks
        ), "All chunks are identical to the original text; splitting may have failed."


# ----------------------------
# Image captioning integration test
# ----------------------------
@pytest.mark.integration
@pytest.mark.parametrize("input_mode", ["dataframe", "tuple_string", "list_tuples_string", "list_tuples_bytesio"])
def test_transform_image_create_vlm_caption_parameterized(input_mode):
    # Locate the image file using environment variable or fallback.
    image_file = os.getenv("INGEST_IMAGE_FILE")
    if not image_file:
        project_root = get_project_root(__file__)
        if project_root:
            candidate = os.path.join(project_root, "data", "chart.png")
            if os.path.exists(candidate):
                image_file = candidate
        if not image_file:
            root_dir = find_root_by_pattern("data/chart.png")
            if root_dir:
                candidate = os.path.join(root_dir, "data", "chart.png")
                if os.path.exists(candidate):
                    image_file = candidate
    if not image_file or not os.path.exists(image_file):
        pytest.skip("No valid image file found for integration test.")

    # Load integration configuration.
    default_schema = ImageCaptionExtractionSchema()
    default_args = default_schema.model_dump() if hasattr(default_schema, "model_dump") else default_schema.dict()
    _VLM_API_KEY = os.getenv("INGEST_VLM_API_KEY", default_args["api_key"])
    _VLM_ENDPOINT = os.getenv("INGEST_VLM_ENDPOINT_URL", default_args["endpoint_url"])
    _VLM_PROMPT = os.getenv("INGEST_VLM_PROMPT", default_args["prompt"])
    _VLM_MODEL = os.getenv("INGEST_VLM_MODEL_NAME", default_args["model_name"])
    integration_args = {
        "api_key": _VLM_API_KEY,
        "prompt": _VLM_PROMPT,
        "endpoint_url": _VLM_ENDPOINT,
        "model_name": _VLM_MODEL,
    }

    # Prepare input according to mode.
    if input_mode == "dataframe":
        # Build a DataFrame manually.
        with open(image_file, "rb") as f:
            image_bytes = f.read()
        valid_png_base64 = base64.b64encode(image_bytes).decode("utf-8")
        df_input = pd.DataFrame(
            {
                "source_name": [image_file],
                "source_id": [image_file],
                "content": [valid_png_base64],
                "document_type": [DocumentTypeEnum.PNG],
                "metadata": [
                    {
                        "audio_metadata": None,
                        "content": valid_png_base64,
                        "content_metadata": {"type": "image"},
                        "error_metadata": None,
                        "image_metadata": {},  # To be populated with caption.
                        "source_metadata": {"source_id": image_file, "source_name": image_file, "source_type": "png"},
                        "text_metadata": None,
                    }
                ],
            }
        )
        input_arg = df_input

    elif input_mode == "tuple_string":
        # A single tuple with (file_path, document_type).
        input_arg = (image_file, DocumentTypeEnum.PNG)

    elif input_mode == "list_tuples_string":
        # A list of tuples with file paths (strings).
        input_arg = [(image_file, DocumentTypeEnum.PNG), (image_file, DocumentTypeEnum.PNG)]

    elif input_mode == "list_tuples_bytesio":
        # A list of tuples with BytesIO objects.
        with open(image_file, "rb") as f:
            image_bytes = f.read()
        bytes_io1 = BytesIO(image_bytes)
        bytes_io2 = BytesIO(image_bytes)
        input_arg = [(bytes_io1, DocumentTypeEnum.PNG), (bytes_io2, DocumentTypeEnum.PNG)]
    else:
        pytest.skip("Invalid input mode.")

    # Call the function under test.
    result_df = transform_image_create_vlm_caption(inputs=input_arg, **integration_args)

    # Validate that each row has a nonempty caption in metadata.image_metadata.
    for idx, row in result_df.iterrows():
        meta = row.get("metadata", {})
        image_meta = meta.get("image_metadata", {})
        assert "caption" in image_meta, f"{input_mode} mode, row {idx}: missing 'caption'."
        caption = image_meta.get("caption", "")
        assert isinstance(caption, str) and caption.strip() != "", f"{input_mode} mode, row {idx}: caption is empty."
