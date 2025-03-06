# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import logging
import pandas as pd
import pytest

from api.tests.utilities_for_test import find_root_by_pattern, get_git_root
from nv_ingest_api.interface.transform import (
    transform_text_create_embeddings,
    transform_text_split_and_tokenize,
    transform_image_create_vlm_caption,
)
from nv_ingest_api.internal.enums.common import DocumentTypeEnum, ContentTypeEnum
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import EmbedExtractionsSchema

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
    default_schema = EmbedExtractionsSchema()
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
    result = transform_text_create_embeddings(df_ledger=df_ledger, **integration_args)

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


# ----------------------------
# Successful splitting integration test
# ----------------------------
@pytest.mark.integration
def test_transform_text_split_and_tokenize_integration_success(monkeypatch):
    # Ensure a valid model predownload path so that os.path.join doesn't receive None.
    monkeypatch.setenv("MODEL_PREDOWNLOAD_PATH", "/tmp")

    # Create a moderate-length text that will be split into chunks.
    text = "This is a test sentence. " * 50  # moderate text length

    df_ledger = pd.DataFrame(
        {
            "source_name": ["./data/split_success.txt"],
            "source_id": ["./data/split_success.txt"],
            "content": ["dummy_base64"],  # not used for splitting; actual text is in metadata.
            "document_type": [DocumentTypeEnum.TXT],
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": text,
                    "content_metadata": {"type": ContentTypeEnum.STRUCTURED},
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/split_success.txt",
                        "source_name": "./data/split_success.txt",
                        "source_type": "text",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    # Read configuration from environment variables with defaults.
    _TOKENIZER = os.getenv("INGEST_SPLIT_TOKENIZER", "bert-base-uncased")
    _CHUNK_SIZE = int(os.getenv("INGEST_SPLIT_CHUNK_SIZE", 50))
    _CHUNK_OVERLAP = int(os.getenv("INGEST_SPLIT_CHUNK_OVERLAP", 10))
    _HF_ACCESS_TOKEN = os.getenv("INGEST_HF_ACCESS_TOKEN", None)
    split_source_types = ["text"]

    # Call the splitting function.
    result_df = transform_text_split_and_tokenize(
        df_ledger=df_ledger,
        tokenizer=_TOKENIZER,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        split_source_types=split_source_types,
        hugging_face_access_token=_HF_ACCESS_TOKEN,
    )

    # Validate that splitting was successful.
    assert not result_df.empty, "Resulting DataFrame should not be empty."
    # We expect more rows than the single input row because the text should be split.
    assert len(result_df) > 1, "Expected the document to be split into multiple chunks."
    # Verify that each split document includes a generated UUID.
    assert "uuid" in result_df.columns, "Missing 'uuid' column in the split results."
    for idx, row in result_df.iterrows():
        # Document type should be preserved.
        assert row["document_type"] == ContentTypeEnum.TEXT.value, f"Row {idx} document_type is not 'text'."
        metadata = row.get("metadata", {})
        assert "content" in metadata, f"Row {idx} missing 'content' in metadata."
        chunk_text = metadata["content"]
        # Each chunk should be nonempty and shorter than the full original text.
        assert isinstance(chunk_text, str) and chunk_text.strip() != "", f"Row {idx} has empty chunk content."
        assert len(chunk_text) < len(text), f"Row {idx} chunk content should be shorter than the original text."


# ----------------------------
# Failure integration test
# ----------------------------
@pytest.mark.integration
def test_transform_text_split_and_tokenize_integration_failure(monkeypatch):
    # Ensure MODEL_PREDOWNLOAD_PATH is set to avoid os.path.join errors.
    monkeypatch.setenv("MODEL_PREDOWNLOAD_PATH", "/tmp")

    # Create a long text that will generate a tokenized sequence exceeding the model's maximum length.
    too_long_text = "This is a test sentence. " * 300  # produces many tokens

    df_ledger = pd.DataFrame(
        {
            "source_name": ["./data/split_failure.txt"],
            "source_id": ["./data/split_failure.txt"],
            "content": ["dummy_base64"],
            "document_type": [DocumentTypeEnum.TXT],
            "metadata": [
                {
                    "audio_metadata": None,
                    "content": too_long_text,
                    "content_metadata": {"type": ContentTypeEnum.STRUCTURED},
                    "error_metadata": None,
                    "image_metadata": None,
                    "source_metadata": {
                        "source_id": "./data/split_failure.txt",
                        "source_name": "./data/split_failure.txt",
                        "source_type": "text",
                    },
                    "text_metadata": None,
                }
            ],
        }
    )

    _TOKENIZER = os.getenv("INGEST_SPLIT_TOKENIZER", "bert-base-uncased")
    # Intentionally set chunk_size high to force the error from the tokenizer.
    _CHUNK_SIZE = int(os.getenv("INGEST_SPLIT_CHUNK_SIZE", 1200))
    _CHUNK_OVERLAP = int(os.getenv("INGEST_SPLIT_CHUNK_OVERLAP", 10))
    _HF_ACCESS_TOKEN = os.getenv("INGEST_HF_ACCESS_TOKEN", None)
    split_source_types = ["text"]

    transform_text_split_and_tokenize(
        df_ledger=df_ledger,
        tokenizer=_TOKENIZER,
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        split_source_types=split_source_types,
        hugging_face_access_token=_HF_ACCESS_TOKEN,
    )


@pytest.mark.integration
def test_transform_image_create_vlm_caption_integration():
    # First, check if an image file is provided via environment variable.
    image_file = os.getenv("INGEST_IMAGE_FILE")

    if not image_file:
        # Use the specialized get_git_root function first.
        git_root = get_git_root(__file__)
        if git_root:
            candidate = os.path.join(git_root, "data", "chart.png")
            if os.path.exists(candidate):
                image_file = candidate
        # If get_git_root did not return a valid root or file wasn't found, fallback to the heuristic.
        if not image_file:
            root_dir = find_root_by_pattern("data/chart.png")
            if root_dir:
                candidate = os.path.join(root_dir, "data", "chart.png")
                if os.path.exists(candidate):
                    image_file = candidate

    if not image_file or not os.path.exists(image_file):
        pytest.skip("No valid image file found for integration test.")

    # Read the image file and encode it to base64.
    with open(image_file, "rb") as f:
        image_bytes = f.read()
    valid_png_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Build a ledger DataFrame with one image row.
    df_ledger = pd.DataFrame(
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

    # Build the default input arguments from the ImageCaptionExtractionSchema.
    default_schema = ImageCaptionExtractionSchema()
    default_args = default_schema.model_dump() if hasattr(default_schema, "model_dump") else default_schema.dict()

    # Pull configuration values from the environment with fallbacks to schema defaults.
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

    # Call the function under test with the constructed parameters.
    result_df = transform_image_create_vlm_caption(df_ledger=df_ledger, **integration_args)

    # Validate that for each image row, metadata.image_metadata.caption is populated.
    for idx, row in result_df.iterrows():
        metadata = row.get("metadata", {})
        image_meta = metadata.get("image_metadata", {})
        assert "caption" in image_meta, f"Row {idx} missing 'caption' in image_metadata."
        caption = image_meta.get("caption", "")
        assert isinstance(caption, str) and caption.strip() != "", f"Row {idx} has an empty caption."
