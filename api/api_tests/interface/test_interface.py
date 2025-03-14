# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch
import pandas as pd
from io import BytesIO
from pandas.testing import assert_frame_equal

from nv_ingest_api.internal.enums.common import DocumentTypeEnum
from nv_ingest_api.interface.transform import (
    transform_text_create_embeddings,
    transform_image_create_vlm_caption,
    transform_text_split_and_tokenize,
)


class TestTransformTextCreateEmbeddings(unittest.TestCase):
    """Test cases for transform_text_create_embeddings function."""

    @patch("nv_ingest_api.interface.transform.transform_create_text_embeddings_internal")
    def test_with_default_parameters(self, mock_internal):
        """Test transform_text_create_embeddings with only required parameters."""
        # Setup
        input_df = pd.DataFrame({"text": ["sample text"]})
        expected_df = pd.DataFrame({"text": ["sample text"], "embedding": [[0.1, 0.2, 0.3]]})
        mock_internal.return_value = (expected_df, None)

        # Execute
        result = transform_text_create_embeddings(inputs=input_df, api_key="test_api_key")

        # Verify
        assert_frame_equal(result, expected_df)
        mock_internal.assert_called_once()

        # Check that the config was properly built with default values
        _, kwargs = mock_internal.call_args
        self.assertEqual(kwargs["transform_config"].api_key, "test_api_key")

    @patch("nv_ingest_api.interface.transform.transform_create_text_embeddings_internal")
    def test_with_all_parameters(self, mock_internal):
        """Test transform_text_create_embeddings with all parameters provided."""
        # Setup
        input_df = pd.DataFrame({"text": ["sample text"]})
        expected_df = pd.DataFrame({"text": ["sample text"], "embedding": [[0.1, 0.2, 0.3]]})
        mock_internal.return_value = (expected_df, None)

        # Execute
        result = transform_text_create_embeddings(
            inputs=input_df,
            api_key="test_api_key",
            batch_size=100,
            embedding_model="test_model",
            embedding_nim_endpoint="test_endpoint",
            encoding_format="test_format",
            input_type="test_input_type",
            truncate="test_truncate",
        )

        # Verify
        assert_frame_equal(result, expected_df)
        mock_internal.assert_called_once()

        # Check that all parameters were passed to the config
        _, kwargs = mock_internal.call_args
        config = kwargs["transform_config"]
        self.assertEqual(config.api_key, "test_api_key")
        self.assertEqual(config.batch_size, 100)
        self.assertEqual(config.embedding_model, "test_model")
        self.assertEqual(config.embedding_nim_endpoint, "test_endpoint")
        self.assertEqual(config.encoding_format, "test_format")
        self.assertEqual(config.input_type, "test_input_type")
        self.assertEqual(config.truncate, "test_truncate")


class TestTransformImageCreateVlmCaption(unittest.TestCase):
    """Test cases for transform_image_create_vlm_caption function."""

    @patch("nv_ingest_api.interface.transform.transform_image_create_vlm_caption_internal")
    def test_with_dataframe_input(self, mock_internal):
        """Test transform_image_create_vlm_caption with DataFrame input."""
        # Setup
        input_metadata = {
            "content": "base64_content",
            "source_metadata": {"source": "test"},
            "content_metadata": {"type": "image"},
            "image_metadata": {},
            "raise_on_failure": False,
        }

        input_df = pd.DataFrame(
            {
                "source_name": ["image.png"],
                "source_id": ["image.png"],
                "content": ["base64_content"],
                "document_type": ["png"],
                "metadata": [input_metadata],
            }
        )

        expected_metadata = input_metadata.copy()
        expected_metadata["image_metadata"] = {"caption": "A test image"}

        expected_df = pd.DataFrame(
            {
                "source_name": ["image.png"],
                "source_id": ["image.png"],
                "content": ["base64_content"],
                "document_type": ["png"],
                "metadata": [expected_metadata],
            }
        )

        mock_internal.return_value = expected_df

        # Execute
        result = transform_image_create_vlm_caption(
            inputs=input_df,
            api_key="test_api_key",
            prompt="Describe this image",
            endpoint_url="test_endpoint",
            model_name="test_model",
        )

        # Verify
        assert_frame_equal(result, expected_df)
        mock_internal.assert_called_once()

        # Check that all parameters were passed to the config
        _, kwargs = mock_internal.call_args
        config = kwargs["transform_config"]
        self.assertEqual(config.api_key, "test_api_key")
        self.assertEqual(config.prompt, "Describe this image")
        self.assertEqual(config.endpoint_url, "test_endpoint")
        self.assertEqual(config.model_name, "test_model")

    @patch("nv_ingest_api.interface.transform.build_dataframe_from_files")
    @patch("nv_ingest_api.interface.transform.transform_image_create_vlm_caption_internal")
    def test_with_single_tuple_input(self, mock_internal, mock_build_df):
        """Test transform_image_create_vlm_caption with tuple input."""
        # Setup
        file_path = "test_image.png"
        doc_type = DocumentTypeEnum.PNG

        input_df = pd.DataFrame(
            {
                "source_name": [file_path],
                "source_id": [file_path],
                "content": ["base64_content"],
                "document_type": [doc_type],
                "metadata": [{}],
            }
        )

        expected_df = pd.DataFrame(
            {
                "source_name": [file_path],
                "source_id": [file_path],
                "content": ["base64_content"],
                "document_type": [doc_type],
                "metadata": [{"image_metadata": {"caption": "A test image"}}],
            }
        )

        mock_build_df.return_value = input_df
        mock_internal.return_value = expected_df

        # Execute
        result = transform_image_create_vlm_caption(inputs=(file_path, doc_type), api_key="test_api_key")

        # Verify
        assert_frame_equal(result, expected_df)
        mock_build_df.assert_called_once_with([file_path], [file_path], [file_path], [doc_type])
        mock_internal.assert_called_once()

    @patch("nv_ingest_api.interface.transform.build_dataframe_from_files")
    @patch("nv_ingest_api.interface.transform.transform_image_create_vlm_caption_internal")
    def test_with_multiple_tuple_input(self, mock_internal, mock_build_df):
        """Test transform_image_create_vlm_caption with list of tuples input."""
        # Setup
        file_paths = ["image1.png", "image2.png"]
        doc_types = [DocumentTypeEnum.PNG, DocumentTypeEnum.PNG]

        input_df = pd.DataFrame(
            {
                "source_name": file_paths,
                "source_id": file_paths,
                "content": ["base64_content1", "base64_content2"],
                "document_type": doc_types,
                "metadata": [{}, {}],
            }
        )

        expected_df = pd.DataFrame(
            {
                "source_name": file_paths,
                "source_id": file_paths,
                "content": ["base64_content1", "base64_content2"],
                "document_type": doc_types,
                "metadata": [
                    {"image_metadata": {"caption": "A test image 1"}},
                    {"image_metadata": {"caption": "A test image 2"}},
                ],
            }
        )

        mock_build_df.return_value = input_df
        mock_internal.return_value = expected_df

        tuples = list(zip(file_paths, doc_types))

        # Execute
        result = transform_image_create_vlm_caption(inputs=tuples, api_key="test_api_key")

        # Verify
        assert_frame_equal(result, expected_df)
        mock_build_df.assert_called_once_with(file_paths, file_paths, file_paths, doc_types)
        mock_internal.assert_called_once()

    @patch("nv_ingest_api.interface.transform.build_dataframe_from_files")
    @patch("nv_ingest_api.interface.transform.transform_image_create_vlm_caption_internal")
    def test_with_bytesio_input(self, mock_internal, mock_build_df):
        """Test transform_image_create_vlm_caption with BytesIO input."""
        # Setup
        bytes_io = BytesIO(b"test_image_data")
        doc_type = DocumentTypeEnum.PNG
        identifier = f"bytesio_{doc_type}"

        input_df = pd.DataFrame(
            {
                "source_name": [identifier],
                "source_id": [identifier],
                "content": ["base64_content"],
                "document_type": [doc_type],
                "metadata": [{}],
            }
        )

        expected_df = pd.DataFrame(
            {
                "source_name": [identifier],
                "source_id": [identifier],
                "content": ["base64_content"],
                "document_type": [doc_type],
                "metadata": [{"image_metadata": {"caption": "A test image"}}],
            }
        )

        mock_build_df.return_value = input_df
        mock_internal.return_value = expected_df

        # Execute
        result = transform_image_create_vlm_caption(inputs=(bytes_io, doc_type), api_key="test_api_key")

        # Verify
        assert_frame_equal(result, expected_df)
        mock_build_df.assert_called_once_with([bytes_io], [identifier], [identifier], [doc_type])
        mock_internal.assert_called_once()

    def test_with_invalid_input_type(self):
        """Test transform_image_create_vlm_caption with invalid input type."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            transform_image_create_vlm_caption(inputs=123, api_key="test_api_key")  # Not a DataFrame, tuple, or list

    def test_with_invalid_tuple_format(self):
        """Test transform_image_create_vlm_caption with invalid tuple format."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            transform_image_create_vlm_caption(
                inputs=[(123, DocumentTypeEnum.PNG, "extra")], api_key="test_api_key"  # Tuple with wrong length
            )


class TestTransformTextSplitAndTokenize(unittest.TestCase):
    """Test cases for transform_text_split_and_tokenize function."""

    @patch("nv_ingest_api.interface.transform.transform_text_split_and_tokenize_internal")
    def test_with_dataframe_input(self, mock_internal):
        """Test transform_text_split_and_tokenize with DataFrame input."""
        # Setup
        input_metadata = {
            "content": "This is a test document.",
            "content_metadata": {"type": "text"},
            "source_metadata": {"source_id": "doc1.txt", "source_name": "doc1.txt", "source_type": "txt"},
            "audio_metadata": None,
            "image_metadata": None,
            "text_metadata": None,
            "raise_on_failure": False,
        }

        input_df = pd.DataFrame(
            {
                "source_name": ["doc1.txt"],
                "source_id": ["doc1.txt"],
                "content": ["base64_content"],
                "document_type": [DocumentTypeEnum.TXT],
                "metadata": [input_metadata],
            }
        )

        expected_metadata = input_metadata.copy()
        expected_metadata["text_metadata"] = {"chunks": ["This is a test document."]}

        expected_df = pd.DataFrame(
            {
                "source_name": ["doc1.txt"],
                "source_id": ["doc1.txt"],
                "content": ["base64_content"],
                "document_type": [DocumentTypeEnum.TXT],
                "metadata": [expected_metadata],
            }
        )

        mock_internal.return_value = expected_df

        # Execute
        result = transform_text_split_and_tokenize(
            inputs=input_df, tokenizer="bert-base-uncased", chunk_size=512, chunk_overlap=50
        )

        # Verify
        assert_frame_equal(result, expected_df)
        mock_internal.assert_called_once()

        # Check that all parameters were passed to the config
        _, kwargs = mock_internal.call_args
        config = kwargs["transform_config"]
        self.assertEqual(config.tokenizer, "bert-base-uncased")
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)

        task_config = kwargs["task_config"]
        self.assertEqual(task_config["params"]["split_source_types"], ["text"])

    @patch("nv_ingest_api.interface.transform.build_dataframe_from_files")
    @patch("nv_ingest_api.interface.transform.transform_text_split_and_tokenize_internal")
    def test_with_string_input(self, mock_internal, mock_build_df):
        """Test transform_text_split_and_tokenize with string input."""
        # Setup
        input_text = "This is a test document."

        input_df = pd.DataFrame(
            {
                "source_name": ["text_0"],
                "source_id": ["text_0"],
                "content": ["base64_content"],
                "document_type": [DocumentTypeEnum.TXT],
                "metadata": [{}],
            }
        )

        expected_df = pd.DataFrame(
            {
                "source_name": ["text_0"],
                "source_id": ["text_0"],
                "content": ["base64_content"],
                "document_type": [DocumentTypeEnum.TXT],
                "metadata": [{"text_metadata": {"chunks": ["This is a test document."]}}],
            }
        )

        mock_build_df.return_value = input_df
        mock_internal.return_value = expected_df

        # Execute
        result = transform_text_split_and_tokenize(
            inputs=input_text, tokenizer="bert-base-uncased", chunk_size=512, chunk_overlap=50
        )

        # Verify
        assert_frame_equal(result, expected_df)

        # Verify BytesIO was created with the correct content and passed to build_dataframe_from_files
        args, _ = mock_build_df.call_args
        file_sources, source_names, source_ids, doc_types = args

        self.assertEqual(len(file_sources), 1)
        self.assertTrue(isinstance(file_sources[0], BytesIO))
        self.assertEqual(source_names, ["text_0"])
        self.assertEqual(source_ids, ["text_0"])
        self.assertEqual(doc_types, [DocumentTypeEnum.TXT])

        mock_internal.assert_called_once()

    @patch("nv_ingest_api.interface.transform.build_dataframe_from_files")
    @patch("nv_ingest_api.interface.transform.transform_text_split_and_tokenize_internal")
    def test_with_list_of_strings_input(self, mock_internal, mock_build_df):
        """Test transform_text_split_and_tokenize with list of strings input."""
        # Setup
        input_texts = ["Document one.", "Document two."]

        input_df = pd.DataFrame(
            {
                "source_name": ["text_0", "text_1"],
                "source_id": ["text_0", "text_1"],
                "content": ["base64_content1", "base64_content2"],
                "document_type": [DocumentTypeEnum.TXT, DocumentTypeEnum.TXT],
                "metadata": [{}, {}],
            }
        )

        expected_df = pd.DataFrame(
            {
                "source_name": ["text_0", "text_1"],
                "source_id": ["text_0", "text_1"],
                "content": ["base64_content1", "base64_content2"],
                "document_type": [DocumentTypeEnum.TXT, DocumentTypeEnum.TXT],
                "metadata": [
                    {"text_metadata": {"chunks": ["Document one."]}},
                    {"text_metadata": {"chunks": ["Document two."]}},
                ],
            }
        )

        mock_build_df.return_value = input_df
        mock_internal.return_value = expected_df

        # Execute
        result = transform_text_split_and_tokenize(
            inputs=input_texts,
            tokenizer="bert-base-uncased",
            chunk_size=512,
            chunk_overlap=50,
            split_source_types=["text", "document"],
            hugging_face_access_token="test_token",
        )

        # Verify
        assert_frame_equal(result, expected_df)

        # Verify BytesIO objects were created with correct content and passed to build_dataframe_from_files
        args, _ = mock_build_df.call_args
        file_sources, source_names, source_ids, doc_types = args

        self.assertEqual(len(file_sources), 2)
        self.assertTrue(all(isinstance(src, BytesIO) for src in file_sources))
        self.assertEqual(source_names, ["text_0", "text_1"])
        self.assertEqual(source_ids, ["text_0", "text_1"])
        self.assertEqual(doc_types, [DocumentTypeEnum.TXT, DocumentTypeEnum.TXT])

        # Verify task_config contains the custom split_source_types and hugging_face_access_token
        _, kwargs = mock_internal.call_args
        task_config = kwargs["task_config"]
        self.assertEqual(task_config["params"]["split_source_types"], ["text", "document"])
        self.assertEqual(task_config["params"]["hf_access_token"], "test_token")

    def test_with_invalid_input_type(self):
        """Test transform_text_split_and_tokenize with invalid input type."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            transform_text_split_and_tokenize(
                inputs=123,  # Not a DataFrame, string, or list of strings
                tokenizer="bert-base-uncased",
                chunk_size=512,
                chunk_overlap=50,
            )

    def test_with_invalid_list_input(self):
        """Test transform_text_split_and_tokenize with list containing non-string items."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            transform_text_split_and_tokenize(
                inputs=["Text", 123, "More text"],  # List with non-string items
                tokenizer="bert-base-uncased",
                chunk_size=512,
                chunk_overlap=50,
            )
