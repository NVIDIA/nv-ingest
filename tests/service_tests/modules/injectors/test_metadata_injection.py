# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import pandas as pd
import ray

from nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector import MetadataInjectionStage
from nv_ingest_api.internal.enums.common import (
    DocumentTypeEnum,
    ContentTypeEnum,
    AccessLevelEnum,
    LanguageEnum,
    TextTypeEnum,
)
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest_api.internal.schemas.meta.metadata_schema import ContentHierarchySchema


# Initialize Ray once at the module level
@pytest.fixture(scope="module", autouse=True)
def ray_fixture():
    """Initialize Ray for the entire test module."""
    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


# Create a proper test subclass of IngestControlMessage
class TestIngestControlMessage(IngestControlMessage):
    """Test subclass of IngestControlMessage with simplified initialization for testing."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to use as the payload.
        """
        super().__init__()
        self._payload = df


# Test class
class TestMetadataInjectionStage:
    """Test class for MetadataInjectionStage."""

    @pytest.fixture(scope="function", autouse=True)
    def setup(self):
        """
        Set up the test environment before each test.

        Creates a Ray actor instance of MetadataInjectionStage and yields control to the test.
        The actor is killed after the test completes.
        """
        # Create a config dictionary for initialization
        config = {}  # Replace with actual config if needed

        # Create a remote actor instance with proper stage_name for decorator annotations
        self.actor_ref = MetadataInjectionStage.remote(config, stage_name="metadata_injector")

        # Yield to run the test
        yield

        # Clean up after each test
        ray.kill(self.actor_ref)

    def test_no_update_required(self):
        """
        Test that no update occurs when rows already have valid metadata.

        This test verifies that when all rows in the DataFrame already contain
        valid metadata with a 'content' field, the MetadataInjectionStage does
        not modify the DataFrame.

        The test creates a DataFrame with two rows, both having properly structured
        metadata containing the 'content' field. After processing, the DataFrame
        should remain unchanged.

        Returns
        -------
        None
        """
        # Prepare a DataFrame where every row already has valid metadata
        df = pd.DataFrame(
            [
                {
                    "document_type": "pdf",
                    "content": "content1",
                    "source_id": 1,
                    "source_name": "SourceA",
                    "metadata": {
                        "content": "content1",
                        "other_info": "exists",
                    },
                },
                {
                    "document_type": "text",
                    "content": "content2",
                    "source_id": 2,
                    "source_name": "SourceB",
                    "metadata": {
                        "content": "content2",
                        "other_info": "exists",
                    },
                },
            ]
        )
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        # If no update was necessary, the payload remains unchanged
        pd.testing.assert_frame_equal(result.payload(), df)

    def test_update_required_missing_metadata(self):
        """
        Test metadata injection when 'metadata' field is missing.

        This test verifies that when a row is missing the 'metadata' field,
        the MetadataInjectionStage adds the correct metadata structure with
        appropriate values based on the document type and other row data.

        The test creates a single row with a PDF document type but no metadata.
        After processing, the row should have a properly structured metadata field.

        Returns
        -------
        None
        """
        # Row missing the 'metadata' key
        df = pd.DataFrame(
            [
                {
                    "document_type": "pdf",
                    "content": "pdf content",
                    "source_id": 10,
                    "source_name": "PDF_Source",
                }
            ]
        )
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        updated_df = result.payload()
        metadata = updated_df.loc[0, "metadata"]

        expected_type = doc_type_to_content_type(DocumentTypeEnum("pdf")).name.lower()
        assert isinstance(metadata, dict)
        assert metadata["content"] == "pdf content"

        # Assert content_metadata defaults
        cm = metadata["content_metadata"]
        assert cm["type"] == expected_type
        assert cm["page_number"] == -1
        assert cm["description"] == ""
        assert cm["hierarchy"] == ContentHierarchySchema().model_dump()
        assert cm["subtype"] == ""
        assert cm["start_time"] == -1
        assert cm["end_time"] == -1

        assert metadata["error_metadata"] is None
        assert metadata["image_metadata"] is None
        assert metadata["text_metadata"] is None

        # Assert source_metadata defaults
        sm = metadata["source_metadata"]
        assert sm["source_id"] == 10
        assert sm["source_name"] == "PDF_Source"
        assert sm["source_type"] == "pdf"
        assert sm["source_location"] == ""
        assert sm["collection_id"] == ""
        assert isinstance(sm["date_created"], str) and sm["date_created"]
        assert isinstance(sm["last_modified"], str) and sm["last_modified"]
        assert sm["summary"] == ""
        assert sm["partition_id"] == -1
        assert sm["access_level"] == AccessLevelEnum.UNKNOWN.value

    def test_update_required_non_dict_metadata(self):
        """
        Test metadata injection when 'metadata' is not a dictionary.

        This test verifies that when a row has a 'metadata' field that is not
        a dictionary (in this case, a string), the MetadataInjectionStage replaces
        it with a properly structured metadata dictionary.

        The test uses an image document type (png) and checks that the appropriate
        image-specific metadata is generated.

        Returns
        -------
        None
        """
        # Row where existing metadata is not a dict
        df = pd.DataFrame(
            [
                {
                    "document_type": "png",
                    "content": "image content",
                    "source_id": 20,
                    "source_name": "Image_Source",
                    "metadata": "invalid_metadata",
                }
            ]
        )
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        updated_df = result.payload()
        metadata = updated_df.loc[0, "metadata"]

        expected_type = doc_type_to_content_type(DocumentTypeEnum("png")).name.lower()
        assert metadata["content"] == "image content"

        # Assert content_metadata defaults
        cm = metadata["content_metadata"]
        assert cm["type"] == expected_type
        assert cm["page_number"] == -1
        assert cm["description"] == ""
        assert cm["hierarchy"] == ContentHierarchySchema().model_dump()
        assert cm["subtype"] == ""
        assert cm["start_time"] == -1
        assert cm["end_time"] == -1

        assert metadata["error_metadata"] is None

        # Assert image_metadata defaults for image type
        im = metadata["image_metadata"]
        assert im["image_type"] == "png"
        assert im["structured_image_type"] == ContentTypeEnum.NONE.value
        assert im["caption"] == ""
        assert im["text"] == ""
        assert im["image_location"] == (0, 0, 0, 0)
        assert im["image_location_max_dimensions"] == (0, 0)
        assert im["uploaded_image_url"] == ""
        assert im["width"] == 0
        assert im["height"] == 0

        assert metadata["text_metadata"] is None

        # Assert source_metadata defaults
        sm = metadata["source_metadata"]
        assert sm["source_id"] == 20
        assert sm["source_name"] == "Image_Source"
        assert sm["source_type"] == "png"
        assert sm["source_location"] == ""
        assert sm["collection_id"] == ""
        assert isinstance(sm["date_created"], str) and sm["date_created"]
        assert isinstance(sm["last_modified"], str) and sm["last_modified"]
        assert sm["summary"] == ""
        assert sm["partition_id"] == -1
        assert sm["access_level"] == AccessLevelEnum.UNKNOWN.value

    def test_update_required_missing_content_in_metadata(self):
        """
        Test metadata injection when metadata exists but 'content' key is missing.

        This test verifies that when a row has a metadata dictionary that doesn't
        contain the required 'content' key, the MetadataInjectionStage updates the
        metadata with a properly structured dictionary that includes the content
        and appropriate type-specific metadata.

        The test uses a text document type and verifies that text-specific metadata
        is correctly generated.

        Returns
        -------
        None
        """
        # Row with a metadata dict that exists but is missing the 'content' key
        df = pd.DataFrame(
            [
                {
                    "document_type": "text",
                    "content": "textual content",
                    "source_id": 30,
                    "source_name": "Text_Source",
                    "metadata": {"other": "value"},
                }
            ]
        )
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        updated_df = result.payload()
        metadata = updated_df.loc[0, "metadata"]

        expected_type = doc_type_to_content_type(DocumentTypeEnum("text")).name.lower()
        assert metadata["content"] == "textual content"

        # Assert content_metadata defaults
        cm = metadata["content_metadata"]
        assert cm["type"] == expected_type
        assert cm["page_number"] == -1
        assert cm["description"] == ""
        assert cm["hierarchy"] == ContentHierarchySchema().model_dump()
        assert cm["subtype"] == ""
        assert cm["start_time"] == -1
        assert cm["end_time"] == -1

        # Assert text_metadata defaults for text type
        tm = metadata["text_metadata"]
        assert tm["text_type"] == TextTypeEnum.DOCUMENT.value
        assert tm["summary"] == ""
        assert tm["keywords"] == ""
        assert tm["language"] == LanguageEnum.UNKNOWN.value
        assert tm["text_location"] == (0, 0, 0, 0)
        assert tm["text_location_max_dimensions"] == (0, 0, 0, 0)

        assert metadata["image_metadata"] is None
        assert metadata["error_metadata"] is None

        # Assert source_metadata defaults
        sm = metadata["source_metadata"]
        assert sm["source_id"] == 30
        assert sm["source_name"] == "Text_Source"
        assert sm["source_type"] == "text"
        assert sm["source_location"] == ""
        assert sm["collection_id"] == ""
        assert isinstance(sm["date_created"], str) and sm["date_created"]
        assert isinstance(sm["last_modified"], str) and sm["last_modified"]
        assert sm["summary"] == ""
        assert sm["partition_id"] == -1
        assert sm["access_level"] == AccessLevelEnum.UNKNOWN.value

    def test_audio_document_type(self):
        """
        Test that audio content types generate the correct metadata structure.

        This test verifies that when processing a row with an audio document type,
        the MetadataInjectionStage correctly sets up the audio-specific metadata
        while leaving other content-type-specific metadata as None.

        Returns
        -------
        None
        """
        # Test with an audio document type
        df = pd.DataFrame(
            [
                {
                    "document_type": "mp3",
                    "content": "audio content",
                    "source_id": 40,
                    "source_name": "Audio_Source",
                }
            ]
        )
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        updated_df = result.payload()
        metadata = updated_df.loc[0, "metadata"]

        expected_type = doc_type_to_content_type(DocumentTypeEnum("mp3")).name.lower()
        assert metadata["content"] == "audio content"

        # Assert content_metadata defaults
        cm = metadata["content_metadata"]
        assert cm["type"] == expected_type
        assert cm["page_number"] == -1
        assert cm["description"] == ""
        assert cm["hierarchy"] == ContentHierarchySchema().model_dump()
        assert cm["subtype"] == ""
        assert cm["start_time"] == -1
        assert cm["end_time"] == -1

        # Assert audio_metadata defaults for audio type
        am = metadata["audio_metadata"]
        assert am["audio_type"] == "mp3"
        assert am["audio_transcript"] == ""

        assert metadata["image_metadata"] is None
        assert metadata["text_metadata"] is None
        assert metadata["error_metadata"] is None

        # Assert source_metadata defaults
        sm = metadata["source_metadata"]
        assert sm["source_id"] == 40
        assert sm["source_name"] == "Audio_Source"
        assert sm["source_type"] == "mp3"
        assert sm["source_location"] == ""
        assert sm["collection_id"] == ""
        assert isinstance(sm["date_created"], str) and sm["date_created"]
        assert isinstance(sm["last_modified"], str) and sm["last_modified"]
        assert sm["summary"] == ""
        assert sm["partition_id"] == -1
        assert sm["access_level"] == AccessLevelEnum.UNKNOWN.value

    def test_empty_dataframe(self):
        """
        Test that an empty DataFrame is handled gracefully.

        This test verifies that when an empty DataFrame is passed to the
        MetadataInjectionStage, it is processed without errors and returned
        unchanged.

        Returns
        -------
        None
        """
        # An empty DataFrame should be handled gracefully
        df = pd.DataFrame([])
        msg = TestIngestControlMessage(df)
        # Call the remote method and get the result
        result = ray.get(self.actor_ref.on_data.remote(msg))
        pd.testing.assert_frame_equal(result.payload(), df)

    def test_inner_exception_on_invalid_document_type(self):
        """
        Test that an invalid document type raises an exception.

        This test verifies that when a row contains an invalid document type
        (one not defined in DocumentTypeEnum), an exception is raised during
        processing.

        The test expects the on_data method to propagate the exception raised
        by DocumentTypeEnum when it encounters an invalid value.

        Returns
        -------
        None
        """
        # If the document_type is invalid, DocumentTypeEnum() should raise an exception
        df = pd.DataFrame(
            [
                {
                    "document_type": "invalid",
                    "content": "content",
                    "source_id": 3,
                    "source_name": "SourceX",
                }
            ]
        )
        msg = TestIngestControlMessage(df)
        # Expect an exception when calling the remote method
        msg = ray.get(self.actor_ref.on_data.remote(msg))
        is_failed = msg.get_metadata("cm_failed", False)
        assert is_failed
        # Add more specific check for decorator's failure annotation
        failure_reason = msg.get_metadata("cm_failed_reason", "")
        assert "Error in on_data" in failure_reason  # General check for error origin

        found_detailed_annotation = False
        for key, value in msg.get_metadata().items():
            if isinstance(key, str) and key.startswith("annotation::") and isinstance(value, dict):
                if value.get("task_id") == "metadata_injector" and value.get("task_result") == "FAILURE":
                    found_detailed_annotation = True
                    # Optionally, assert specific details from the value if needed
                    # For example: assert "invalid" in value.get("message", "").lower()
                    break
        assert found_detailed_annotation, "Detailed failure annotation from metadata_injector not found or incorrect."
