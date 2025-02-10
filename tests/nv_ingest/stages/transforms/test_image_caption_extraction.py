import unittest
import pandas as pd
from typing import Any, Dict
from unittest.mock import patch, MagicMock

# Import the functions under test from the module.
from nv_ingest.stages.transforms.image_caption_extraction import (
    _prepare_dataframes_mod,
    _generate_captions,
    caption_extract_stage,
)

# Define the module path for patching.
MODULE_UNDER_TEST = "nv_ingest.stages.transforms.image_caption_extraction"


# For testing _prepare_dataframes_mod we need to supply a dummy for ContentTypeEnum.
class DummyContentTypeEnum:
    IMAGE = "image"


# A dummy BaseModel that simply returns a dictionary when model_dump is called.
class DummyBaseModel:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def model_dump(self):
        return self.data


class TestImageCaptionExtraction(unittest.TestCase):

    # ------------------------------
    # Tests for _prepare_dataframes_mod
    # ------------------------------
    def test_prepare_dataframes_mod_empty_df(self):
        df = pd.DataFrame()
        full_df, image_df, bool_index = _prepare_dataframes_mod(df)
        self.assertTrue(full_df.empty)
        self.assertTrue(image_df.empty)
        self.assertTrue(bool_index.empty)

    def test_prepare_dataframes_mod_no_document_type(self):
        df = pd.DataFrame({"some_column": [1, 2, 3]})
        full_df, image_df, bool_index = _prepare_dataframes_mod(df)
        self.assertTrue(full_df.equals(df))
        self.assertTrue(image_df.empty)
        self.assertTrue(bool_index.empty)

    @patch(f"{MODULE_UNDER_TEST}.ContentTypeEnum", new=DummyContentTypeEnum)
    def test_prepare_dataframes_mod_valid(self):
        # Build a DataFrame with a "document_type" column.
        df = pd.DataFrame({"document_type": ["image", "text", "image"], "other_column": [10, 20, 30]})
        full_df, image_df, bool_index = _prepare_dataframes_mod(df)
        # Only the rows with document_type equal to "image" should be selected.
        expected_mask = df["document_type"] == "image"
        self.assertTrue(bool_index.equals(expected_mask))
        self.assertEqual(len(image_df), 2)
        self.assertTrue((image_df["document_type"] == "image").all())

    # ------------------------------
    # Tests for _generate_captions
    # ------------------------------
    @patch(f"{MODULE_UNDER_TEST}.scale_image_to_encoding_size")
    @patch(f"{MODULE_UNDER_TEST}.NimClient")
    def test_generate_captions_success(self, mock_NimClient, mock_scale):
        # For each call, just append "_scaled" to the input image.
        mock_scale.side_effect = lambda b64: (b64 + "_scaled", None)
        # Create a fake NimClient instance whose infer() returns a list of captions.
        fake_client = MagicMock()
        fake_client.infer.return_value = ["caption1", "caption2"]
        mock_NimClient.return_value = fake_client

        base64_images = ["image1", "image2"]
        prompt = "Test prompt"
        api_key = "dummy_api_key"
        endpoint_url = "http://dummy-endpoint"
        model_name = "dummy_model"

        captions = _generate_captions(base64_images, prompt, api_key, endpoint_url, model_name)

        # Check that scale_image_to_encoding_size was called once per image.
        self.assertEqual(mock_scale.call_count, len(base64_images))
        # Verify that the scaled images are passed to NimClient.infer.
        expected_data = {
            "base64_images": ["image1_scaled", "image2_scaled"],
            "prompt": prompt,
        }
        fake_client.infer.assert_called_once_with(expected_data, model_name=model_name, max_batch_size=1)
        # Check that the returned captions match the fake infer result.
        self.assertEqual(captions, ["caption1", "caption2"])

    # ------------------------------
    # Tests for caption_extract_stage
    # ------------------------------
    @patch(f"{MODULE_UNDER_TEST}._generate_captions")
    def test_caption_extract_stage_success(self, mock_generate_captions):
        # Setup the mock to return a predictable list of captions.
        mock_generate_captions.return_value = ["caption1", "caption2"]

        # Create a DataFrame with three rows; two with image content and one with non-image.
        data = [
            {"metadata": {"content": "img1", "content_metadata": {"type": "image"}}},
            {"metadata": {"content": "img2", "content_metadata": {"type": "image"}}},
            {"metadata": {"content": "txt1", "content_metadata": {"type": "text"}}},
        ]
        df = pd.DataFrame(data)

        # Prepare task properties and validated config.
        task_props = {
            "api_key": "dummy_api_key",
            "prompt": "Test prompt",
            "endpoint_url": "http://dummy-endpoint",
            "model_name": "dummy_model",
        }
        # Simulate validated_config as an object with attributes.
        DummyConfig = type(
            "DummyConfig",
            (),
            {
                "api_key": "dummy_api_key_conf",
                "prompt": "Test prompt conf",
                "endpoint_url": "http://dummy-endpoint-conf",
                "model_name": "dummy_model_conf",
            },
        )
        validated_config = DummyConfig()

        # Call caption_extract_stage.
        updated_df = caption_extract_stage(df.copy(), task_props, validated_config)

        # Verify that _generate_captions was called once with the list of base64 images for image rows.
        mock_generate_captions.assert_called_once()
        args, _ = mock_generate_captions.call_args
        # Expect the two image contents.
        self.assertEqual(args[0], ["img1", "img2"])
        self.assertEqual(args[1], task_props["prompt"])
        self.assertEqual(args[2], task_props["api_key"])
        self.assertEqual(args[3], task_props["endpoint_url"])
        self.assertEqual(args[4], task_props["model_name"])

        # Check that the metadata for image rows got updated with the corresponding captions.
        meta0 = updated_df.at[0, "metadata"]
        meta1 = updated_df.at[1, "metadata"]
        self.assertEqual(meta0.get("image_metadata", {}).get("caption"), "caption1")
        self.assertEqual(meta1.get("image_metadata", {}).get("caption"), "caption2")
        # The non-image row (index 2) should remain unchanged.
        meta2 = updated_df.at[2, "metadata"]
        self.assertNotIn("image_metadata", meta2)

    @patch(f"{MODULE_UNDER_TEST}._generate_captions")
    def test_caption_extract_stage_no_images(self, mock_generate_captions):
        # Create a DataFrame with no image rows.
        data = [
            {"metadata": {"content": "txt1", "content_metadata": {"type": "text"}}},
            {"metadata": {"content": "txt2", "content_metadata": {"type": "text"}}},
        ]
        df = pd.DataFrame(data)
        task_props = {
            "api_key": "dummy_api_key",
            "prompt": "Test prompt",
            "endpoint_url": "http://dummy-endpoint",
            "model_name": "dummy_model",
        }
        DummyConfig = type(
            "DummyConfig",
            (),
            {
                "api_key": "dummy_api_key_conf",
                "prompt": "Test prompt conf",
                "endpoint_url": "http://dummy-endpoint-conf",
                "model_name": "dummy_model_conf",
            },
        )
        validated_config = DummyConfig()

        # With no images, _generate_captions should not be called.
        updated_df = caption_extract_stage(df.copy(), task_props, validated_config)
        mock_generate_captions.assert_not_called()
        # The returned DataFrame should be the same as the input.
        self.assertTrue(df.equals(updated_df))

    @patch(f"{MODULE_UNDER_TEST}._generate_captions")
    def test_caption_extract_stage_error(self, mock_generate_captions):
        # Simulate an error when generating captions.
        mock_generate_captions.side_effect = Exception("Test error")
        data = [{"metadata": {"content": "img1", "content_metadata": {"type": "image"}}}]
        df = pd.DataFrame(data)
        task_props = {
            "api_key": "dummy_api_key",
            "prompt": "Test prompt",
            "endpoint_url": "http://dummy-endpoint",
            "model_name": "dummy_model",
        }
        DummyConfig = type(
            "DummyConfig",
            (),
            {
                "api_key": "dummy_api_key_conf",
                "prompt": "Test prompt conf",
                "endpoint_url": "http://dummy-endpoint-conf",
                "model_name": "dummy_model_conf",
            },
        )
        validated_config = DummyConfig()

        with self.assertRaises(Exception) as context:
            caption_extract_stage(df.copy(), task_props, validated_config)
        self.assertIn("Test error", str(context.exception))
