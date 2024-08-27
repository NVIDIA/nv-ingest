# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json

import pandas as pd
import pytest

from nv_ingest.modules.transforms.associate_nearby_text import _associate_nearby_text_blocks
from nv_ingest.modules.transforms.associate_nearby_text import _get_bbox
from nv_ingest.modules.transforms.associate_nearby_text import _get_center
from nv_ingest.modules.transforms.associate_nearby_text import _is_nearby_text
from nv_ingest.schemas.metadata_schema import TextTypeEnum

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if MORPHEUS_IMPORT_OK and CUDA_DRIVER_OK:
    pass


@pytest.fixture
def create_sample_df():
    """Fixture to create a sample DataFrame with varying metadata scenarios."""
    data = {
        "metadata": [
            json.dumps(
                {
                    "content_metadata": {"hierarchy": {"page": 1}},
                    "image_metadata": {"image_location": (0, 0, 10, 10)},
                    "content": "Image",
                }
            ),
            json.dumps(
                {
                    "content_metadata": {"hierarchy": {"page": 1}, "text_type": "NEARBY_BLOCK"},
                    "text_metadata": {"text_location": (1, 1, 2, 2)},
                    "content": "Text Block",
                }
            ),
            json.dumps({"content_metadata": {"hierarchy": {"page": 2}}, "image_metadata": None, "content": "Text"}),
        ]
    }
    return pd.DataFrame(data)


def test_get_center():
    assert _get_center((0, 0, 10, 10)) == (5, 5), "Should return the center of the bounding box"
    assert _get_center((1, 2, 3, 4)) == (2, 3), "Should return the center of the bounding box with non-zero origin"


def test_is_nearby_text_true():
    row = {"text_metadata": {"text_type": TextTypeEnum.NEARBY_BLOCK}}
    assert _is_nearby_text(row) is True, "Should identify text as NEARBY_BLOCK correctly"


def test_is_nearby_text_false():
    row = {"text_metadata": {"text_type": "OTHER_TYPE"}}
    assert _is_nearby_text(row) is False, "Should correctly identify non-NEARBY_BLOCK text"


def test_is_nearby_text_no_metadata():
    row = {}
    assert _is_nearby_text(row) is False, "Should return False when no text_metadata is present"


def test_get_bbox_from_text_metadata():
    row = {"text_metadata": {"text_location": (1, 2, 3, 4)}}
    assert _get_bbox(row) == (1, 2, 3, 4), "Should extract bbox from text metadata correctly"


def test_get_bbox_from_image_metadata():
    row = {"image_metadata": {"image_location": (5, 6, 7, 8)}}
    assert _get_bbox(row) == (5, 6, 7, 8), "Should extract bbox from image metadata when text metadata is absent"


def test_get_bbox_no_metadata():
    row = {}
    assert _get_bbox(row) is None, "Should return None when no relevant metadata is present"


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_no_images(create_sample_df):
    """Test behavior when there are no images in the dataframe."""
    df = create_sample_df
    # Remove image metadata to simulate no images scenario
    df.at[0, "metadata"] = json.dumps({"content_metadata": {"hierarchy": {"page": 1}}})
    result_df = _associate_nearby_text_blocks(df, n_neighbors=1)
    metadata = json.loads(result_df.iloc[0]["metadata"])
    assert "image_metadata" not in metadata, "No images should be present in the metadata."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_no_text_blocks_but_images(create_sample_df):
    """Test behavior when there are images but no text blocks."""
    df = create_sample_df
    # Remove text block to simulate no text blocks scenario
    df = df.drop(index=1)
    result_df = _associate_nearby_text_blocks(df, n_neighbors=1)
    metadata = json.loads(result_df.iloc[0]["metadata"])
    assert "nearby_objects" not in metadata["content_metadata"]["hierarchy"], "No text blocks should be associated."


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_empty_dataframe():
    """Test behavior when the dataframe is empty."""
    df = pd.DataFrame()

    with pytest.raises(KeyError):
        result_df = _associate_nearby_text_blocks(df, n_neighbors=1)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_page_without_images_or_text_blocks(create_sample_df):
    """Test behavior when pages have neither images nor text blocks."""
    df = create_sample_df
    # Modify to have no valid metadata
    df.at[0, "metadata"] = json.dumps({"content_metadata": {"hierarchy": {"page": 3}}})
    df.at[1, "metadata"] = json.dumps({"content_metadata": {"hierarchy": {"page": 3}}})
    result_df = _associate_nearby_text_blocks(df, n_neighbors=1)
    metadata = json.loads(result_df.iloc[0]["metadata"])
    assert "nearby_objects" not in metadata["content_metadata"]["hierarchy"], "No associations should be present."
