# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datetime import datetime

import pytest
from pydantic import ValidationError

from nv_ingest.schemas import validate_metadata

# TODO, add info message


# Helper function to generate valid base metadata for modification in tests
def get_valid_metadata():
    return {
        "content": "Example content",
        "source_metadata": {
            "source_name": "Source Name",
            "source_id": "123",
            "source_location": "Location",
            "source_type": "source_type_1",
            "collection_id": "456",
            "date_created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "summary": "Summary",
            "partition_id": 1,
            "access_level": 1,
        },
        "content_metadata": {
            "type": "text",
            "description": "Description",
            "page_number": 1,
            "hierarchy": {
                "page_count": 1,
                "page": 1,
                "block": 1,
                "line": 1,
                "span": 1,
                "nearby_objects": {
                    "text": {"content": [], "bbox": []},
                    "images": {"content": [], "bbox": []},
                    "structured": {"content": [], "bbox": []},
                },
            },
        },
        "text_metadata": {
            "text_type": "body",
            "summary": "Summary",
            "keywords": ["keyword1", "keyword2"],
            "language": "en",
            "text_location": (0, 1, 10, 20),
        },
        "image_metadata": {
            "image_type": "image_type_1",
            "structured_image_type": "image_type_1",
            "caption": "Caption",
            "text": "Text",
            "image_location": (0, 1, 10, 20),
        },
    }


# Test for successful validation
def test_validate_metadata_success():
    metadata = get_valid_metadata()
    assert validate_metadata(metadata)


# Test for missing required fields in main and sub-schemas
@pytest.mark.parametrize(
    "key",
    [
        # ("source_metadata"),
        # ("content_metadata"),
        #    ("text_metadata"),
        #    ("image_metadata")
    ],
)
def test_missing_required_fields_main_schema(key):
    metadata = get_valid_metadata()
    metadata.pop(key)
    with pytest.raises(ValidationError):
        validate_metadata(metadata)


@pytest.mark.parametrize(
    "sub_schema_key,missing_key",
    [
        ("source_metadata", "source_name"),
        ("source_metadata", "source_type"),
        ("content_metadata", "type"),
        ("text_metadata", "text_type"),
        # ("image_metadata", "image_type")
    ],
)
def test_missing_required_fields_sub_schemas(sub_schema_key, missing_key):
    metadata = get_valid_metadata()
    metadata[sub_schema_key].pop(missing_key)
    with pytest.raises(ValidationError):
        validate_metadata(metadata)


# Test for invalid enum values
@pytest.mark.parametrize(
    "sub_schema_key,enum_key,invalid_value",
    [
        #    ("source_metadata", "source_type", "invalid"),
        ("content_metadata", "type", "invalid"),
        ("text_metadata", "text_type", "invalid"),
        ("text_metadata", "language", "invalid"),
        #    ("image_metadata", "image_type", "invalid"),
        #    ("image_metadata", "structured_image_type", "invalid")
    ],
)
def test_invalid_enum_values(sub_schema_key, enum_key, invalid_value):
    metadata = get_valid_metadata()
    metadata[sub_schema_key][enum_key] = invalid_value
    with pytest.raises(ValidationError):
        validate_metadata(metadata)


# Test for incorrect data types
# This is an example; you should extend these tests for each field where type mismatches can occur.
@pytest.mark.parametrize(
    "key,value",
    [
        ("source_metadata", "not a dict"),  # Should be a dict
        (
            "source_metadata.date_created",
            "not a datetime",
        ),  # Incorrect type inside a dict
    ],
)
def test_incorrect_data_types(key, value):
    metadata = get_valid_metadata()
    if "." in key:
        sub_key, inner_key = key.split(".")
        metadata[sub_key][inner_key] = value
    else:
        metadata[key] = value
    with pytest.raises(ValidationError):
        validate_metadata(metadata)
