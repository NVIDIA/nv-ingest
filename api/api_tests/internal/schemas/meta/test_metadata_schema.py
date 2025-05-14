# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import (  # Adjust import path
    ContentTypeEnum,
    validate_metadata,
    ImageMetadataSchema,
    TableMetadataSchema,
    SourceMetadataSchema,
    AudioMetadataSchema,
    datetools,
    MetadataSchema,
)


def test_valid_text_metadata_auto_nulls_others():
    data = {
        "content": "example text",
        "content_metadata": {"type": "text"},
        "text_metadata": {"text_type": "body"},  # ✅ Use valid enum value
        "image_metadata": {"image_type": "pdf"},  # Should be nullified by validator
        "audio_metadata": {"audio_transcript": "hello"},  # Should be nullified
    }
    result = validate_metadata(data)
    assert result.content_metadata.type == ContentTypeEnum.TEXT
    assert result.text_metadata.text_type == "body"
    assert result.image_metadata is None
    assert result.audio_metadata is None


def test_valid_image_metadata_auto_nulls_others():
    data = {
        "content_metadata": {"type": "image"},
        "image_metadata": {"image_type": "pdf", "width": -100, "height": -100},
        "text_metadata": {"text_type": "plain"},
    }
    result = validate_metadata(data)
    assert result.content_metadata.type == ContentTypeEnum.IMAGE
    assert isinstance(result.image_metadata, ImageMetadataSchema)
    # Width and height should have been clamped to 0 by validator
    assert result.image_metadata.width == 0
    assert result.image_metadata.height == 0
    assert result.text_metadata is None


def test_valid_structured_metadata_auto_nulls_others():
    data = {
        "content_metadata": {"type": "structured"},
        "table_metadata": {"table_format": "html"},
        "chart_metadata": {"table_format": "html"},  # ✅ Fixed to a valid enum value
    }
    result = validate_metadata(data)
    assert result.content_metadata.type == ContentTypeEnum.STRUCTURED
    assert result.table_metadata.table_format == "html"
    assert result.chart_metadata.table_format == "html"


def test_valid_audio_metadata_auto_nulls_others():
    data = {
        "content_metadata": {"type": "audio"},
        "audio_metadata": {"audio_transcript": "sample audio"},
        "text_metadata": {"text_type": "plain"},
    }
    result = validate_metadata(data)
    assert result.content_metadata.type == ContentTypeEnum.AUDIO
    assert isinstance(result.audio_metadata, AudioMetadataSchema)
    assert result.text_metadata is None


def test_invalid_source_metadata_dates_raise_error():
    bad_date = "not-a-date"
    data = {
        "content_metadata": {"type": "text"},
        "source_metadata": {
            "source_name": "source1",
            "source_id": "id1",
            "source_type": "pdf",
            "date_created": bad_date,
            "last_modified": bad_date,
        },
    }
    with pytest.raises(ValueError) as excinfo:
        validate_metadata(data)
    assert "Invalid ISO 8601 date" in str(excinfo.value) or "Invalid isoformat" in str(excinfo.value)


def test_validate_image_type_invalid_type_raises():
    with pytest.raises(ValidationError) as excinfo:
        ImageMetadataSchema(image_type=123)
    message = str(excinfo.value)
    assert "Input should be 'bmp'" in message  # Part of enum validation message
    assert "Input should be a valid string" in message


def test_text_metadata_accepts_keywords_flexibly():
    schema = MetadataSchema(
        content_metadata={"type": "text"},
        text_metadata={
            "text_type": TextTypeEnum.BODY,  # ✅ Use valid enum
            "keywords": ["keyword1", "keyword2"],
        },
    )
    assert schema.text_metadata.text_type == TextTypeEnum.BODY
    assert schema.text_metadata.keywords == ["keyword1", "keyword2"]


def test_table_metadata_requires_format():
    with pytest.raises(ValidationError):
        TableMetadataSchema()  # Missing required 'table_format'


def test_chart_metadata_requires_format():
    with pytest.raises(ValidationError):
        TableMetadataSchema()  # Missing required 'table_format'


def test_source_metadata_defaults_work():
    schema = SourceMetadataSchema(source_name="test", source_id="id", source_type="pdf")
    # Should have default ISO format dates
    datetools.validate_iso8601(schema.date_created)
    datetools.validate_iso8601(schema.last_modified)
    assert schema.access_level == schema.access_level.__class__.UNKNOWN
