from datetime import datetime

import pytest
from pydantic import ValidationError

from nv_ingest.schemas.metadata_schema import ChartMetadataSchema  # Adjust the import as per your file structure
from nv_ingest.schemas.metadata_schema import ContentHierarchySchema
from nv_ingest.schemas.metadata_schema import ContentMetadataSchema
from nv_ingest.schemas.metadata_schema import ErrorMetadataSchema
from nv_ingest.schemas.metadata_schema import ImageMetadataSchema
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import NearbyObjectsSchema
from nv_ingest.schemas.metadata_schema import SourceMetadataSchema
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TableMetadataSchema
from nv_ingest.schemas.metadata_schema import TextMetadataSchema


# Test cases for SourceMetadataSchema
def test_source_metadata_schema_defaults():
    config = SourceMetadataSchema(source_name="Test Source", source_id="1234", source_type="TestType")
    assert config.source_location == ""
    assert config.collection_id == ""
    assert config.partition_id == -1
    assert config.access_level == -1


def test_source_metadata_schema_invalid_date():
    with pytest.raises(ValidationError):
        SourceMetadataSchema(
            source_name="Test Source", source_id="1234", source_type="TestType", date_created="invalid_date"
        )


# Test cases for NearbyObjectsSchema
def test_nearby_objects_schema_defaults():
    config = NearbyObjectsSchema()
    assert config.text.content == []
    assert config.images.content == []
    assert config.structured.content == []


# Test cases for ContentHierarchySchema
def test_content_hierarchy_schema_defaults():
    config = ContentHierarchySchema()
    assert config.page_count == -1
    assert config.page == -1
    assert config.block == -1
    assert config.line == -1
    assert config.span == -1


def test_content_hierarchy_schema_with_nearby_objects():
    config = ContentHierarchySchema(
        nearby_objects=NearbyObjectsSchema(text={"content": ["sample text"]}, images={"content": ["sample image"]})
    )
    assert config.nearby_objects.text.content == ["sample text"]
    assert config.nearby_objects.images.content == ["sample image"]


# Test cases for ContentMetadataSchema
def test_content_metadata_schema_defaults():
    config = ContentMetadataSchema(type="text")
    print(config)
    assert config.description == ""
    assert config.page_number == -1


def test_content_metadata_schema_invalid_type():
    with pytest.raises(ValidationError):
        ContentMetadataSchema(type="InvalidType")


# Test cases for TextMetadataSchema
def test_text_metadata_schema_defaults():
    config = TextMetadataSchema(text_type="document")
    assert config.summary == ""
    assert config.keywords == ""
    assert config.language == "en"
    assert config.text_location == (0, 0, 0, 0)


def test_text_metadata_schema_with_keywords():
    config = TextMetadataSchema(text_type="body", keywords=["keyword1", "keyword2"])
    assert config.keywords == ["keyword1", "keyword2"]


# Test cases for ImageMetadataSchema
def test_image_metadata_schema_defaults():
    config = ImageMetadataSchema(image_type="image")
    assert config.caption == ""
    assert config.width == 0
    assert config.height == 0


def test_image_metadata_schema_invalid_type():
    with pytest.raises(ValidationError):
        ImageMetadataSchema(image_type=3.14)  # Using a float value


# Test cases for TableMetadataSchema
@pytest.mark.parametrize("table_format", ["html", "markdown", "latex", "image"])
def test_table_metadata_schema_defaults(table_format):
    config = TableMetadataSchema(table_format=table_format)
    assert config.caption == ""
    assert config.table_content == ""


def test_table_metadata_schema_with_location():
    config = TableMetadataSchema(table_format="latex", table_location=(1, 2, 3, 4))
    assert config.table_location == (1, 2, 3, 4)


@pytest.mark.parametrize("schema_class", [TableMetadataSchema, ChartMetadataSchema])
@pytest.mark.parametrize(
    "table_format", [TableFormatEnum.HTML, TableFormatEnum.MARKDOWN, TableFormatEnum.LATEX, TableFormatEnum.IMAGE]
)
def test_schema_valid_table_format(schema_class, table_format):
    config = schema_class(table_format=table_format)
    assert config.caption == ""
    assert config.table_content == ""


def test_table_metadata_schema_invalid_table_format():
    with pytest.raises(ValidationError):
        TableMetadataSchema(table_format="invalid_format")


# Test cases for ChartMetadataSchema
def test_chart_metadata_schema_defaults():
    config = ChartMetadataSchema(table_format="html")
    assert config.caption == ""
    assert config.table_content == ""


# Test cases for ErrorMetadataSchema
def test_error_metadata_schema_defaults():
    config = ErrorMetadataSchema(task="embed", status="error", error_msg="An error occurred.")
    assert config.source_id == ""


def test_error_metadata_schema_invalid_status():
    with pytest.raises(ValidationError):
        ErrorMetadataSchema(task="TaskType1", status="InvalidStatus", error_msg="An error occurred.")


# Test cases for InfoMessageMetadataSchema
def test_info_message_metadata_schema_defaults():
    config = InfoMessageMetadataSchema(
        task="transform", status="success", message="This is an info message.", filter=False
    )
    assert config.filter is False


def test_info_message_metadata_schema_invalid_task():
    with pytest.raises(ValidationError):
        InfoMessageMetadataSchema(task="InvalidTaskType", status="InfoStatus", message="This is an info message.")
