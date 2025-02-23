import uuid

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest_api.util.pdf import construct_text_metadata


def test_construct_text_metadata_basic():
    accumulated_text = ["This is the first page.", "This is another page."]
    keywords = ["test", "metadata"]
    page_idx = 1
    block_idx = -1
    line_idx = -1
    span_idx = -1
    page_count = 10
    text_depth = "document"
    source_metadata = {"source_name": "test_source", "source_id": "test_source_id", "source_type": "PDF"}
    base_unified_metadata = {}

    result = construct_text_metadata(
        accumulated_text,
        keywords,
        page_idx,
        block_idx,
        line_idx,
        span_idx,
        page_count,
        text_depth,
        source_metadata,
        base_unified_metadata,
    )

    extracted_text = " ".join(accumulated_text)

    assert result[0] == ContentTypeEnum.TEXT
    assert result[1]["content"] == extracted_text
    for key, val in source_metadata.items():
        assert result[1]["source_metadata"][key] == val
    for key in [
        "access_level",
        "collection_id",
        "date_created",
        "last_modified",
        "partition_id",
        "source_location",
        "summary",
    ]:
        assert key in result[1]["source_metadata"]
    assert result[1]["content_metadata"]["page_number"] == page_idx
    assert result[1]["content_metadata"]["hierarchy"]["page"] == page_idx
    assert result[1]["content_metadata"]["hierarchy"]["page_count"] == page_count
    assert result[1]["content_metadata"]["type"] == ContentTypeEnum.TEXT

    assert result[1]["text_metadata"]["text_type"] == text_depth
    assert result[1]["text_metadata"]["keywords"] == keywords
    assert result[1]["text_metadata"]["language"] == "en"
    assert result[1]["text_metadata"]["text_location"] == (-1, -1, -1, -1)

    uuid.UUID(result[2])  # This will raise an exception if it's not a valid UUID


def test_construct_text_metadata_empty_text():
    accumulated_text = []
    keywords = ["test", "metadata"]
    page_idx = 0
    block_idx = -1
    line_idx = -1
    span_idx = -1
    page_count = 10
    text_depth = "page"
    source_metadata = {"source_name": "test_source", "source_id": "test_source_id", "source_type": "PDF"}
    base_unified_metadata = {}

    result = construct_text_metadata(
        accumulated_text,
        keywords,
        page_idx,
        block_idx,
        line_idx,
        span_idx,
        page_count,
        text_depth,
        source_metadata,
        base_unified_metadata,
    )

    assert result[0] == ContentTypeEnum.TEXT
    assert result[1]["content"] == ""
    assert result[1]["content_metadata"]["page_number"] == page_idx
    assert result[1]["content_metadata"]["hierarchy"]["page"] == page_idx
    assert result[1]["content_metadata"]["hierarchy"]["page_count"] == page_count
    assert result[1]["content_metadata"]["type"] == ContentTypeEnum.TEXT

    assert result[1]["text_metadata"]["text_type"] == text_depth
    assert result[1]["text_metadata"]["keywords"] == keywords
    assert result[1]["text_metadata"]["language"] == "unknown"
