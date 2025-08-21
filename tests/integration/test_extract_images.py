import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient


@pytest.mark.integration
@pytest.mark.parametrize(
    "image_file",
    [
        "./data/multimodal_test.bmp",
        "./data/multimodal_test.jpeg",
        "./data/multimodal_test.png",
        "./data/multimodal_test.tiff",
    ],
)
def test_images_extract_only(
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_second_table_markdown,
    multimodal_first_chart_xaxis,
    multimodal_first_chart_yaxis,
    multimodal_second_chart_xaxis,
    multimodal_second_chart_yaxis,
    expected_table_bmp_full_markdown_variants,
    expected_table_jpeg_full_markdown_variants,
    expected_table_png_full_markdown_variants,
    expected_table_tiff_full_markdown_variants,
    multimodal_first_chart_xaxis_variants_images,
    image_file,
):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files(image_file)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            table_output_format="markdown",
            extract_infographics=True,
            text_depth="page",
        )
    )

    results = ingestor.ingest()

    assert len(results) == 1

    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    assert len(texts) == 0  # text extraction is not supported for raw images.

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) == 2

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) == 1
    assert len(charts) == 1
    assert len(infographics) == 0

    # Assert exact table content per format based on observed outputs
    assert len(tables) == 1
    extracted_table = tables[0]["metadata"]["table_metadata"]["table_content"]
    if image_file.endswith(".bmp"):
        expected_variants = expected_table_bmp_full_markdown_variants
    elif image_file.endswith(".jpeg"):
        expected_variants = expected_table_jpeg_full_markdown_variants
    elif image_file.endswith(".png"):
        expected_variants = expected_table_png_full_markdown_variants
    elif image_file.endswith(".tiff"):
        expected_variants = expected_table_tiff_full_markdown_variants
    else:
        raise AssertionError(f"Unhandled image format for test: {image_file}")
    assert extracted_table in expected_variants

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    assert any(v in chart_contents for v in multimodal_first_chart_xaxis_variants_images)
    assert multimodal_first_chart_yaxis in chart_contents
