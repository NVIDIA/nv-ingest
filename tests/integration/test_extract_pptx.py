import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from tests.integration.utilities_for_test import canonicalize_markdown_table


@pytest.mark.integration
def test_pptx_extract_only(
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_first_chart_xaxis,
    multimodal_first_chart_yaxis,
    multimodal_second_chart_xaxis,
    multimodal_second_chart_yaxis,
):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pptx")
        .extract(
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

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) >= 3

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) >= 1
    assert len(charts) >= 2
    assert len(infographics) == 0

    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert any(canonicalize_markdown_table(x) == multimodal_first_table_markdown for x in table_contents)

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    assert multimodal_first_chart_xaxis in chart_contents
    assert multimodal_first_chart_yaxis in chart_contents
    assert multimodal_second_chart_xaxis in chart_contents
    assert multimodal_second_chart_yaxis.replace("1 - ", "") in chart_contents
