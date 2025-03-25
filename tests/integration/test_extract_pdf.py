import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient


def test_pdfium_extract_pdf(pipeline_process):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_infographics=True,
            text_depth="page",
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1

    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    assert len(texts) == 3

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) == 5

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) == 2
    assert len(charts) == 3
    assert len(infographics) == 0


def test_nemoretriever_parse_extract_pdf(pipeline_process):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_method="nemoretriever_parse",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            paddle_output_format="markdown",
            extract_infographics=True,
            text_depth="page",
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1

    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    assert len(texts) == 3

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) == 5

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]
    assert len(tables) == 2
    assert len(charts) == 3
    assert len(infographics) == 0
