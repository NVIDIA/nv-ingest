import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.util.milvus import nvingest_retrieval


def test_pdfium_extract_only(
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_second_table_markdown,
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

    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert multimodal_first_table_markdown in " ".join(table_contents)
    assert multimodal_second_table_markdown in " ".join(table_contents)

    chart_contents = [x["metadata"]["table_metadata"]["table_content"] for x in charts]
    assert multimodal_first_chart_yaxis in " ".join(chart_contents)
    assert multimodal_second_chart_xaxis in " ".join(chart_contents)
    assert multimodal_second_chart_yaxis in " ".join(chart_contents)


def test_nemoretriever_parse_extract_only(
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_second_table_markdown,
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

    table_contents = [x["metadata"]["table_metadata"]["table_content"] for x in tables]
    assert multimodal_first_table_markdown in " ".join(table_contents)
    assert multimodal_second_table_markdown in " ".join(table_contents)

    chart_contents = [x["metadata"]["table_metadata"]["table_content"] for x in charts]
    assert multimodal_first_chart_yaxis in " ".join(chart_contents)
    assert multimodal_second_chart_xaxis in " ".join(chart_contents)
    assert multimodal_second_chart_yaxis in " ".join(chart_contents)


def test_pdfium_extract_embed_upload_query(pipeline_process):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    milvus_uri = "milvus.db"
    collection_name = "test"
    sparse = False

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
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            dense_dim=2048,
            sparse=sparse,
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1

    queries = ["Which animal is responsible for the typos?"]

    retrieved_docs = nvingest_retrieval(
        queries,
        collection_name,
        milvus_uri=milvus_uri,
        hybrid=sparse,
        top_k=1,
    )
    extracted_content = retrieved_docs[0][0]["entity"]["text"]

    assert len(retrieved_docs) == 1
    assert "This table describes some animals" in extracted_content
