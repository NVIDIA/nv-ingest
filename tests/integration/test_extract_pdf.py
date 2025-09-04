import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.util.milvus import nvingest_retrieval


def create_client():
    return NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )


@pytest.mark.integration
def run_basic_extract_test(
    extract_method=None,
    expected_text_count=3,
    expected_table_count=2,
    expected_chart_count=3,
    expected_infographic_count=0,
    table_markdowns=None,
    chart_phrases=None,
):
    client = create_client()

    extract_kwargs = {
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_images": True,
        "table_output_format": "markdown",
        "extract_infographics": True,
        "text_depth": "page",
    }

    if extract_method:
        extract_kwargs["extract_method"] = extract_method

    ingestor = Ingestor(client=client).files("./data/multimodal_test.pdf").extract(**extract_kwargs)

    results = ingestor.ingest()
    assert len(results) == 1

    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    assert len(texts) == expected_text_count

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) == expected_table_count + expected_chart_count + expected_infographic_count

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]

    assert len(tables) == expected_table_count
    assert len(charts) == expected_chart_count
    assert len(infographics) == expected_infographic_count

    table_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in tables)
    for markdown in table_markdowns or []:
        assert markdown in table_contents

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    for phrase in chart_phrases or []:
        alt_phrase = phrase.replace("Bluetooth speaker", "Bluetoothspeaker")
        assert (phrase in chart_contents) or (alt_phrase in chart_contents)


@pytest.mark.integration
@pytest.mark.parametrize("extract_method", [None])  # , "nemoretriever_parse"])
def test_extract_only(
    extract_method,
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_second_table_markdown,
    multimodal_first_chart_xaxis,
    multimodal_first_chart_yaxis,
    multimodal_second_chart_xaxis,
    multimodal_second_chart_yaxis,
):
    run_basic_extract_test(
        extract_method=extract_method,
        table_markdowns=[
            multimodal_first_table_markdown,
            multimodal_second_table_markdown,
        ],
        chart_phrases=[
            multimodal_first_chart_xaxis,
            multimodal_first_chart_yaxis,
            multimodal_second_chart_xaxis,
            multimodal_second_chart_yaxis,
        ],
    )


@pytest.mark.integration
@pytest.mark.skip("Skipping for the moment")
def test_pdfium_extract_embed_upload_query(pipeline_process):
    client = create_client()

    milvus_uri = "milvus.db"
    collection_name = "test"

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            dense_dim=2048,
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1
    assert len([x for x in results[0] if x["metadata"]["embedding"]]) == 8

    queries = ["Which table shows some popular colors that cars might come in?"]

    retrieved_docs = nvingest_retrieval(
        queries,
        collection_name,
        milvus_uri=milvus_uri,
        top_k=10,
    )
    assert len(retrieved_docs[0]) == 8

    extracted_content = retrieved_docs[0][0]["entity"]["text"]
    assert "This table shows some popular colors that cars might come in" in extracted_content
