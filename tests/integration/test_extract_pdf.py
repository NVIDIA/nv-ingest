import sys
import time

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import start_pipeline_subprocess


def test_pdfium_extract_pdf():
    config = PipelineCreationSchema()
    pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)
    time.sleep(10)

    assert pipeline_process.poll() is None, "Error running pipeline subprocess."

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
    assert len(results[0]) == 3


def test_nemoretriever_parse_extract_pdf():
    config = PipelineCreationSchema()
    pipeline_process = start_pipeline_subprocess(config, stderr=sys.stderr, stdout=sys.stdout)
    time.sleep(10)

    assert pipeline_process.poll() is None, "Error running pipeline subprocess."

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
    assert len(results[0]) == 3
