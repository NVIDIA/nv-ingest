import subprocess
import sys
import time

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient


def test_images_extract_only(
    pipeline_process,
):
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.wav")
        .extract(
            extract_method="audio",
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1

    transcript = results[0][0]["metadata"]["audio_metadata"]["audio_transcript"]
    expected = (
        "Section one, this is the first section of the document. "
        "It has some more placeholder text to show how the document looks like. "
        "The text is not meant to be meaningful or informative, "
        "but rather to demonstrate the layout and formatting of the document."
    )
    assert transcript == expected
