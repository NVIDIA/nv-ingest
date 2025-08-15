import subprocess
import sys
import time
import os
import socket

import pytest
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient


def _parse_host_port(endpoint: str) -> tuple[str, int] | tuple[None, None]:
    """
    Parse an endpoint string like "host:port" into (host, port).
    Returns (None, None) if parsing fails.
    """
    try:
        if not endpoint:
            return None, None
        # Simple split on last ':' to support hostnames with colons minimally
        if ":" not in endpoint:
            return endpoint, 0
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    except Exception:
        return None, None


def _wait_for_service(host: str, port: int, timeout: float = 3.0) -> bool:
    """Return True if TCP connect to host:port succeeds within timeout."""
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@pytest.mark.integration
def test_audio_extract_only(
    pipeline_process,
):
    # Skip test if audio extraction deps aren't installed (librosa required)
    if os.environ.get("INSTALL_AUDIO_EXTRACTION_DEPS", "").lower() not in ("1", "true", "yes"):
        pytest.skip(
            "Audio extraction dependencies not installed. Set INSTALL_AUDIO_EXTRACTION_DEPS=true to enable this test."
        )

    # Determine audio endpoint and skip if not reachable
    audio_grpc_endpoint = os.environ.get(
        "AUDIO_GRPC_ENDPOINT", "localhost:8021"
    )  # This is from the bare metal perspective
    host, port = _parse_host_port(audio_grpc_endpoint)
    if not _wait_for_service(host, port, timeout=3.0):
        pytest.skip(
            f"Audio service not reachable at {audio_grpc_endpoint}. "
            "Set AUDIO_GRPC_ENDPOINT to a reachable Riva ASR endpoint or start the audio NIM container."
        )

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = Ingestor(client=client).files("./data/multimodal_test.wav").extract()

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
