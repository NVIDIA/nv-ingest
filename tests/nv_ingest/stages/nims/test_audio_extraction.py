from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest

from nv_ingest.schemas.audio_extractor_schema import AudioConfigSchema
from nv_ingest.stages.nim.audio_extraction import _transcribe_audio
from nv_ingest.stages.nim.audio_extraction import _update_metadata
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.nim.parakeet import create_audio_inference_client


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "metadata": {
                    "content": "base64_audio_data",
                    "content_metadata": {"type": "audio"},
                }
            }
        ]
    )


@pytest.fixture
def mock_audio_client():
    """Fixture to provide a mock audio inference client."""
    client = Mock()
    client.infer.return_value = "Transcribed audio text"
    return client


def test_update_metadata_valid_audio(mock_audio_client):
    """Test _update_metadata correctly processes an audio row."""
    row = pd.Series(
        {
            "metadata": {
                "content": "base64_audio_data",
                "content_metadata": {"type": "audio"},
            }
        }
    )
    trace_info = {}

    result_metadata = _update_metadata(row, mock_audio_client, trace_info)

    assert "audio_metadata" in result_metadata
    assert result_metadata["audio_metadata"]["audio_transcript"] == "Transcribed audio text"


def test_update_metadata_missing_metadata():
    """Test _update_metadata raises an error if metadata is missing."""
    row = pd.Series({})
    mock_audio_client = Mock()
    trace_info = {}

    with pytest.raises(ValueError, match="Row does not contain 'metadata'"):
        _update_metadata(row, mock_audio_client, trace_info)


def test_update_metadata_non_audio_content():
    """Test _update_metadata does not modify metadata for non-audio content."""
    row = pd.Series(
        {
            "metadata": {
                "content": None,
                "content_metadata": {"type": "text"},
            }
        }
    )
    mock_audio_client = Mock()
    trace_info = {}

    result_metadata = _update_metadata(row, mock_audio_client, trace_info)
    assert "audio_metadata" not in result_metadata


@pytest.fixture
def mock_validated_config():
    """Fixture to provide a mock validated config."""
    return Mock(
        audio_extraction_config=AudioConfigSchema(
            audio_endpoints=("grpc://test", None),
            audio_infer_protocol="grpc",
            auth_token="test_token",
            use_ssl=True,
        )
    )
