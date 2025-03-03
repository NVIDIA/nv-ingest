import pytest
from nv_ingest_client.primitives.tasks.audio_extraction import AudioExtractionTask


@pytest.fixture
def sample_task():
    return AudioExtractionTask(
        auth_token="test_token",
        grpc_endpoint="localhost:50051",
        infer_protocol="grpc",
        use_ssl=True,
        ssl_cert="test_cert",
    )


def test_audio_extraction_task_init(sample_task):
    """Test initialization of AudioExtractionTask."""
    assert sample_task._auth_token == "test_token"
    assert sample_task._grpc_endpoint == "localhost:50051"
    assert sample_task._infer_protocol == "grpc"
    assert sample_task._use_ssl is True
    assert sample_task._ssl_cert == "test_cert"


def test_audio_extraction_task_str(sample_task):
    """Test string representation of AudioExtractionTask."""
    task_str = str(sample_task)
    assert "Audio Extraction Task:" in task_str
    assert "auth_token: [redacted]" in task_str
    assert "grpc_endpoint: localhost:50051" in task_str
    assert "infer_protocol: grpc" in task_str
    assert "use_ssl: True" in task_str
    assert "ssl_cert: [redacted]" in task_str


def test_audio_extraction_task_to_dict(sample_task):
    """Test conversion of AudioExtractionTask to dictionary."""
    expected_dict = {
        "type": "audio_data_extract",
        "task_properties": {
            "auth_token": "test_token",
            "grpc_endpoint": "localhost:50051",
            "infer_protocol": "grpc",
            "use_ssl": True,
            "ssl_cert": "test_cert",
        },
    }
    assert sample_task.to_dict() == expected_dict


def test_audio_extraction_task_empty():
    """Test AudioExtractionTask with default arguments."""
    task = AudioExtractionTask()
    expected_dict = {
        "type": "audio_data_extract",
        "task_properties": {},
    }
    assert task.to_dict() == expected_dict
    assert "Audio Extraction Task:" in str(task)
