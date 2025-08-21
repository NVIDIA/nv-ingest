import pytest
from nv_ingest_client.primitives.tasks.audio_extraction import AudioExtractionTask


@pytest.fixture
def sample_task():
    return AudioExtractionTask(
        auth_token="test_token",
        grpc_endpoint="localhost:50051",
        http_endpoint="http://localhost:8000",
        infer_protocol="grpc",
        use_ssl=True,
        ssl_cert="test_cert",
        segment_audio=True,
    )


def test_audio_extraction_task_init(sample_task):
    """Test initialization of AudioExtractionTask."""
    assert sample_task._auth_token == "test_token"
    assert sample_task._grpc_endpoint == "localhost:50051"
    assert sample_task._http_endpoint == "http://localhost:8000"
    assert sample_task._infer_protocol == "grpc"
    assert sample_task._use_ssl is True
    assert sample_task._ssl_cert == "test_cert"
    assert sample_task._segment_audio is True


def test_audio_extraction_task_str(sample_task):
    """Test string representation of AudioExtractionTask."""
    task_str = str(sample_task)
    assert "Audio Extraction Task:" in task_str
    assert "auth_token: [redacted]" in task_str
    assert "grpc_endpoint: localhost:50051" in task_str
    assert "http_endpoint: http://localhost:8000" in task_str
    assert "infer_protocol: grpc" in task_str
    assert "use_ssl: True" in task_str
    assert "ssl_cert: [redacted]" in task_str
    assert "segment_audio: True" in task_str


def test_audio_extraction_task_to_dict(sample_task):
    """Test conversion of AudioExtractionTask to dictionary."""
    expected_dict = {
        "type": "audio_data_extract",
        "task_properties": {
            "auth_token": "test_token",
            "grpc_endpoint": "localhost:50051",
            "http_endpoint": "http://localhost:8000",
            "infer_protocol": "grpc",
            "use_ssl": True,
            "ssl_cert": "test_cert",
            "segment_audio": True,
        },
    }
    assert sample_task.to_dict() == expected_dict


def test_audio_extraction_task_empty():
    """Test AudioExtractionTask with default arguments."""
    task = AudioExtractionTask()
    assert task._auth_token is None
    assert task._grpc_endpoint is None
    assert task._http_endpoint is None
    assert task._infer_protocol is None
    assert task._use_ssl is None
    assert task._ssl_cert is None
    assert task._segment_audio is None


# Schema Consolidation Tests


def test_audio_extraction_task_schema_consolidation():
    """Test that AudioExtractionTask uses API schema for validation."""
    # Test that valid parameters work
    task = AudioExtractionTask(
        auth_token="test_token",
        grpc_endpoint="localhost:50051",
        http_endpoint="http://localhost:8000",
        infer_protocol="grpc",
        function_id="test_function",
        use_ssl=True,
        ssl_cert="test_cert",
        segment_audio=True,
    )

    assert task._auth_token == "test_token"
    assert task._grpc_endpoint == "localhost:50051"
    assert task._http_endpoint == "http://localhost:8000"
    assert task._infer_protocol == "grpc"
    assert task._function_id == "test_function"
    assert task._use_ssl is True
    assert task._ssl_cert == "test_cert"
    assert task._segment_audio is True


def test_audio_extraction_task_api_schema_validation():
    """Test that AudioExtractionTask validates against API schema constraints."""
    # Test that None values are handled correctly
    task = AudioExtractionTask()

    assert task._auth_token is None
    assert task._grpc_endpoint is None
    assert task._http_endpoint is None
    assert task._infer_protocol is None
    assert task._function_id is None
    assert task._use_ssl is None
    assert task._ssl_cert is None
    assert task._segment_audio is None


def test_audio_extraction_task_serialization_with_api_schema():
    """Test AudioExtractionTask serialization works correctly with API schema."""
    task = AudioExtractionTask(
        auth_token="test_token",
        grpc_endpoint="localhost:50051",
        http_endpoint="http://localhost:8000",
        infer_protocol="grpc",
    )

    task_dict = task.to_dict()

    assert task_dict["type"] == "audio_data_extract"
    assert task_dict["task_properties"]["auth_token"] == "test_token"
    assert task_dict["task_properties"]["grpc_endpoint"] == "localhost:50051"
    assert task_dict["task_properties"]["http_endpoint"] == "http://localhost:8000"
    assert task_dict["task_properties"]["infer_protocol"] == "grpc"
