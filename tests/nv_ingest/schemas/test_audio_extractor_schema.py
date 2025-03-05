import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioConfigSchema, AudioExtractorSchema


def test_audio_config_schema_valid_grpc():
    """Test AudioConfigSchema with a valid gRPC endpoint."""
    config = AudioConfigSchema(audio_endpoints=("grpc://localhost:50051", ""))
    assert config.audio_endpoints == ("grpc://localhost:50051", None)
    assert config.audio_infer_protocol == "grpc"


def test_audio_config_schema_valid_http():
    """Test AudioConfigSchema with a valid HTTP endpoint."""
    config = AudioConfigSchema(audio_endpoints=("", "http://localhost:8080"))
    assert config.audio_endpoints == (None, "http://localhost:8080")
    assert config.audio_infer_protocol == "http"


def test_audio_config_schema_valid_both_endpoints():
    """Test AudioConfigSchema with both gRPC and HTTP endpoints provided."""
    config = AudioConfigSchema(audio_endpoints=("grpc://localhost:50051", "http://localhost:8080"))
    assert config.audio_endpoints == ("grpc://localhost:50051", "http://localhost:8080")
    assert config.audio_infer_protocol == "http"  # Defaults to HTTP when both exist


def test_audio_config_schema_empty_endpoints():
    """Test AudioConfigSchema validation error when both gRPC and HTTP endpoints are empty."""
    with pytest.raises(ValidationError, match="Both gRPC and HTTP services cannot be empty"):
        AudioConfigSchema(audio_endpoints=("", ""))


def test_audio_config_schema_whitespace_endpoints():
    """Test AudioConfigSchema removes whitespace-only values."""
    config = AudioConfigSchema(audio_endpoints=("   ", "http://localhost:8080"))
    assert config.audio_endpoints == (None, "http://localhost:8080")
    assert config.audio_infer_protocol == "http"


def test_audio_config_schema_ssl_enabled():
    """Test AudioConfigSchema with SSL enabled."""
    config = AudioConfigSchema(audio_endpoints=("grpc://localhost:50051", ""), use_ssl=True, ssl_cert="cert.pem")
    assert config.use_ssl is True
    assert config.ssl_cert == "cert.pem"


def test_audio_config_schema_default_protocol():
    """Test that the default protocol is inferred correctly."""
    config = AudioConfigSchema(audio_endpoints=("grpc://localhost:50051", None))
    assert config.audio_infer_protocol == "grpc"

    config = AudioConfigSchema(audio_endpoints=(None, "http://localhost:8080"))
    assert config.audio_infer_protocol == "http"


def test_audio_extractor_schema_defaults():
    """Test default values for AudioExtractorSchema."""
    extractor = AudioExtractorSchema()
    assert extractor.max_queue_size == 1
    assert extractor.n_workers == 16
    assert extractor.raise_on_failure is False
    assert extractor.audio_extraction_config is None


def test_audio_extractor_schema_with_audio_config():
    """Test AudioExtractorSchema with an embedded AudioConfigSchema."""
    audio_config = AudioConfigSchema(audio_endpoints=("grpc://localhost:50051", ""))
    extractor = AudioExtractorSchema(audio_extraction_config=audio_config)
    assert extractor.audio_extraction_config.audio_endpoints == ("grpc://localhost:50051", None)
    assert extractor.audio_extraction_config.audio_infer_protocol == "grpc"


def test_audio_extractor_schema_invalid_extra_field():
    """Test that AudioExtractorSchema forbids extra fields."""
    with pytest.raises(ValidationError):
        AudioExtractorSchema(max_queue_size=5, unknown_field="invalid")
