# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioConfigSchema, AudioExtractorSchema
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging


### Tests for AudioConfigSchema ###


def test_valid_grpc_only_endpoint():
    config = AudioConfigSchema(audio_endpoints=("grpc_service", None))
    assert config.audio_endpoints == ("grpc_service", None)
    assert config.audio_infer_protocol == "grpc"


def test_valid_http_only_endpoint():
    config = AudioConfigSchema(audio_endpoints=(None, "http_service"))
    assert config.audio_endpoints == (None, "http_service")
    assert config.audio_infer_protocol == "http"


def test_valid_both_endpoints():
    config = AudioConfigSchema(audio_endpoints=("grpc_service", "http_service"), audio_infer_protocol="grpc")
    assert config.audio_endpoints == ("grpc_service", "http_service")
    assert config.audio_infer_protocol == "grpc"


def test_invalid_both_endpoints_empty():
    with pytest.raises(ValidationError) as excinfo:
        AudioConfigSchema(audio_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty" in str(excinfo.value)


def test_clean_service_empty_string_spaces_quotes():
    with pytest.raises(ValidationError) as excinfo:
        AudioConfigSchema(audio_endpoints=("   ", '   "  '))
    assert "Both gRPC and HTTP services cannot be empty" in str(excinfo.value)


def test_default_protocol_resolution_when_not_provided():
    config = AudioConfigSchema(audio_endpoints=("grpc_service", None))
    assert config.audio_infer_protocol == "grpc"

    config = AudioConfigSchema(audio_endpoints=(None, "http_service"))
    assert config.audio_infer_protocol == "http"


def test_protocol_normalized_to_lowercase():
    config = AudioConfigSchema(audio_endpoints=("grpc_service", "http_service"), audio_infer_protocol="GRPC")
    assert config.audio_infer_protocol == "grpc"


def test_extra_fields_forbidden_in_audio_config():
    with pytest.raises(ValidationError):
        AudioConfigSchema(audio_endpoints=("grpc_service", "http_service"), extra_field="oops")


### Tests for AudioExtractorSchema ###


def test_valid_audio_extractor_defaults():
    schema = AudioExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 16
    assert schema.raise_on_failure is False
    assert schema.audio_extraction_config is None


def test_audio_extractor_with_custom_config():
    audio_config = AudioConfigSchema(audio_endpoints=("grpc_service", None))
    schema = AudioExtractorSchema(
        max_queue_size=10, n_workers=4, raise_on_failure=True, audio_extraction_config=audio_config
    )
    assert schema.max_queue_size == 10
    assert schema.n_workers == 4
    assert schema.raise_on_failure is True
    assert schema.audio_extraction_config.audio_endpoints == ("grpc_service", None)


def test_audio_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        AudioExtractorSchema(max_queue_size=10, n_workers=4, invalid_field="should_fail")


def test_audio_extractor_invalid_types():
    with pytest.raises(ValidationError):
        AudioExtractorSchema(max_queue_size="not_an_int", n_workers="also_wrong")


### Sanitization and Redaction Tests ###


def test_audio_config_repr_hides_sensitive_fields_and_sanitize_redacts():
    config = AudioConfigSchema(
        audio_endpoints=("grpc_service", None),
        auth_token="super_secret_token",
        ssl_cert="-----BEGIN CERTIFICATE-----\nABC\n-----END CERTIFICATE-----",
    )

    # repr/str should not include sensitive values because repr=False on fields
    rep = repr(config)
    s = str(config)
    assert "super_secret_token" not in rep
    assert "super_secret_token" not in s
    assert "BEGIN CERTIFICATE" not in rep
    assert "BEGIN CERTIFICATE" not in s

    # sanitize_for_logging should redact sensitive keys
    sanitized = sanitize_for_logging(config)
    assert isinstance(sanitized, dict)
    assert sanitized.get("auth_token") == "***REDACTED***"
    assert sanitized.get("ssl_cert") == "***REDACTED***"


def test_audio_extractor_schema_sanitize_nested_config():
    audio_config = AudioConfigSchema(
        audio_endpoints=("grpc_service", None),
        auth_token="top_secret",
        ssl_cert="CERTDATA",
    )
    schema = AudioExtractorSchema(
        max_queue_size=2,
        n_workers=1,
        raise_on_failure=False,
        audio_extraction_config=audio_config,
    )

    sanitized = sanitize_for_logging(schema)
    assert isinstance(sanitized, dict)
    assert sanitized["audio_extraction_config"]["auth_token"] == "***REDACTED***"
    assert sanitized["audio_extraction_config"]["ssl_cert"] == "***REDACTED***"
