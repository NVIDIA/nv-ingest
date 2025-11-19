# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema, ImageExtractorSchema
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging


### Tests for ImageConfigSchema ###


def test_valid_yolox_grpc_only():
    config = ImageConfigSchema(yolox_endpoints=("grpc_service", None))
    assert config.yolox_endpoints == ("grpc_service", None)
    assert config.yolox_infer_protocol == "grpc"


def test_valid_yolox_http_only():
    config = ImageConfigSchema(yolox_endpoints=(None, "http_service"))
    assert config.yolox_endpoints == (None, "http_service")
    assert config.yolox_infer_protocol == "http"


def test_valid_yolox_both():
    config = ImageConfigSchema(yolox_endpoints=("grpc_service", "http_service"))
    assert config.yolox_endpoints == ("grpc_service", "http_service")
    assert config.yolox_infer_protocol == "http"  # defaults to http if both are provided


def test_invalid_yolox_empty_both():
    with pytest.raises(ValidationError) as excinfo:
        ImageConfigSchema(yolox_endpoints=(None, None))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_cleaning_yolox_endpoints_spaces_and_quotes():
    with pytest.raises(ValidationError) as excinfo:
        ImageConfigSchema(yolox_endpoints=("  ", '  "  '))
    assert "Both gRPC and HTTP services cannot be empty for yolox_endpoints." in str(excinfo.value)


def test_custom_protocol_is_normalized():
    config = ImageConfigSchema(yolox_endpoints=("grpc_service", "http_service"), yolox_infer_protocol="GRPC")
    assert config.yolox_infer_protocol == "grpc"


def test_extra_fields_forbidden_in_image_config():
    with pytest.raises(ValidationError):
        ImageConfigSchema(yolox_endpoints=("grpc_service", None), extra_field="fail")


### Tests for ImageExtractorSchema ###


def test_image_extractor_schema_defaults():
    schema = ImageExtractorSchema()
    assert schema.max_queue_size == 1
    assert schema.n_workers == 16
    assert schema.raise_on_failure is False
    assert schema.image_extraction_config is None


def test_image_extractor_with_config():
    image_config = ImageConfigSchema(yolox_endpoints=("grpc_service", None))
    schema = ImageExtractorSchema(
        max_queue_size=10, n_workers=5, raise_on_failure=True, image_extraction_config=image_config
    )
    assert schema.max_queue_size == 10
    assert schema.n_workers == 5
    assert schema.raise_on_failure is True
    assert schema.image_extraction_config == image_config


def test_image_extractor_forbid_extra_fields():
    with pytest.raises(ValidationError):
        ImageExtractorSchema(max_queue_size=10, n_workers=5, invalid_field="fail")


def test_image_extractor_invalid_types():
    with pytest.raises(ValidationError):
        ImageExtractorSchema(max_queue_size="not_int", n_workers="also_not_int")


### Sanitization and Redaction Tests ###


def test_image_config_repr_hides_sensitive_fields_and_sanitize_redacts():
    config = ImageConfigSchema(
        yolox_endpoints=("grpc_service", None),
        auth_token="very_secret",
    )

    rep = repr(config)
    s = str(config)
    assert "very_secret" not in rep
    assert "very_secret" not in s

    sanitized = sanitize_for_logging(config)
    assert isinstance(sanitized, dict)
    assert sanitized.get("auth_token") == "***REDACTED***"


def test_image_extractor_schema_sanitize_nested_config():
    image_config = ImageConfigSchema(
        yolox_endpoints=("grpc_service", None),
        auth_token="nested_secret",
    )
    schema = ImageExtractorSchema(
        max_queue_size=3,
        n_workers=2,
        raise_on_failure=False,
        image_extraction_config=image_config,
    )

    sanitized = sanitize_for_logging(schema)
    assert isinstance(sanitized, dict)
    assert sanitized["image_extraction_config"]["auth_token"] == "***REDACTED***"
