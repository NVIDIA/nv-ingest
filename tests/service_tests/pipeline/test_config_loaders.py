# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=line-too-long
# flake8: noqa

import os
import tempfile
import pytest
from unittest.mock import patch

from nv_ingest.pipeline.config.loaders import load_pipeline_config
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from pydantic import ValidationError


class TestLoadPipelineConfig:
    """Comprehensive blackbox tests for load_pipeline_config function."""

    def test_load_valid_config_file(self):
        """Test loading a valid pipeline configuration file."""
        config_content = """
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "source_stage"
    type: "source"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      broker_client:
        client_type: "redis"
        host: "localhost"
        port: 6379
      task_queue: "test_queue"
      poll_interval: 0.1
    replicas:
      cpu_count_min: 0
      cpu_count_max: 1

  - name: "test_stage"
    type: "stage"
    phase: 1
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param1: "value1"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2
    runs_after:
      - "source_stage"

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            try:
                config = load_pipeline_config(f.name)
                assert isinstance(config, PipelineConfigSchema)
                assert config.name == "Test Pipeline"
                assert config.description == "A test pipeline configuration"
                assert len(config.stages) == 2
                assert len(config.edges) == 1
                assert config.stages[0].name == "source_stage"
                assert config.stages[0].type == "source"
                assert config.stages[0].phase == 0
                assert config.stages[1].name == "test_stage"
                assert config.edges[0].from_stage == "source_stage"
                assert config.edges[0].to_stage == "test_stage"
                assert config.pipeline.disable_dynamic_scaling == False
            finally:
                os.unlink(f.name)

    def test_load_config_with_env_var_substitution(self):
        """Test loading config with environment variable substitution."""
        config_content = """
name: "Test Pipeline with Env Vars"
description: "A test pipeline with environment variable substitution"
stages:
  - name: "source_stage"
    type: "source"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      broker_client:
        client_type: "redis"
        host: $TEST_HOST|"localhost"
        port: $TEST_PORT|6379
      task_queue: $QUEUE_NAME|"default_queue"
      poll_interval: 0.1
    replicas:
      cpu_count_min: 0
      cpu_count_max: 1

  - name: "test_stage"
    type: "stage"
    phase: 1
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      auth_token: $TEST_TOKEN|""
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2
    runs_after:
      - "source_stage"

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            try:
                # Test with environment variables set
                with patch.dict(
                    os.environ, {"TEST_HOST": "prod-host", "TEST_PORT": "9000", "QUEUE_NAME": "prod_queue"}
                ):
                    config = load_pipeline_config(f.name)
                    assert config.stages[0].config["broker_client"]["host"] == "prod-host"
                    assert config.stages[0].config["broker_client"]["port"] == 9000  # YAML parses as int
                    assert config.stages[0].config["task_queue"] == "prod_queue"
                    assert config.stages[1].config["auth_token"] == ""

                # Test with defaults when env vars not set
                with patch.dict(os.environ, {}, clear=True):
                    config = load_pipeline_config(f.name)
                    assert config.stages[0].config["broker_client"]["host"] == "localhost"
                    assert config.stages[0].config["broker_client"]["port"] == 6379  # YAML parses as int
                    assert config.stages[0].config["task_queue"] == "default_queue"
            finally:
                os.unlink(f.name)

    def test_load_config_with_quoted_env_defaults(self):
        """Test loading config with quoted environment variable defaults."""
        config_content = """
name: "Test Pipeline with Quoted Defaults"
description: "A test pipeline with quoted environment variable defaults"
stages:
  - name: "source_stage"
    type: "source"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      broker_client:
        client_type: "redis"
        host: "localhost"
        port: 6379
      task_queue: "test_queue"
      poll_interval: 0.1
    replicas:
      cpu_count_min: 0
      cpu_count_max: 1

  - name: "test_stage"
    type: "stage"
    phase: 1
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      service_name: $SERVICE_NAME|"default-service"
      endpoint_url: $ENDPOINT_URL|'http://localhost:8000'
      json_config: $JSON_CONFIG|'{"key": "value"}'
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2
    runs_after:
      - "source_stage"

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            try:
                with patch.dict(os.environ, {}, clear=True):
                    config = load_pipeline_config(f.name)
                    assert config.stages[1].config["service_name"] == "default-service"
                    assert config.stages[1].config["endpoint_url"] == "http://localhost:8000"
                    assert config.stages[1].config["json_config"] == '{"key": "value"}'
            finally:
                os.unlink(f.name)

    def test_load_config_with_complex_structure(self):
        """Test loading config with complex nested structures."""
        config_content = """
name: "Test Pipeline with Complex Structure"
description: "A test pipeline with complex nested structures"
stages:
  - name: "source_stage"
    type: "source"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      broker_client:
        client_type: "redis"
        host: "localhost"
        port: 6379
      task_queue: "ingest_queue"
      poll_interval: 0.1
    replicas:
      cpu_count_min: 0
      cpu_count_max: 1

  - name: "pdf_extractor"
    type: "stage"
    phase: 1
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      pdfium_config:
        auth_token: $NGC_API_KEY|""
        endpoints: [
          $ENDPOINT1|"service1:8001",
          $ENDPOINT2|"service2:8002"
        ]
      nemoretriever_config:
        model_name: $MODEL_NAME|"default-model"
        timeout: $TIMEOUT|30
    replicas:
      cpu_count_min: 1
      cpu_count_max: 4
    runs_after:
      - "source_stage"

edges:
  - from: "source_stage"
    to: "pdf_extractor"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: $DISABLE_SCALING|false
  dynamic_memory_threshold: $MEMORY_THRESHOLD|0.75
  launch_simple_broker: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            try:
                with patch.dict(os.environ, {"NGC_API_KEY": "secret-key", "MODEL_NAME": "prod-model"}):
                    config = load_pipeline_config(f.name)
                    assert config.stages[1].config["pdfium_config"]["auth_token"] == "secret-key"
                    assert config.stages[1].config["nemoretriever_config"]["model_name"] == "prod-model"
                    assert config.stages[1].config["nemoretriever_config"]["timeout"] == 30  # YAML parses as int
                    assert config.pipeline.dynamic_memory_threshold == 0.75  # YAML parses as float
            finally:
                os.unlink(f.name)

    def test_load_config_file_not_found(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_pipeline_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml_syntax(self):
        """Test loading config with invalid YAML syntax."""
        invalid_yaml_content = """
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "test_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param1: "value1
      param2: [invalid, yaml, syntax
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml_content)
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    load_pipeline_config(f.name)
                assert "Failed to parse YAML after environment variable substitution" in str(exc_info.value)
                assert "--- Substituted Content ---" in str(exc_info.value)
            finally:
                os.unlink(f.name)

    def test_load_config_yaml_with_substitution_errors(self):
        """Test loading config where env var substitution creates invalid YAML."""
        config_content = """
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "test_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      endpoints: [$ENDPOINT1|"service1:8001", $ENDPOINT2|"service2:8002"]
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()

            try:
                # This should work (commas preserved in this case)
                with patch.dict(os.environ, {}, clear=True):
                    config = load_pipeline_config(f.name)
                    assert len(config.stages) == 1
            finally:
                os.unlink(f.name)

    def test_load_config_invalid_schema(self):
        """Test loading config with valid YAML but invalid schema."""
        invalid_schema_content = """
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "test_stage"
    type: "stage"
    # Missing required phase and actor
    config:
      param1: "value1"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_schema_content)
            f.flush()

            try:
                with pytest.raises(ValidationError):
                    load_pipeline_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_config_missing_required_fields(self):
        """Test loading config missing required fields."""
        missing_fields_content = """
name: "Test Pipeline"
# Missing description
stages:
  - name: "test_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param1: "value1"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

# Missing required edges
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(missing_fields_content)
            f.flush()

            try:
                with pytest.raises(ValidationError):
                    load_pipeline_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_config_invalid_data_types(self):
        """Test loading config with invalid data types."""
        invalid_types_content = """
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "test_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param1: "value1"
    replicas:
      cpu_count_min: "not_a_number"  # Should be int
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: "not_a_boolean"  # Should be bool
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_types_content)
            f.flush()

            try:
                with pytest.raises(ValidationError):
                    load_pipeline_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_config_empty_file(self):
        """Test loading an empty config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            try:
                with pytest.raises(TypeError):
                    load_pipeline_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_config_with_comments(self):
        """Test loading config with YAML comments."""
        config_with_comments = """
# Pipeline configuration for testing
name: "Test Pipeline"
description: "A test pipeline configuration"
stages:
  - name: "test_stage"  # This is a test stage
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param1: "value1"  # Test parameter
    replicas:
      cpu_count_min: 1  # Minimum CPU count
      cpu_count_max: 2  # Maximum CPU count

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false  # Dynamic scaling setting
  dynamic_memory_threshold: 0.8   # Memory threshold
  launch_simple_broker: false     # Broker launch setting
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_with_comments)
            f.flush()

            try:
                config = load_pipeline_config(f.name)
                assert isinstance(config, PipelineConfigSchema)
                assert config.name == "Test Pipeline"
                assert config.description == "A test pipeline configuration"
                assert len(config.stages) == 1
                assert config.stages[0].name == "test_stage"
                assert config.stages[0].type == "stage"
            finally:
                os.unlink(f.name)

    def test_load_config_with_multiple_stages(self):
        """Test loading config with multiple stages."""
        multi_stage_config = """
name: "Test Pipeline with Multiple Stages"
description: "A test pipeline with multiple stages"
stages:
  - name: "stage1"
    type: "source"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      param1: "value1"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

  - name: "stage2"
    type: "stage"
    phase: 1
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param2: "value2"
    replicas:
      cpu_count_min: 2
      cpu_count_max: 4

  - name: "stage3"
    type: "stage"
    phase: 2
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      param3: "value3"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 1

edges:
  - from: "stage1"
    to: "stage2"
    queue_size: 32
  - from: "stage2"
    to: "stage3"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(multi_stage_config)
            f.flush()

            try:
                config = load_pipeline_config(f.name)
                assert len(config.stages) == 3
                assert config.stages[0].name == "stage1"
                assert config.stages[1].name == "stage2"
                assert config.stages[2].name == "stage3"
                assert config.stages[0].type == "source"
                assert config.stages[1].type == "stage"
                assert config.stages[2].type == "stage"
            finally:
                os.unlink(f.name)

    def test_load_config_with_env_vars_in_nested_structures(self):
        """Test environment variable substitution in deeply nested structures."""
        nested_config = """
name: "Test Pipeline with Nested Env Vars"
description: "A test pipeline with environment variable substitution in nested structures"
stages:
  - name: "complex_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      database:
        host: $DB_HOST|"localhost"
        port: $DB_PORT|5432
        credentials:
          username: $DB_USER|"admin"
          password: $DB_PASS|"secret"
        settings:
          timeout: $DB_TIMEOUT|30
          pool_size: $DB_POOL_SIZE|10
      services:
        - name: $SERVICE1_NAME|"service1"
          endpoint: $SERVICE1_ENDPOINT|"http://service1:8001"
        - name: $SERVICE2_NAME|"service2"
          endpoint: $SERVICE2_ENDPOINT|"http://service2:8002"
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "complex_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(nested_config)
            f.flush()

            try:
                env_vars = {
                    "DB_HOST": "prod-db",
                    "DB_USER": "prod_user",
                    "SERVICE1_NAME": "prod-service1",
                    "SERVICE1_ENDPOINT": "https://prod-service1:9001",
                }
                with patch.dict(os.environ, env_vars):
                    config = load_pipeline_config(f.name)

                    db_config = config.stages[0].config["database"]
                    assert db_config["host"] == "prod-db"
                    assert db_config["port"] == 5432  # Default used
                    assert db_config["credentials"]["username"] == "prod_user"
                    assert db_config["credentials"]["password"] == "secret"  # Default used

                    services = config.stages[0].config["services"]
                    assert services[0]["name"] == "prod-service1"
                    assert services[0]["endpoint"] == "https://prod-service1:9001"
                    assert services[1]["name"] == "service2"  # Default used
            finally:
                os.unlink(f.name)

    def test_load_config_with_real_pipeline_structure(self):
        """Test loading config that mimics real pipeline structure."""
        real_pipeline_config = """
name: "Test Pipeline with Real Structure"
description: "A test pipeline with real structure"
stages:
  - name: "pdf_extractor"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      pdfium_config:
        auth_token: $NGC_API_KEY|""
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"page-elements:8001",
          $YOLOX_HTTP_ENDPOINT|"http://page-elements:8000/v1/infer"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|"grpc"
      nemoretriever_parse_config:
        auth_token: $NGC_API_KEY|""
        model_name: $NEMORETRIEVER_MODEL|"nvidia/nemoretriever-parse"
        endpoints: [
          $NEMORETRIEVER_GRPC_ENDPOINT|"",
          $NEMORETRIEVER_HTTP_ENDPOINT|"http://nemoretriever:8000/v1/chat/completions"
        ]
    replicas:
      cpu_count_min: 1
      cpu_count_max: 4

edges:
  - from: "source_stage"
    to: "pdf_extractor"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.75
  launch_simple_broker: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(real_pipeline_config)
            f.flush()

            try:
                with patch.dict(os.environ, {"NGC_API_KEY": "test-key"}):
                    config = load_pipeline_config(f.name)

                    assert config.stages[0].name == "pdf_extractor"
                    pdfium_config = config.stages[0].config["pdfium_config"]
                    assert pdfium_config["auth_token"] == "test-key"
                    assert len(pdfium_config["yolox_endpoints"]) == 2

                    nemo_config = config.stages[0].config["nemoretriever_parse_config"]
                    assert nemo_config["auth_token"] == "test-key"
                    assert nemo_config["model_name"] == "nvidia/nemoretriever-parse"

                    assert config.pipeline.launch_simple_broker == True
            finally:
                os.unlink(f.name)

    def test_load_config_with_special_yaml_types(self):
        """Test loading config with special YAML types (null, boolean, numeric)."""
        special_types_config = """
name: "Test Pipeline with Special YAML Types"
description: "A test pipeline with special YAML types"
stages:
  - name: "test_stage"
    type: "stage"
    phase: 0
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      string_value: "hello"
      int_value: 42
      float_value: 3.14
      bool_true: true
      bool_false: false
      null_value: null
      env_int: $INT_VAR|100
      env_float: $FLOAT_VAR|2.5
      env_bool: $BOOL_VAR|true
    replicas:
      cpu_count_min: 1
      cpu_count_max: 2

edges:
  - from: "source_stage"
    to: "test_stage"
    queue_size: 32

pipeline:
  disable_dynamic_scaling: false
  dynamic_memory_threshold: 0.8
  launch_simple_broker: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(special_types_config)
            f.flush()

            try:
                with patch.dict(os.environ, {"INT_VAR": "200", "FLOAT_VAR": "4.5"}):
                    config = load_pipeline_config(f.name)

                    stage_config = config.stages[0].config
                    assert stage_config["string_value"] == "hello"
                    assert stage_config["int_value"] == 42
                    assert stage_config["float_value"] == 3.14
                    assert stage_config["bool_true"] == True
                    assert stage_config["bool_false"] == False
                    assert stage_config["null_value"] is None
                    assert stage_config["env_int"] == 200  # YAML parses as int
                    assert stage_config["env_float"] == 4.5  # YAML parses as float
                    assert stage_config["env_bool"] == True  # YAML parses as bool
            finally:
                os.unlink(f.name)
