# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nv_ingest_api.internal.schemas.transform.transform_structural_text_splitter_schema import StructuralTextSplitterSchema


### Test Scenario 1.1: Basic Schema Instantiation ###


def test_structural_text_splitter_schema_defaults():
    """Test that schema instantiation with default values matches business requirements."""
    schema = StructuralTextSplitterSchema()
    
    # Verify business-specific defaults
    assert schema.max_chunk_size_tokens == 800  # NOT 512 like old splitter
    assert schema.enable_llm_enhancement is False  # Conservative default
    assert schema.markdown_headers_to_split_on == ["#", "##", "###", "####", "#####", "######"]
    
    # Verify LLM configuration defaults
    assert schema.llm_endpoint is None
    assert schema.llm_model_name == "meta/llama-3.1-8b-instruct"
    assert schema.llm_api_key_env_var == "NVIDIA_API_KEY"
    assert schema.max_llm_splits_per_document == 25
    
    # Verify prompt template contains required placeholder
    assert "{text}" in schema.llm_prompt
    assert "logical place to split" in schema.llm_prompt


def test_structural_text_splitter_schema_field_types():
    """Test that all fields have correct types."""
    schema = StructuralTextSplitterSchema()
    
    assert isinstance(schema.markdown_headers_to_split_on, list)
    assert isinstance(schema.max_chunk_size_tokens, int)
    assert isinstance(schema.enable_llm_enhancement, bool)
    assert isinstance(schema.llm_model_name, str)
    assert isinstance(schema.llm_api_key_env_var, str)
    assert isinstance(schema.max_llm_splits_per_document, int)
    assert isinstance(schema.llm_prompt, str)


### Test Scenario 1.2: Parameter Override Testing ###


def test_structural_text_splitter_schema_parameter_overrides():
    """Test that configuration overrides work correctly."""
    schema = StructuralTextSplitterSchema(
        max_chunk_size_tokens=1000,
        enable_llm_enhancement=True,
        markdown_headers_to_split_on=["#", "##"],
        llm_endpoint="https://custom.endpoint.com/v1",
        llm_model_name="custom/model",
        llm_api_key_env_var="CUSTOM_API_KEY",
        max_llm_splits_per_document=50,
        llm_prompt="Custom prompt with {text} placeholder"
    )
    
    # Verify each field is properly set
    assert schema.max_chunk_size_tokens == 1000
    assert schema.enable_llm_enhancement is True
    assert schema.markdown_headers_to_split_on == ["#", "##"]
    assert schema.llm_endpoint == "https://custom.endpoint.com/v1"
    assert schema.llm_model_name == "custom/model"
    assert schema.llm_api_key_env_var == "CUSTOM_API_KEY"
    assert schema.max_llm_splits_per_document == 50
    assert schema.llm_prompt == "Custom prompt with {text} placeholder"


def test_structural_text_splitter_schema_partial_overrides():
    """Test that partial parameter overrides work while keeping defaults."""
    schema = StructuralTextSplitterSchema(
        max_chunk_size_tokens=1200,
        enable_llm_enhancement=True
    )
    
    # Verify overridden fields
    assert schema.max_chunk_size_tokens == 1200
    assert schema.enable_llm_enhancement is True
    
    # Verify defaults are preserved
    assert schema.markdown_headers_to_split_on == ["#", "##", "###", "####", "#####", "######"]
    assert schema.llm_endpoint is None
    assert schema.llm_model_name == "meta/llama-3.1-8b-instruct"


### Test Scenario 1.3: Invalid Parameter Rejection ###


def test_structural_text_splitter_schema_rejects_negative_chunk_size():
    """Test that schema rejects negative chunk sizes."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(max_chunk_size_tokens=-100)
    
    error_msg = str(exc_info.value)
    assert "max_chunk_size_tokens" in error_msg


def test_structural_text_splitter_schema_rejects_zero_chunk_size():
    """Test that schema rejects zero chunk size."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(max_chunk_size_tokens=0)
    
    error_msg = str(exc_info.value)
    assert "max_chunk_size_tokens" in error_msg


def test_structural_text_splitter_schema_rejects_empty_header_list():
    """Test that schema rejects empty markdown header lists."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(markdown_headers_to_split_on=[])
    
    error_msg = str(exc_info.value)
    assert "markdown_headers_to_split_on" in error_msg


def test_structural_text_splitter_schema_rejects_invalid_header_types():
    """Test that schema rejects non-string header values."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(markdown_headers_to_split_on=["#", 123, "##"])
    
    error_msg = str(exc_info.value)
    assert "markdown_headers_to_split_on" in error_msg


def test_structural_text_splitter_schema_rejects_negative_max_llm_splits():
    """Test that schema rejects negative max LLM splits."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(max_llm_splits_per_document=-5)
    
    error_msg = str(exc_info.value)
    assert "max_llm_splits_per_document" in error_msg


def test_structural_text_splitter_schema_rejects_prompt_without_placeholder():
    """Test that schema rejects LLM prompt without {text} placeholder."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(llm_prompt="This prompt has no placeholder")
    
    error_msg = str(exc_info.value)
    assert "llm_prompt" in error_msg
    assert "{text}" in error_msg


def test_structural_text_splitter_schema_rejects_invalid_field_types():
    """Test that schema rejects incorrect field types."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(enable_llm_enhancement="not_a_boolean")
    
    error_msg = str(exc_info.value)
    assert "enable_llm_enhancement" in error_msg


def test_structural_text_splitter_schema_rejects_extra_fields():
    """Test that schema rejects extra fields for security."""
    with pytest.raises(ValidationError) as exc_info:
        StructuralTextSplitterSchema(
            max_chunk_size_tokens=800,
            extra_field="should_be_rejected"
        )
    
    error_msg = str(exc_info.value)
    assert "Extra inputs are not permitted" in error_msg


### Edge Cases and Business Logic Validation ###


def test_structural_text_splitter_schema_accepts_single_character_headers():
    """Test that single character headers are accepted."""
    schema = StructuralTextSplitterSchema(markdown_headers_to_split_on=["#"])
    assert schema.markdown_headers_to_split_on == ["#"]


def test_structural_text_splitter_schema_accepts_custom_header_patterns():
    """Test that custom header patterns are accepted."""
    custom_headers = ["=", "-", "*", "#", "##"]
    schema = StructuralTextSplitterSchema(markdown_headers_to_split_on=custom_headers)
    assert schema.markdown_headers_to_split_on == custom_headers


def test_structural_text_splitter_schema_accepts_large_chunk_sizes():
    """Test that large chunk sizes are accepted (for enterprise use)."""
    schema = StructuralTextSplitterSchema(max_chunk_size_tokens=5000)
    assert schema.max_chunk_size_tokens == 5000


def test_structural_text_splitter_schema_accepts_small_chunk_sizes():
    """Test that small but positive chunk sizes are accepted."""
    schema = StructuralTextSplitterSchema(max_chunk_size_tokens=50)
    assert schema.max_chunk_size_tokens == 50


def test_structural_text_splitter_schema_accepts_zero_max_llm_splits():
    """Test that zero max LLM splits is accepted (disables LLM splitting)."""
    schema = StructuralTextSplitterSchema(max_llm_splits_per_document=0)
    assert schema.max_llm_splits_per_document == 0


def test_structural_text_splitter_schema_llm_endpoint_url_validation():
    """Test LLM endpoint URL validation."""
    # Valid URLs should be accepted
    valid_urls = [
        "https://api.nvidia.com/v1",
        "http://localhost:8000/v1",
        "https://custom.endpoint.com/v1/chat/completions"
    ]
    
    for url in valid_urls:
        schema = StructuralTextSplitterSchema(llm_endpoint=url)
        assert schema.llm_endpoint == url
    
    # Invalid URLs should be rejected
    invalid_urls = [
        "not_a_url",
        "ftp://invalid.protocol.com",
        "https://",
        ""
    ]
    
    for url in invalid_urls:
        with pytest.raises(ValidationError):
            StructuralTextSplitterSchema(llm_endpoint=url)


### Configuration Consistency Tests ###


def test_structural_text_splitter_schema_llm_config_consistency():
    """Test that LLM configuration is consistent."""
    # When LLM enhancement is enabled, endpoint should be provided
    schema = StructuralTextSplitterSchema(
        enable_llm_enhancement=True,
        llm_endpoint="https://api.nvidia.com/v1"
    )
    assert schema.enable_llm_enhancement is True
    assert schema.llm_endpoint is not None


def test_structural_text_splitter_schema_business_requirement_alignment():
    """Test that schema aligns with business requirements."""
    schema = StructuralTextSplitterSchema()
    
    # Verify this is NOT a tokenization splitter (different defaults)
    assert schema.max_chunk_size_tokens != 512  # Old splitter default
    assert schema.max_chunk_size_tokens != 1024  # Text splitter default
    
    # Verify structural focus
    assert len(schema.markdown_headers_to_split_on) == 6  # Full header hierarchy
    assert all(h.startswith("#") for h in schema.markdown_headers_to_split_on)
    
    # Verify conservative LLM defaults
    assert schema.enable_llm_enhancement is False
    assert schema.max_llm_splits_per_document == 25  # Safety valve


### Step 2: Core Markdown Splitting Logic Tests ###

from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import pandas as pd

from nv_ingest_api.internal.transform.structural_split_text import (
    _split_by_markdown,
    _build_structural_split_documents,
    _get_llm_split_point,
    transform_text_split_structural_internal,
)
from nv_ingest_api.internal.enums.common import ContentTypeEnum


### Test Scenario 2.1: Basic Header Detection ###


def test_split_by_markdown_basic_header_detection():
    """Test basic header detection with exact document structure."""
    # Exact test document from requirements
    text = """# Introduction
This is intro content.

## Background  
This is background content.

### Details
This is detailed content."""

    # Mock tokenizer that returns token count equal to character count for simplicity
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    mock_tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=True: "".join(chr(t) for t in tokens)
    
    # Mock config
    config = StructuralTextSplitterSchema(
        max_chunk_size_tokens=1000,  # High enough to avoid splitting
        enable_llm_enhancement=False
    )
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Required assertions from specification
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    # Chunk 1: content="This is intro content." + header="Introduction"
    assert chunks[0] == "This is intro content."
    assert headers[0] == "Introduction"
    
    # Chunk 2: content="This is background content." + header="Introduction > Background"
    assert chunks[1] == "This is background content."
    assert headers[1] == "Introduction > Background"
    
    # Chunk 3: content="This is detailed content." + header="Introduction > Background > Details"
    assert chunks[2] == "This is detailed content."
    assert headers[2] == "Introduction > Background > Details"


### Test Scenario 2.2: Header Hierarchy Reset ###


def test_split_by_markdown_header_hierarchy_reset():
    """Test header hierarchy reset when encountering same-level headers."""
    # Test document from requirements
    text = """# Chapter 1
Content 1

## Section A
Content A

# Chapter 2  
Content 2

## Section B
Content B"""

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    mock_tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=True: "".join(chr(t) for t in tokens)
    
    config = StructuralTextSplitterSchema(
        max_chunk_size_tokens=1000,
        enable_llm_enhancement=False
    )
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should have 4 chunks
    assert len(chunks) == 4
    
    # Verify content
    assert chunks[0] == "Content 1"
    assert chunks[1] == "Content A"
    assert chunks[2] == "Content 2"
    assert chunks[3] == "Content B"
    
    # Verify hierarchy
    assert headers[0] == "Chapter 1"
    assert headers[1] == "Chapter 1 > Section A"
    
    # Chapter 2 should reset hierarchy (not "Chapter 1 > Chapter 2")
    assert headers[2] == "Chapter 2"
    
    # Must verify proper hierarchy: "Chapter 2 > Section B"
    assert headers[3] == "Chapter 2 > Section B"


### Test Scenario 2.3: Oversized Section Handling ###


def test_split_by_markdown_oversized_section_handling():
    """Test handling of oversized sections with LLM enhancement."""
    text = """# Large Section
This is a very large section that will exceed the token limit and should be split using LLM enhancement if enabled."""

    # Mock tokenizer that reports high token count for the content
    mock_tokenizer = MagicMock()
    def mock_encode(text, add_special_tokens=False):
        if "This is a very large section" in text:
            return list(range(1000))  # Return 1000 tokens (oversized)
        return list(range(len(text)))
    
    mock_tokenizer.encode.side_effect = mock_encode
    mock_tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=True: "decoded_chunk"
    
    # Mock LLM client
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "very large section"
    mock_client.chat.completions.create.return_value = mock_completion
    
    config = StructuralTextSplitterSchema(
        max_chunk_size_tokens=500,  # Set low to trigger oversized handling
        enable_llm_enhancement=True
    )
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, mock_client)
    
    # Should have more than 1 chunk due to splitting
    assert len(chunks) > 1
    
    # Verify LLM was called
    assert mock_client.chat.completions.create.called
    
    # All chunks should have the same header
    for header in headers:
        assert header == "Large Section"


def test_split_by_markdown_fallback_tokenization_splitting():
    """Test fallback to tokenization splitting when LLM is disabled."""
    text = """# Oversized Section
This is a very large section that will exceed the token limit and should be split using tokenization fallback."""

    # Mock tokenizer that reports high token count
    mock_tokenizer = MagicMock()
    def mock_encode(text, add_special_tokens=False):
        if "This is a very large section" in text:
            return list(range(1000))  # Return 1000 tokens (oversized)
        return list(range(len(text)))
    
    mock_tokenizer.encode.side_effect = mock_encode
    mock_tokenizer.decode.side_effect = lambda tokens, skip_special_tokens=True: f"chunk_{len(tokens)}_tokens"
    
    config = StructuralTextSplitterSchema(
        max_chunk_size_tokens=300,  # Set low to trigger splitting
        enable_llm_enhancement=False  # Disable LLM to test fallback
    )
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should have multiple chunks from fallback splitting
    assert len(chunks) > 1
    
    # All chunks should have the same header
    for header in headers:
        assert header == "Oversized Section"
    
    # Verify tokenizer decode was called for sub-chunks
    assert mock_tokenizer.decode.called


### Test Scenario 2.4: Edge Cases ###


def test_split_by_markdown_no_headers():
    """Test document with no headers."""
    text = "This is a document with no headers at all. Just plain text content."
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should have 1 chunk with empty header
    assert len(chunks) == 1
    assert chunks[0] == text
    assert headers[0] == ""


def test_split_by_markdown_empty_sections():
    """Test handling of empty sections."""
    text = """# Header 1

## Header 2
Some content here.

### Header 3

## Header 4
More content here."""
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should only have chunks for sections with content
    assert len(chunks) == 2
    assert chunks[0] == "Some content here."
    assert chunks[1] == "More content here."
    assert headers[0] == "Header 1 > Header 2"
    assert headers[1] == "Header 1 > Header 4"


def test_split_by_markdown_malformed_headers():
    """Test handling of malformed headers."""
    text = """# Proper Header
Content 1

#MissingSpace
Content 2

## Proper Sub Header
Content 3"""
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Malformed header should be treated as content
    assert len(chunks) == 2
    assert "Content 1" in chunks[0] and "#MissingSpace" in chunks[0] and "Content 2" in chunks[0]
    assert chunks[1] == "Content 3"
    assert headers[0] == "Proper Header"
    assert headers[1] == "Proper Header > Proper Sub Header"


def test_split_by_markdown_headers_without_content():
    """Test headers without content."""
    text = """# Header 1
Content 1

## Header 2

### Header 3
Content 3"""
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should only include chunks with actual content
    assert len(chunks) == 2
    assert chunks[0] == "Content 1"
    assert chunks[1] == "Content 3"
    assert headers[0] == "Header 1"
    assert headers[1] == "Header 1 > Header 2 > Header 3"


### Test _build_structural_split_documents Function ###


def test_build_structural_split_documents():
    """Test building structural split documents with hierarchical headers."""
    # Create a mock row
    row = SimpleNamespace()
    row.metadata = {
        "content": "original content",
        "source_metadata": {"source_id": "test_doc"}
    }
    row.document_type = ContentTypeEnum.TEXT
    
    chunks = ["First chunk", "Second chunk"]
    headers = ["Header 1", "Header 1 > Header 2"]
    
    documents = _build_structural_split_documents(row, chunks, headers)
    
    # Should have 2 documents
    assert len(documents) == 2
    
    # Verify first document
    assert documents[0]["document_type"] == ContentTypeEnum.TEXT.value
    assert documents[0]["metadata"]["content"] == "First chunk"
    assert documents[0]["metadata"]["custom_content"]["hierarchical_header"] == "Header 1"
    
    # Verify second document
    assert documents[1]["document_type"] == ContentTypeEnum.TEXT.value
    assert documents[1]["metadata"]["content"] == "Second chunk"
    assert documents[1]["metadata"]["custom_content"]["hierarchical_header"] == "Header 1 > Header 2"
    
    # Verify each document has a unique UUID
    assert documents[0]["uuid"] != documents[1]["uuid"]


def test_build_structural_split_documents_skip_empty_chunks():
    """Test that empty chunks are skipped."""
    row = SimpleNamespace()
    row.metadata = {"content": "original"}
    row.document_type = ContentTypeEnum.TEXT
    
    chunks = ["Valid chunk", "", "   ", None, "Another valid chunk"]
    headers = ["H1", "H2", "H3", "H4", "H5"]
    
    documents = _build_structural_split_documents(row, chunks, headers)
    
    # Should only have 2 documents (skipping empty ones)
    assert len(documents) == 2
    assert documents[0]["metadata"]["content"] == "Valid chunk"
    assert documents[1]["metadata"]["content"] == "Another valid chunk"


### Test LLM Integration ###


def test_get_llm_split_point_success():
    """Test successful LLM split point retrieval."""
    text = "This is a long text that needs to be split at a logical point."
    
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "logical point"
    mock_client.chat.completions.create.return_value = mock_completion
    
    config = StructuralTextSplitterSchema()
    
    result = _get_llm_split_point(text, mock_client, config)
    
    assert result == "logical point"
    assert mock_client.chat.completions.create.called


def test_get_llm_split_point_invalid_response():
    """Test handling of invalid LLM response."""
    text = "This is a long text that needs to be split."
    
    # Mock OpenAI client with invalid response
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "invalid substring not in text"
    mock_client.chat.completions.create.return_value = mock_completion
    
    config = StructuralTextSplitterSchema()
    
    result = _get_llm_split_point(text, mock_client, config)
    
    assert result is None


def test_get_llm_split_point_exception_handling():
    """Test handling of LLM API exceptions."""
    text = "This is a long text that needs to be split."
    
    # Mock OpenAI client that raises an exception
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    config = StructuralTextSplitterSchema()
    
    result = _get_llm_split_point(text, mock_client, config)
    
    assert result is None


### Test Integration with Different Header Patterns ###


def test_split_by_markdown_custom_header_patterns():
    """Test splitting with custom header patterns."""
    text = """# Standard Header
Content 1

## Another Header
Content 2

### Third Level
Content 3"""
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    # Test with custom header patterns
    config = StructuralTextSplitterSchema(
        markdown_headers_to_split_on=["#", "##"]  # Only first two levels
    )
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should have 2 chunks (### not recognized as header, becomes content)
    assert len(chunks) == 2
    assert chunks[0] == "Content 1"
    assert chunks[1] == "Content 2\n\n### Third Level\nContent 3"
    
    assert headers[0] == "Standard Header"
    assert headers[1] == "Standard Header > Another Header"


### Test Performance and Edge Cases ###


def test_split_by_markdown_large_number_of_headers():
    """Test handling of documents with many headers."""
    # Generate document with many headers
    sections = []
    for i in range(50):
        sections.append(f"## Section {i}")
        sections.append(f"Content for section {i}")
    
    text = "# Main Document\n" + "\n\n".join(sections)
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    # Should have 50 chunks (one per section)
    assert len(chunks) == 50
    
    # All headers should start with "Main Document > Section"
    for header in headers:
        assert header.startswith("Main Document > Section")


def test_split_by_markdown_unicode_headers():
    """Test handling of Unicode characters in headers."""
    text = """# 文档标题
中文内容

## Español Sección
Contenido en español

### Русский раздел
Русский контент"""
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
    
    config = StructuralTextSplitterSchema()
    
    chunks, headers = _split_by_markdown(text, mock_tokenizer, config, None)
    
    assert len(chunks) == 3
    assert headers[0] == "文档标题"
    assert headers[1] == "文档标题 > Español Sección"
    assert headers[2] == "文档标题 > Español Sección > Русский раздел"


### Step 3: Ray Pipeline Stage Testing ###

from unittest.mock import patch, MagicMock
import uuid

from src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter import structural_text_splitter_fn
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.control_message_task import ControlMessageTask


def create_mock_control_message(tasks=None, df_data=None, task_properties=None):
    """Helper function to create mock control messages for testing."""
    control_message = IngestControlMessage()
    
    # Set payload DataFrame
    if df_data is None:
        df_data = [{"document_type": "text", "metadata": {"content": "# Default\nDefault content"}}]
    
    test_df = pd.DataFrame(df_data)
    control_message.payload(test_df)
    
    # Add tasks
    if tasks:
        for task_type in tasks:
            # Use provided task properties or empty dict
            properties = task_properties or {}
            task = ControlMessageTask(
                type=task_type,
                id=str(uuid.uuid4()),
                properties=properties
            )
            control_message.add_task(task)
    
    return control_message


### Test Scenario 3.1: Task Routing Validation ###


def test_structural_text_splitter_fn_task_routing_structural_split():
    """Test that ONLY 'structural_split' tasks are processed."""
    # Create control message with structural_split task
    control_msg = create_mock_control_message(tasks=["structural_split"])
    schema = StructuralTextSplitterSchema()
    
    # Mock the internal function to verify it's called
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame([{"test": "result"}])
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        # Verify internal function was called
        assert mock_transform.called
        
        # Verify task was removed
        assert not any(task.type == "structural_split" for task in result.get_tasks())


def test_structural_text_splitter_fn_task_routing_split_task():
    """Test that 'split' tasks are NOT processed by structural splitter."""
    # Create control message with 'split' task (regular text splitter)
    control_msg = create_mock_control_message(tasks=["split"])
    schema = StructuralTextSplitterSchema()
    
    # Should raise ValueError when trying to remove non-existent 'structural_split' task
    with pytest.raises(ValueError, match="Task 'structural_split' not found"):
        structural_text_splitter_fn(control_msg, schema)


def test_structural_text_splitter_fn_task_routing_other_task():
    """Test that other tasks are NOT processed by structural splitter."""
    # Create control message with 'embed' task
    control_msg = create_mock_control_message(tasks=["embed"])
    schema = StructuralTextSplitterSchema()
    
    # Should raise ValueError when trying to remove non-existent 'structural_split' task
    with pytest.raises(ValueError, match="Task 'structural_split' not found"):
        structural_text_splitter_fn(control_msg, schema)


def test_structural_text_splitter_fn_task_routing_multiple_tasks():
    """Test processing with multiple tasks where only structural_split is handled."""
    # Create control message with multiple tasks
    control_msg = create_mock_control_message(tasks=["embed", "structural_split", "caption"])
    schema = StructuralTextSplitterSchema()
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame([{"test": "result"}])
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        # Verify structural_split task was removed
        remaining_tasks = [task.type for task in result.get_tasks()]
        assert "structural_split" not in remaining_tasks
        
        # Verify other tasks remain
        assert "embed" in remaining_tasks
        assert "caption" in remaining_tasks


### Test Scenario 3.2: Configuration Integration ###


def test_structural_text_splitter_fn_config_integration():
    """Test that task config overrides schema defaults."""
    # Create task with custom properties
    task_properties = {
        "max_chunk_size_tokens": 1000,
        "enable_llm_enhancement": True,
        "markdown_headers_to_split_on": ["#", "##"]
    }
    
    control_msg = IngestControlMessage()
    control_msg.payload(pd.DataFrame([{"document_type": "text", "metadata": {"content": "test"}}]))
    
    task = ControlMessageTask(
        type="structural_split",
        id=str(uuid.uuid4()),
        properties=task_properties
    )
    control_msg.add_task(task)
    
    # Schema with different defaults
    schema = StructuralTextSplitterSchema(
        max_chunk_size_tokens=800,
        enable_llm_enhancement=False,
        markdown_headers_to_split_on=["#", "##", "###", "####", "#####", "######"]
    )
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame([{"test": "result"}])
        
        structural_text_splitter_fn(control_msg, schema)
        
        # Verify function was called with merged config
        assert mock_transform.called
        call_args = mock_transform.call_args
        
        # Check task_config parameter
        task_config = call_args[1]['task_config']
        assert task_config["max_chunk_size_tokens"] == 1000  # From task
        assert task_config["enable_llm_enhancement"] is True  # From task
        assert task_config["markdown_headers_to_split_on"] == ["#", "##"]  # From task


def test_structural_text_splitter_fn_remove_task_by_type():
    """Test that remove_task_by_type works correctly."""
    # Create control message with structural_split task and valid schema properties
    task_properties = {"max_chunk_size_tokens": 1200, "enable_llm_enhancement": False}
    
    control_msg = IngestControlMessage()
    control_msg.payload(pd.DataFrame([{"document_type": "text", "metadata": {"content": "test"}}]))
    
    task = ControlMessageTask(
        type="structural_split",
        id=str(uuid.uuid4()),
        properties=task_properties
    )
    control_msg.add_task(task)
    
    schema = StructuralTextSplitterSchema()
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame([{"test": "result"}])
        
        structural_text_splitter_fn(control_msg, schema)
        
        # Verify the task properties were extracted correctly
        call_args = mock_transform.call_args
        task_config = call_args[1]['task_config']
        assert task_config["max_chunk_size_tokens"] == 1200
        assert task_config["enable_llm_enhancement"] is False


### Test Scenario 3.3: DataFrame Processing ###


def test_structural_text_splitter_fn_dataframe_processing():
    """Test realistic DataFrame processing with hierarchical headers."""
    # Create realistic test DataFrame
    test_data = [{
        "document_type": "text",
        "metadata": {
            "content": "# Title\nContent here\n## Section\nMore content",
            "source_metadata": {"source_id": "test_doc"},
            "content_metadata": {"type": "text"}
        }
    }]
    
    control_msg = create_mock_control_message(
        tasks=["structural_split"],
        df_data=test_data
    )
    schema = StructuralTextSplitterSchema()
    
    # Don't mock the internal function - test actual processing
    with patch('nv_ingest_api.internal.transform.structural_split_text.AutoTokenizer') as mock_tokenizer_class:
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        # Verify output has multiple rows (split chunks)
        result_df = result.payload()
        assert len(result_df) > 1, "Should have multiple chunks from splitting"
        
        # Verify hierarchical headers are present
        for _, row in result_df.iterrows():
            metadata = row["metadata"]
            assert "custom_content" in metadata, "Should have custom_content field"
            assert "hierarchical_header" in metadata["custom_content"], "Should have hierarchical_header"
        
        # Verify content was split properly
        contents = [row["metadata"]["content"] for _, row in result_df.iterrows()]
        assert any("Content here" in content for content in contents)
        assert any("More content" in content for content in contents)
        
        # Verify hierarchical header structure
        headers = [row["metadata"]["custom_content"]["hierarchical_header"] for _, row in result_df.iterrows()]
        assert any("Title" in header for header in headers)
        assert any("Title > Section" in header for header in headers)


def test_structural_text_splitter_fn_dataframe_processing_empty_input():
    """Test handling of empty DataFrame input."""
    control_msg = create_mock_control_message(
        tasks=["structural_split"],
        df_data=[]  # Empty DataFrame
    )
    schema = StructuralTextSplitterSchema()
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame()
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        # Should handle empty input gracefully
        assert isinstance(result, IngestControlMessage)
        assert len(result.payload()) == 0


def test_structural_text_splitter_fn_dataframe_processing_mixed_types():
    """Test processing DataFrame with mixed document types."""
    test_data = [
        {
            "document_type": "text",
            "metadata": {
                "content": "# Text Document\nText content here",
                "content_metadata": {"type": "text"}
            }
        },
        {
            "document_type": "image", 
            "metadata": {
                "content": "base64_image_data",
                "content_metadata": {"type": "image"}
            }
        }
    ]
    
    control_msg = create_mock_control_message(
        tasks=["structural_split"],
        df_data=test_data
    )
    schema = StructuralTextSplitterSchema()
    
    with patch('nv_ingest_api.internal.transform.structural_split_text.AutoTokenizer') as mock_tokenizer_class:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        result_df = result.payload()
        
        # Should have processed text documents and left image documents unchanged
        text_docs = result_df[result_df["document_type"] == "text"]
        image_docs = result_df[result_df["document_type"] == "image"]
        
        # Text document should be processed (potentially split)
        assert len(text_docs) >= 1
        
        # Image document should remain unchanged
        assert len(image_docs) == 1
        assert image_docs.iloc[0]["metadata"]["content"] == "base64_image_data"


### Test Error Handling ###


def test_structural_text_splitter_fn_error_handling():
    """Test error handling scenarios."""
    control_msg = create_mock_control_message(tasks=["structural_split"])
    schema = StructuralTextSplitterSchema()
    
    # Test internal function error handling
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.side_effect = Exception("Internal processing error")
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Internal processing error"):
            structural_text_splitter_fn(control_msg, schema)


def test_structural_text_splitter_fn_invalid_payload():
    """Test handling of invalid payload types."""
    control_msg = IngestControlMessage()
    
    # Try to set invalid payload (not a DataFrame)
    with pytest.raises(ValueError, match="Payload must be a pandas DataFrame"):
        control_msg.payload("not_a_dataframe")


### Test Logging and Metadata ###


def test_structural_text_splitter_fn_logging():
    """Test that appropriate logging occurs during processing."""
    control_msg = create_mock_control_message(tasks=["structural_split"])
    schema = StructuralTextSplitterSchema()
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame([{"test": "result"}])
        
        with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.logger') as mock_logger:
            structural_text_splitter_fn(control_msg, schema)
            
            # Verify logging calls were made
            assert mock_logger.debug.called or mock_logger.info.called


def test_structural_text_splitter_fn_payload_update():
    """Test that control message payload is properly updated."""
    original_data = [{"document_type": "text", "metadata": {"content": "original"}}]
    control_msg = create_mock_control_message(
        tasks=["structural_split"],
        df_data=original_data
    )
    schema = StructuralTextSplitterSchema()
    
    new_data = [{"document_type": "text", "metadata": {"content": "processed"}}]
    
    with patch('src.nv_ingest.framework.orchestration.ray.stages.transforms.structural_text_splitter.transform_text_split_structural_internal') as mock_transform:
        mock_transform.return_value = pd.DataFrame(new_data)
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        # Verify payload was updated
        result_df = result.payload()
        assert len(result_df) == 1
        assert result_df.iloc[0]["metadata"]["content"] == "processed"


### Integration with Real Markdown Processing ###


def test_structural_text_splitter_fn_real_markdown_integration():
    """Test integration with real markdown processing (minimal mocking)."""
    markdown_content = """# Introduction
This is the introduction section with important information.

## Background
Here we provide background context that helps understand the problem.

### Technical Details
This section contains technical implementation details.

## Implementation
The implementation follows these principles."""
    
    test_data = [{
        "document_type": "text",
        "metadata": {
            "content": markdown_content,
            "source_metadata": {"source_id": "test_doc", "source_type": "md"},
            "content_metadata": {"type": "text"}
        }
    }]
    
    control_msg = create_mock_control_message(
        tasks=["structural_split"],
        df_data=test_data
    )
    
    # Use higher chunk size to avoid triggering oversized section handling
    schema = StructuralTextSplitterSchema(max_chunk_size_tokens=2000)
    
    # Mock only the tokenizer, let everything else run normally
    with patch('nv_ingest_api.internal.transform.structural_split_text.AutoTokenizer') as mock_tokenizer_class:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x, add_special_tokens=False: list(range(len(x)))
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        result = structural_text_splitter_fn(control_msg, schema)
        
        result_df = result.payload()
        
        # Should have 4 chunks: Introduction, Background, Technical Details, Implementation
        assert len(result_df) == 4
        
        # Verify hierarchical headers
        headers = [row["metadata"]["custom_content"]["hierarchical_header"] for _, row in result_df.iterrows()]
        
        assert "Introduction" in headers
        assert "Introduction > Background" in headers
        assert "Introduction > Background > Technical Details" in headers
        assert "Introduction > Implementation" in headers
        
        # Verify content preservation
        contents = [row["metadata"]["content"] for _, row in result_df.iterrows()]
        assert any("introduction section" in content.lower() for content in contents)
        assert any("background context" in content.lower() for content in contents)
        assert any("technical implementation" in content.lower() for content in contents)
        assert any("implementation follows" in content.lower() for content in contents)