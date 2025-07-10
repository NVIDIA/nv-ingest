# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict, HttpUrl


class StructuralTextSplitterSchema(BaseModel):
    """
    Configuration schema for the structural text splitter stage.
    
    This splitter primarily splits documents by markdown headers to preserve
    hierarchical structure. Optionally, it can use an LLM to find logical
    split points for sections that exceed the token limit.
    """

    markdown_headers_to_split_on: List[str] = Field(
        default=["#", "##", "###", "####", "#####", "######"],
        min_length=1,
        description="A list of markdown header prefixes (e.g., '#', '##') to use as primary split points.",
    )
    max_chunk_size_tokens: int = Field(
        default=800,
        gt=0,
        description="The maximum number of tokens for a chunk. Sections exceeding this limit will trigger LLM enhancement (if enabled).",
    )
    enable_llm_enhancement: bool = Field(
        default=False,
        description="If True, use an LLM to find logical split points for sections that exceed the max_chunk_size_tokens.",
    )
    llm_endpoint: Optional[str] = Field(
        default=None,
        description="The API endpoint for the LLM service (e.g., NVIDIA NIM endpoint). If not set, LLM enhancement will be disabled.",
    )
    llm_model_name: Optional[str] = Field(
        default="meta/llama-3.1-8b-instruct",
        description="The name of the model to use for finding split points.",
    )
    llm_api_key_env_var: Optional[str] = Field(
        default="NVIDIA_API_KEY",
        description="The name of the environment variable containing the API key for the LLM service.",
    )
    max_llm_splits_per_document: int = Field(
        default=25,
        ge=0,
        description="A safety valve to limit the number of LLM calls for a single document to prevent excessive cost/latency.",
    )
    llm_prompt: str = Field(
        default="""You are a text processing utility. Your task is to find the most logical place to split the following text into two parts.
The text is a chunk from a larger document that is too long.
Respond with ONLY the first 15 words of the second part, starting from the most logical split point.
Your response must be a verbatim substring from the original text. Do not add any commentary, explanations, or quotation marks.

Original Text:
{text}
""",
        description="The prompt template to use for requesting split points from the LLM. Must include a '{text}' placeholder.",
    )

    @field_validator("llm_endpoint")
    @classmethod
    def validate_llm_endpoint(cls, v):
        if v is not None:
            # Empty string should be rejected
            if v == "":
                raise ValueError("llm_endpoint cannot be an empty string")
            # Basic URL validation - must start with http/https
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("llm_endpoint must be a valid HTTP/HTTPS URL")
            # Must have more than just the protocol
            if v in ["http://", "https://"]:
                raise ValueError("llm_endpoint must be a complete URL")
        return v

    @field_validator("llm_prompt")
    @classmethod
    def validate_llm_prompt(cls, v):
        if "{text}" not in v:
            raise ValueError("llm_prompt must contain a '{text}' placeholder")
        return v

    model_config = ConfigDict(extra="forbid") 