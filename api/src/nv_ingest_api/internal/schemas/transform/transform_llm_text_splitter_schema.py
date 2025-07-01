# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
from pydantic import BaseModel, Field


class LLMTextSplitterSchema(BaseModel):
    """
    Configuration schema for the LLM-powered text splitter stage.
    """

    markdown_headers_to_split_on: List[str] = Field(
        default=["#", "##", "###", "####", "#####", "######"],
        description="A list of markdown header prefixes (e.g., '#', '##') to use as primary split points.",
    )
    chunk_size: int = Field(
        default=512,
        description="The target maximum number of tokens for a chunk.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="The number of tokens to overlap between chunks. Used for both hard and LLM-based splits.",
    )
    subsplit_with_llm: bool = Field(
        default=False,
        description="If True, use an LLM to find logical split points for chunks that exceed the chunk_size.",
    )
    llm_endpoint: Optional[str] = Field(
        default=None,
        description="The API endpoint for the LLM service (e.g., NVIDIA NIM endpoint). If not set, LLM splitting will be disabled.",
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