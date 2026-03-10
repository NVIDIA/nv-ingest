# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class AgentConfig(BaseModel):
    """Configuration for the agent."""

    model_config = ConfigDict(extra="forbid")
    system_prompt: str = "02_v1.j2"
    enforce_top_k: bool = True
    target_top_k: Optional[int] = 10
    extended_relevance: bool = True
    max_steps: Optional[int] = None
    disable_think: bool = False
    only_warn_on_error: bool = True
    end_tool: str = "final_results"
    use_image_explainer: bool = False
    use_query_rewriting: bool = False
    image_explainer_prompt: str = "simple"
    ensure_new_docs: bool = True
    user_msg_type: str = "with_results"
    selection_topk_list: List[int] = [5, 10]
    calculate_rrf: bool = True
    selection_prompt: str = "01_v0.j2"
    selection_max_steps: int = 10
    main_agent_only: bool = False


class LLMConfig(BaseModel):
    """Configuration for the LLM (LiteLLM wrapper)."""

    model_config = ConfigDict(extra="forbid")
    model: str
    api_key: Optional[str] = None
    tool_choice: str = "auto"
    reasoning_effort: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    num_retries: Optional[int] = 4
    max_completion_tokens: Optional[int] = None
    raw_log_pardir: Optional[str] = None
    instant_log: bool = False
    strict_error_handling: bool = False
    drop_params: bool = False
    allowed_openai_params: Optional[List[str]] = None
