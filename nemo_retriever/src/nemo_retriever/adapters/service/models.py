# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request/response models for the online REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StageMetric(BaseModel):
    stage: str
    duration_sec: float


class IngestResponse(BaseModel):
    ok: bool
    source_path: Optional[str] = None
    total_duration_sec: float = 0.0
    stages: List[StageMetric] = Field(default_factory=list)
    rows_written: int = 0
    error: Optional[str] = None


class AsyncIngestResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "queued", "running", "completed", "failed"]
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    lancedb_uri: Optional[str] = None
    lancedb_table: Optional[str] = None
    reranker: Optional[bool] = False
    reranker_endpoint: Optional[str] = None


class QueryHit(BaseModel):
    text: Optional[str] = None
    source: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = Field(default=None, description="Distance or rerank score")


class QueryResponse(BaseModel):
    query: str
    hits: List[Dict[str, Any]] = Field(default_factory=list)


class QueriesRequest(BaseModel):
    queries: List[str]
    top_k: int = 10
    lancedb_uri: Optional[str] = None
    lancedb_table: Optional[str] = None
    reranker: Optional[bool] = False
    reranker_endpoint: Optional[str] = None


class QueriesResponse(BaseModel):
    results: List[QueryResponse] = Field(default_factory=list)
