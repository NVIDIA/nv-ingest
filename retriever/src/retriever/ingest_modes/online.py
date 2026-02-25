# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Online runmode.

Low-latency request/response serving: documents are submitted to a Ray Serve
REST API (see ingest_modes.serve) which runs the same pipeline as inprocess
on a single document per request. This module provides the client-side
Ingestor that delegates to that REST endpoint.
"""

from __future__ import annotations

import glob
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from ..ingest import Ingestor
from ..params import EmbedParams
from ..params import ExtractParams
from ..params import IngestExecuteParams
from ..params import VdbUploadParams


class OnlineIngestor(Ingestor):
    """
    Client-side ingestor that submits documents to the online ingest REST API.

    Use create_ingestor(run_mode="online", base_url="http://localhost:7670")
    then .files(...).extract(...).embed(...).vdb_upload().ingest().
    extract/embed/vdb_upload record configuration for compatibility; the
    actual pipeline runs on the server. ingest() POSTs each file to the
    /ingest endpoint and returns a list of response metrics per document.
    """

    RUN_MODE = "online"

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        base_url: str = "http://localhost:7670",
    ) -> None:
        super().__init__(documents=documents)
        self._base_url = base_url.rstrip("/")
        self._input_documents: List[str] = self._documents

    def files(self, documents: Union[str, List[str]]) -> "OnlineIngestor":
        """Add local file paths or globs for submission to the online service."""
        if isinstance(documents, str):
            documents = [documents]

        for pattern in documents:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"Invalid document pattern: {pattern!r}")

            matches = glob.glob(pattern, recursive=True)
            if matches:
                files = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not files:
                    raise FileNotFoundError(f"Pattern resolved, but no files found: {pattern!r}")
                self._input_documents.extend(files)
                continue

            if os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
                continue

            raise FileNotFoundError(f"No local files found for: {pattern!r}")

        return self

    def buffers(self, buffers: Union[Tuple[str, BytesIO], List[Tuple[str, BytesIO]]]) -> "OnlineIngestor":
        """Add in-memory buffers for submission. Name is used as source_path."""
        if isinstance(buffers, (list, tuple)) and len(buffers) == 2 and not isinstance(buffers[0], (list, tuple)):
            buffers = [buffers]
        for name, buf in buffers:
            self._buffers.append((str(name), buf))
        return self

    def load(self) -> "OnlineIngestor":
        """No-op for API compatibility."""
        return self

    def extract(self, params: ExtractParams) -> "OnlineIngestor":
        """Record extraction config (server uses its own config). API compatibility."""
        _ = params
        return self

    def extract_txt(self, params: ExtractParams) -> "OnlineIngestor":
        """Record txt config. API compatibility. Online mode typically serves PDF only."""
        _ = params
        return self

    def embed(self, params: EmbedParams) -> "OnlineIngestor":
        """Record embed config (server uses its own config). API compatibility."""
        _ = params
        return self

    def vdb_upload(self, params: VdbUploadParams | None = None) -> "OnlineIngestor":
        """Record vdb config (server uses its own config). API compatibility."""
        _ = params
        return self

    def ingest(self, params: IngestExecuteParams | None = None) -> List[Dict[str, Any]]:
        """
        Submit each configured file (and buffer) to the online ingest REST API.

        Returns a list of response dicts (one per document) with ok, total_duration_sec,
        stages, rows_written, and optional error.
        """
        _ = params
        import httpx

        ingest_url = f"{self._base_url}/ingest"
        results: List[Dict[str, Any]] = []
        docs = list(self._input_documents)
        for path in docs:
            try:
                with open(path, "rb") as f:
                    body = f.read()
            except Exception as e:
                results.append({"ok": False, "error": str(e), "source_path": path})
                continue

            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(
                        ingest_url,
                        files={"file": (os.path.basename(path), body, "application/pdf")},
                        headers={"X-Source-Path": path},
                    )
            except Exception as e:
                results.append({"ok": False, "error": str(e), "source_path": path})
                continue

            try:
                data = resp.json()
            except Exception:
                data = {"ok": False, "error": resp.text[:500], "source_path": path}
            if resp.status_code != 200:
                data["ok"] = False
                data.setdefault("error", f"HTTP {resp.status_code}")
            results.append(data)

        for name, buf in self._buffers:
            buf.seek(0)
            body = buf.read()
            try:
                with httpx.Client(timeout=300.0) as client:
                    resp = client.post(
                        ingest_url,
                        files={"file": (name, body, "application/pdf")},
                        headers={"X-Source-Path": name},
                    )
            except Exception as e:
                results.append({"ok": False, "error": str(e), "source_path": name})
                continue
            try:
                data = resp.json()
            except Exception:
                data = {"ok": False, "error": resp.text[:500], "source_path": name}
            if resp.status_code != 200:
                data["ok"] = False
                data.setdefault("error", f"HTTP {resp.status_code}")
            results.append(data)

        return results

    def all_tasks(self) -> "OnlineIngestor":
        return self

    def dedup(self) -> "OnlineIngestor":
        return self

    def filter(self) -> "OnlineIngestor":
        return self

    def split(self) -> "OnlineIngestor":
        return self

    def store(self) -> "OnlineIngestor":
        return self

    def store_embed(self) -> "OnlineIngestor":
        return self

    def udf(self, *args: Any, **kwargs: Any) -> "OnlineIngestor":
        return self

    def save_intermediate_results(self, output_dir: str) -> "OnlineIngestor":
        return self

    def save_to_disk(self, *args: Any) -> "OnlineIngestor":
        _ = args
        return self

    def caption(self) -> "OnlineIngestor":
        return self

    def pdf_split_config(self, pages_per_chunk: int = 32) -> "OnlineIngestor":
        return self

    def completed_jobs(self) -> int:
        return 0

    def failed_jobs(self) -> int:
        return 0

    def cancelled_jobs(self) -> int:
        return 0

    def remaining_jobs(self) -> int:
        return 0

    def get_status(self) -> Dict[str, str]:
        return {}
