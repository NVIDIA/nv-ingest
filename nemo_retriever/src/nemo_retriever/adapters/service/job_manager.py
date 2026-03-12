# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
In-memory async job tracker for the online ingest server.

Each job represents a single document being processed through the ingest
pipeline. Jobs transition through: pending -> running -> completed | failed.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


class JobManager:
    """Manages async ingest jobs with in-memory state.

    Designed to run inside a Ray Serve actor: the ``run_fn`` callback
    executes the pipeline synchronously on a thread (via ``asyncio.to_thread``)
    so that the event loop remains responsive for status polling.
    """

    def __init__(
        self,
        run_fn: Callable[[pd.DataFrame, str, str], Tuple[bool, Dict[str, Any]]],
        pipeline_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self._run_fn = run_fn
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._semaphore = pipeline_semaphore or asyncio.Semaphore(1)

    def create_job(self) -> str:
        job_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        self._jobs[job_id] = {
            "status": "pending",
            "submitted_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        return job_id

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job_id,
            "status": job["status"],
            "submitted_at": job["submitted_at"],
            "completed_at": job["completed_at"],
            "error": job["error"],
        }

    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job["status"] not in ("completed", "failed"):
            return None
        return job["result"]

    async def submit(
        self,
        job_id: str,
        initial_df: pd.DataFrame,
        source_path: str,
        file_category: str,
    ) -> None:
        """Launch the pipeline on a background thread.

        The caller should ``asyncio.create_task(manager.submit(...))`` so this
        does not block the request handler.
        """
        self._jobs[job_id]["status"] = "queued"
        async with self._semaphore:
            self._jobs[job_id]["status"] = "running"
            try:
                ok, result = await asyncio.to_thread(self._run_fn, initial_df, source_path, file_category)
                now = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["status"] = "completed" if ok else "failed"
                self._jobs[job_id]["completed_at"] = now
                self._jobs[job_id]["result"] = result
                if not ok:
                    self._jobs[job_id]["error"] = result.get("error")
            except Exception as exc:
                now = datetime.now(timezone.utc).isoformat()
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["completed_at"] = now
                self._jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
                self._jobs[job_id]["result"] = {
                    "ok": False,
                    "source_path": source_path,
                    "error": self._jobs[job_id]["error"],
                }
