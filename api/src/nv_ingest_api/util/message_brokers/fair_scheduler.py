# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional
import logging


class FairScheduler:
    """
    Simplified scheduler with support for multiple Redis queues derived from a base name.

    Queues supported (derived from `base_queue`):
    - default: `<base_queue>`
    - immediate: `<base_queue>_immediate` (always highest priority)
    - micro: `<base_queue>_micro`
    - small: `<base_queue>_small`
    - medium: `<base_queue>_medium`
    - large: `<base_queue>_large`

    Current behavior intentionally remains the same as before: only pulls from the
    default queue via `fetch_next()`. A multi-queue aware method `fetch_next_multi()`
    is provided but not used by default.
    """

    def __init__(
        self,
        base_queue: str,
        total_buffer_capacity: int = 1,
        num_prefetch_threads: int = 0,
        prefetch_poll_interval: float = 0.0,
        prefetch_non_immediate: bool = False,
    ) -> None:
        self.base_queue = base_queue

        # Define all derived queues; default behavior still uses only "default"
        self.queues: Dict[str, str] = {
            "default": f"{base_queue}",
            "immediate": f"{base_queue}_immediate",
            "micro": f"{base_queue}_micro",
            "small": f"{base_queue}_small",
            "medium": f"{base_queue}_medium",
            "large": f"{base_queue}_large",
        }

        # Priority order for multi-queue fetching; "immediate" always first
        self._priority_order = [
            "immediate",
            "micro",
            "small",
            "medium",
            "large",
            "default",
        ]

        # Logger
        self._logger = logging.getLogger(__name__)

        # No prefetching - just direct calls
        self._total_buffer_capacity: int = int(total_buffer_capacity)
        self._num_prefetch_threads: int = int(num_prefetch_threads)
        self._prefetch_poll_interval: float = float(prefetch_poll_interval)
        self._prefetch_non_immediate: bool = bool(prefetch_non_immediate)

    # Context manager helpers for clean shutdown
    def __enter__(self) -> "FairScheduler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------- Public API ----------------------------
    def fetch_next(self, client, timeout: float = 0.0) -> Optional[dict]:
        """
        Fetch from default queue only - reproduces original behavior.
        """
        try:
            job = client.fetch_message(self.queues["default"], timeout)
        except TimeoutError:
            job = None
        return job

    def fetch_next_multi(self, client, timeout: float = 0.0) -> Optional[dict]:
        """
        Multi-queue aware fetch that prioritizes the immediate queue first, then
        walks other queues by priority. This method is provided for future use
        and is not invoked by default callers.

        Parameters
        ----------
        client : Any
            Broker client with a fetch_message(queue, timeout) API.
        timeout : float, optional
            Total timeout budget to apply when polling the highest-priority queue.
            Lower-priority queues use non-blocking checks (timeout=0) to avoid
            starving the immediate queue.
        """
        # Always give the immediate queue first chance with the provided timeout
        try:
            job = client.fetch_message(self.queues["immediate"], timeout)
            if job is not None:
                return job
        except TimeoutError:
            # No immediate job within timeout; continue to best-effort checks
            pass

        # Best-effort/non-blocking checks for other queues in priority order
        for qname in self._priority_order:
            if qname in ("immediate",):
                continue  # already handled
            try:
                job = client.fetch_message(self.queues[qname], 0)
                if job is not None:
                    return job
            except TimeoutError:
                continue

        return None

    def close(self) -> None:
        """Cleanly stop - no-op since no threads."""
        pass

    def get_cycle_quotas(self) -> Dict[str, int]:
        return {}
