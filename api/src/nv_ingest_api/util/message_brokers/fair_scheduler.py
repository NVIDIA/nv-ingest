# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional
import logging
import time


class FairScheduler:
    """
    Simplified scheduler that fetches jobs from the default queue only.
    Uses the provided timeout value when polling the broker.
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
        Non-blocking sweep across queues in priority order (default last).
        If no job is found on a full sweep:
        - If timeout <= 0: return None immediately.
        - Else: sleep in 0.5s increments and retry until accumulated elapsed time >= timeout.
        """
        start = time.monotonic()
        while True:
            # Probe all queues without blocking (immediate -> micro -> small -> medium -> large -> default)
            for qname in ("immediate", "micro", "small", "medium", "large", "default"):
                try:
                    job = client.fetch_message(self.queues[qname], 0)
                    if job is not None:
                        return job
                except TimeoutError:
                    # Treat as no job available for this queue right now
                    continue

            # No job found in this sweep
            if timeout <= 0:
                return None

            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                return None

            # Sleep up to 0.5s, but not beyond remaining timeout
            remaining = timeout - elapsed
            sleep_time = 0.5 if remaining > 0.5 else remaining
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                return None
