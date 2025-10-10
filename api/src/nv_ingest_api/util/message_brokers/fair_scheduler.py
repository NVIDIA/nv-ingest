# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Deque, Any
from collections import deque
import threading
import logging


class FairScheduler:
    """
    Fair scheduler for broker queues derived from a base queue name.

    Given a base queue, derives the following physical queues (string names):
      - {base}_immediate
      - {base}_micro
      - {base}_small
      - {base}_medium
      - {base}_large
      - {base}  (backwards compatibility, treated as "default")

    Scheduling rules per cycle (in order):
      1) Drain 'immediate' first until empty.
      2) Try up to 4 pulls from 'micro' (skip if empty)
      3) Try up to 4 pulls from 'small' (skip if empty)
      4) Try up to 2 pulls from 'medium' (skip if empty)
      5) Try up to 1 pull from 'large' (skip if empty)
      6) Try up to 1 pull from 'default' (skip if empty)

    Starvation rule:
      - If any queue has been starved for more than N cycles (empty queues do not starve),
        ensure we pull from it next (unless there are items in 'immediate').

    This class is stateless with respect to the broker implementation. It expects a client
    with a method: fetch_message(queue: str, timeout: float) -> Optional[dict]
    """

    def __init__(
        self,
        base_queue: str,
        starvation_cycles: int = 10,
        total_buffer_capacity: int = 16,
        num_prefetch_threads: int = 5,
        prefetch_poll_interval: float = 0.005,
    ) -> None:
        self.base_queue = base_queue
        self.starvation_cycles = max(1, int(starvation_cycles))

        # Derived queue names
        self.queues: Dict[str, str] = {
            "immediate": f"{base_queue}_immediate",
            "micro": f"{base_queue}_micro",
            "small": f"{base_queue}_small",
            "medium": f"{base_queue}_medium",
            "large": f"{base_queue}_large",
            "default": f"{base_queue}",
        }

        # Quotas per cycle
        self._cycle_quotas_default: Dict[str, int] = {
            "micro": 4,
            "small": 4,
            "medium": 2,
            "large": 1,
            "default": 1,
        }
        self._cycle_quotas: Dict[str, int] = dict(self._cycle_quotas_default)

        # Starvation tracking (counts cycles since last served/attempted)
        self._starved_cycles: Dict[str, int] = {k: 0 for k in self._cycle_quotas_default.keys()}

        # Per-cycle bookkeeping
        self._attempted_this_cycle: Dict[str, bool] = {k: False for k in self._cycle_quotas_default.keys()}
        self._served_this_cycle: Dict[str, bool] = {k: False for k in self._cycle_quotas_default.keys()}

        # Logger
        self._logger = logging.getLogger(__name__)

        # Prefetch buffers and threading primitives (multi-threaded prefetch)
        self._total_buffer_capacity: int = int(total_buffer_capacity)
        self._num_prefetch_threads: int = int(num_prefetch_threads)
        # Per-thread fixed capacities summing to total (distribute remainder to first threads)
        base = self._total_buffer_capacity // self._num_prefetch_threads
        rem = self._total_buffer_capacity % self._num_prefetch_threads
        self._per_thread_caps: List[int] = [base + (1 if i < rem else 0) for i in range(self._num_prefetch_threads)]
        self._buffers: List[Deque[Any]] = [deque() for _ in range(self._num_prefetch_threads)]
        # Dedicated buffer for immediate queue to preserve strict ordering
        self._immediate_buffer: Deque[Any] = deque()
        self._next_buffer_idx: int = 0
        self._buffer_cond = threading.Condition()
        self._stop_event = threading.Event()
        self._prefetch_threads: List[threading.Thread] = []
        self._prefetch_client: Any = None
        self._prefetch_poll_interval: float = float(prefetch_poll_interval)
        self._state_lock = threading.Lock()  # guards quotas/starvation/attempted/served

    # Context manager helpers for clean shutdown
    def __enter__(self) -> "FairScheduler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------- Public API ----------------------------
    def fetch_next(self, client, timeout: float = 0.0) -> Optional[dict]:
        """
        Attempt to fetch the next job following fairness rules.
        Returns the first job found or None if no job is available across all queues for this cycle.
        """
        # Start prefetchers lazily
        if not self._prefetch_threads:
            with self._buffer_cond:
                if not self._prefetch_threads:
                    self._prefetch_client = client
                    for idx in range(self._num_prefetch_threads):
                        t = threading.Thread(target=self._prefetch_loop, args=(idx,), daemon=True)
                        t.start()
                        self._prefetch_threads.append(t)

        # Immediate (highest priority): return first available without blocking
        job = self._drain_immediate(client, timeout)
        if job is not None:
            self._logger.debug("FairScheduler.fetch_next: served from immediate")
            return job

        # Serve any prefetched immediate items before other buffered items
        with self._buffer_cond:
            if self._immediate_buffer:
                item = self._immediate_buffer.popleft()
                self._buffer_cond.notify_all()
                self._logger.debug("FairScheduler.fetch_next: served from immediate_buffer")
                return item

        # Do NOT serve from generic buffers here to preserve per-cycle ordering

        # Fair quotas across categories (non-blocking; skip empty)
        for category in ["micro", "small", "medium", "large", "default"]:
            remaining = self._cycle_quotas.get(category, 0)
            if remaining <= 0:
                continue
            # Try once for this category; if empty, move on (do not block)
            job = self._try_fetch_category(client, category, timeout)
            self._logger.debug("FairScheduler.fetch_next: poll category=%s -> %s", category, bool(job))
            if job is not None:
                return job

        # Starvation override (only when immediate is empty and quotas yielded nothing)
        starving_queue = self._choose_starving_queue()
        if starving_queue is not None:
            job = self._try_fetch_category(client, starving_queue, timeout)
            if job is not None:
                self._logger.debug("FairScheduler.fetch_next: served starving category=%s", starving_queue)
                return job

        # Final buffer check to avoid race with prefetchers filling after our scan
        with self._buffer_cond:
            if self._immediate_buffer:
                item = self._immediate_buffer.popleft()
                self._buffer_cond.notify_all()
                self._logger.debug("FairScheduler.fetch_next: served from immediate_buffer (post-scan)")
                return item

        # End of cycle – update starvation and reset quotas
        self._logger.debug("FairScheduler.fetch_next: end of cycle; no items across categories")
        with self._state_lock:
            self._end_cycle_update()
        return None

    # ---------------------------- Internal helpers ----------------------------
    def _drain_immediate(self, client, timeout: float) -> Optional[dict]:
        # Legacy helper retained for compatibility (unused in BLPOP path)
        try:
            job = client.fetch_message(self.queues["immediate"], timeout)
        except TimeoutError:
            job = None
        return job

    def _try_fetch_category(self, client, category: str, timeout: float) -> Optional[dict]:
        # Mark attempted this cycle
        if category in self._attempted_this_cycle:
            self._attempted_this_cycle[category] = True
        try:
            job = client.fetch_message(self.queues[category], timeout)
        except TimeoutError:
            job = None
        if job is not None:
            # Mark served and decrement quota
            if category in self._served_this_cycle:
                self._served_this_cycle[category] = True
            if category in self._cycle_quotas:
                self._cycle_quotas[category] = max(0, self._cycle_quotas[category] - 1)
            return job
        return None

    # ---------------------------- Prefetch thread ----------------------------
    def _prefetch_loop(self, buffer_index: int) -> None:
        client = self._prefetch_client
        idle_backoff: float = self._prefetch_poll_interval
        max_backoff: float = 0.05
        while not self._stop_event.is_set():
            # Wait if our buffer is at capacity
            with self._buffer_cond:
                if len(self._buffers[buffer_index]) >= self._per_thread_caps[buffer_index]:
                    self._buffer_cond.wait(timeout=idle_backoff)
                    continue

            # Try to fetch one item using non-blocking fairness logic
            job: Optional[dict] = None
            # We decide starvation/quotas under the state lock, but perform network calls outside it
            # Immediate first (not quota limited)
            try:
                job = client.fetch_message(self.queues["immediate"], 0.0)
            except TimeoutError:
                job = None
            if job is not None:
                with self._buffer_cond:
                    self._immediate_buffer.append(job)
                    self._buffer_cond.notify_all()
                idle_backoff = self._prefetch_poll_interval
                continue

            # Do not prefetch non-immediate categories to preserve cycle ordering/quotas
            with self._buffer_cond:
                self._buffer_cond.wait(timeout=idle_backoff)
            # Exponential-ish backoff up to max_backoff
            idle_backoff = min(max_backoff, idle_backoff * 1.5)

    def _prefetch_one_locked(self, client) -> Optional[dict]:
        """Attempt to fetch a single job under the buffer lock, updating quotas/state.
        Non-blocking; returns None if no job found across all categories for this pass.
        """
        # Immediate first
        try:
            job = client.fetch_message(self.queues["immediate"], 0.0)
        except TimeoutError:
            job = None
        if job is not None:
            # immediate is not quota-limited; mark attempted/served bookkeeping unaffected
            return job

        # Starvation override
        starving_queue = self._choose_starving_queue()
        if starving_queue is not None and self._cycle_quotas.get(starving_queue, 0) > 0:
            if starving_queue in self._attempted_this_cycle:
                self._attempted_this_cycle[starving_queue] = True
            try:
                job = client.fetch_message(self.queues[starving_queue], 0.0)
            except TimeoutError:
                job = None
            if job is not None:
                if starving_queue in self._served_this_cycle:
                    self._served_this_cycle[starving_queue] = True
                if starving_queue in self._cycle_quotas:
                    self._cycle_quotas[starving_queue] = max(0, self._cycle_quotas[starving_queue] - 1)
                return job

        # Quotas order
        for category in ["micro", "small", "medium", "large", "default"]:
            if self._cycle_quotas.get(category, 0) <= 0:
                continue
            if category in self._attempted_this_cycle:
                self._attempted_this_cycle[category] = True
            try:
                job = client.fetch_message(self.queues[category], 0.0)
            except TimeoutError:
                job = None
            if job is not None:
                if category in self._served_this_cycle:
                    self._served_this_cycle[category] = True
                if category in self._cycle_quotas:
                    self._cycle_quotas[category] = max(0, self._cycle_quotas[category] - 1)
                return job

        return None

    def stop_prefetch(self) -> None:
        self._stop_event.set()
        with self._buffer_cond:
            self._buffer_cond.notify_all()
        for t in self._prefetch_threads:
            t.join(timeout=1.0)
        self._prefetch_threads.clear()

    def close(self) -> None:
        """Cleanly stop background prefetch threads."""
        self.stop_prefetch()

    def _choose_starving_queue(self) -> Optional[str]:
        # Find any category with starved_cycles >= threshold
        candidates: List[Tuple[str, int]] = [
            (cat, cycles) for cat, cycles in self._starved_cycles.items() if cycles >= self.starvation_cycles
        ]
        if not candidates:
            return None
        # Prefer higher priority categories first
        priority_order = ["micro", "small", "medium", "large", "default"]
        candidates.sort(key=lambda x: priority_order.index(x[0]))
        return candidates[0][0]

    # Removed buffer-based fill; fetch_next now directly uses BLPOP or per-queue polling

    def _end_cycle_update(self) -> None:
        # Update starvation counters:
        # - Queues served this cycle: reset to 0
        # - Queues attempted but empty: reset to 0 (empty queues do not starve)
        # - Queues not attempted: increment by 1 (they can starve)
        for cat in self._cycle_quotas_default.keys():
            if self._served_this_cycle.get(cat, False):
                self._starved_cycles[cat] = 0
            elif self._attempted_this_cycle.get(cat, False):
                self._starved_cycles[cat] = 0
            else:
                self._starved_cycles[cat] += 1

        # Reset per-cycle state
        self._cycle_quotas = dict(self._cycle_quotas_default)
        for cat in self._cycle_quotas_default.keys():
            self._attempted_this_cycle[cat] = False
            self._served_this_cycle[cat] = False

    # Expose for testing/introspection
    def get_starved_cycles(self) -> Dict[str, int]:
        return dict(self._starved_cycles)

    def get_cycle_quotas(self) -> Dict[str, int]:
        return dict(self._cycle_quotas)
