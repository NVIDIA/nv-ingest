# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Optional
import logging
import time
import random


class _SchedulingStrategy:
    """
    Base scheduling strategy interface. Implementations must provide a non-blocking
    single-sweep attempt over non-immediate queues and return a job or None.
    """

    def try_once(self, client, queues: Dict[str, str], order: list[str]) -> Optional[dict]:
        raise NotImplementedError


class _LotteryStrategy(_SchedulingStrategy):
    """
    Lottery scheduling with fixed weights.
    Weights: micro=4, small=2, large=1, medium=1, default=1
    """

    def __init__(self, prioritize_immediate: bool = True) -> None:
        self._weights: Dict[str, int] = {
            "micro": 4,
            "small": 2,
            "large": 1,
            "medium": 1,
            "default": 1,
        }
        self._prioritize_immediate: bool = bool(prioritize_immediate)

    def try_once(self, client, queues: Dict[str, str], order: list[str]) -> Optional[dict]:
        # Immediate-first if enabled (non-blocking)
        if self._prioritize_immediate:
            try:
                job = client.fetch_message(queues["immediate"], 0)
                if job is not None:
                    return job
            except TimeoutError:
                pass
        candidates = list(order)
        weights = [self._weights[q] for q in candidates]
        while candidates:
            try:
                chosen = random.choices(candidates, weights=weights, k=1)[0]
                job = client.fetch_message(queues[chosen], 0)
                if job is not None:
                    return job
            except TimeoutError:
                pass
            finally:
                idx = candidates.index(chosen)
                del candidates[idx]
                del weights[idx]
        return None


class _SimpleStrategy(_SchedulingStrategy):
    """
    Simple strategy placeholder. Actual simple-mode handling is done in QosScheduler.fetch_next
    to directly fetch from the base 'default' queue using the provided timeout.
    """

    def try_once(self, client, queues: Dict[str, str], order: list[str]) -> Optional[dict]:
        # Block up to 30s on the base/default queue and return first available job
        try:
            return client.fetch_message(queues["default"], 30.0)
        except TimeoutError:
            return None


class _RoundRobinStrategy(_SchedulingStrategy):
    """
    Simple round-robin over non-immediate queues. Maintains rotation across calls.
    """

    def __init__(self, order: list[str], prioritize_immediate: bool = True) -> None:
        self._order = list(order)
        self._len = len(self._order)
        self._idx = 0
        self._prioritize_immediate: bool = bool(prioritize_immediate)

    def try_once(self, client, queues: Dict[str, str], order: list[str]) -> Optional[dict]:
        # Immediate-first if enabled (non-blocking)
        if self._prioritize_immediate:
            try:
                job = client.fetch_message(queues["immediate"], 0)
                if job is not None:
                    return job
            except TimeoutError:
                pass
        start_idx = self._idx
        for step in range(self._len):
            i = (start_idx + step) % self._len
            qname = self._order[i]
            try:
                job = client.fetch_message(queues[qname], 0)
                if job is not None:
                    # advance rotation to the position after the chosen one
                    self._idx = (i + 1) % self._len
                    return job
            except TimeoutError:
                continue
        return None


class _WeightedRoundRobinStrategy(_SchedulingStrategy):
    """
    Smooth Weighted Round Robin (SWRR) using weights micro=4, small=2, large=1, medium=1, default=1.
    Maintains current weights across calls.
    """

    def __init__(self, prioritize_immediate: bool = True) -> None:
        self._weights: Dict[str, int] = {
            "micro": 4,
            "small": 2,
            "large": 1,
            "medium": 1,
            "default": 1,
        }
        self._current: Dict[str, int] = {k: 0 for k in self._weights.keys()}
        self._total: int = sum(self._weights.values())
        self._prioritize_immediate: bool = bool(prioritize_immediate)

    def try_once(self, client, queues: Dict[str, str], order: list[str]) -> Optional[dict]:
        # Immediate-first if enabled (non-blocking)
        if self._prioritize_immediate:
            try:
                job = client.fetch_message(queues["immediate"], 0)
                if job is not None:
                    return job
            except TimeoutError:
                pass
        # Attempt up to len(order) selections per sweep, excluding queues that prove empty
        active = list(order)
        for _ in range(len(order)):
            if not active:
                break
            for q in active:
                self._current[q] += self._weights[q]
            chosen = max(active, key=lambda q: self._current[q])
            self._current[chosen] -= self._total
            try:
                job = client.fetch_message(queues[chosen], 0)
                if job is not None:
                    return job
            except TimeoutError:
                job = None
            # If no job available from chosen, exclude it for the remainder of this sweep
            if job is None and chosen in active:
                active.remove(chosen)
        # Fallback: single non-blocking attempt for each queue in order
        for q in order:
            try:
                job = client.fetch_message(queues[q], 0)
                if job is not None:
                    return job
            except TimeoutError:
                continue
        return None


class QosScheduler:
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
        strategy: str = "lottery",
        prioritize_immediate: bool = True,
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

        # Non-immediate queue order reference
        self._non_immediate_order = ["micro", "small", "large", "medium", "default"]

        # Logger
        self._logger = logging.getLogger(__name__)

        # No prefetching - just direct calls
        self._total_buffer_capacity: int = int(total_buffer_capacity)
        self._num_prefetch_threads: int = int(num_prefetch_threads)
        self._prefetch_poll_interval: float = float(prefetch_poll_interval)
        self._prefetch_non_immediate: bool = bool(prefetch_non_immediate)

        # Strategy selection
        self._simple_mode: bool = False
        if strategy == "simple":
            self._strategy_impl: _SchedulingStrategy = _SimpleStrategy()
            self._simple_mode = True
        elif strategy == "round_robin":
            self._strategy_impl = _RoundRobinStrategy(self._non_immediate_order, prioritize_immediate)
        elif strategy == "weighted_round_robin":
            self._strategy_impl = _WeightedRoundRobinStrategy(prioritize_immediate)
        else:
            self._strategy_impl = _LotteryStrategy(prioritize_immediate)

    # Context manager helpers for clean shutdown
    def __enter__(self) -> "QosScheduler":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ---------------------------- Public API ----------------------------
    def close(self) -> None:
        """
        Cleanly close the scheduler. No-op for the current implementation
        since we do not spin background threads.
        """
        return None

    def fetch_next(self, client, timeout: float = 0.0) -> Optional[dict]:
        """
        Immediate-first, then strategy-based scheduling among non-immediate queues.

        Behavior:
        - Always check 'immediate' first (non-blocking). If present, return immediately.
        - If not, select using the configured strategy (lottery, round_robin, weighted_round_robin).
        - If no job is found in a full pass:
          - If timeout <= 0: return None.
          - Else: sleep in 0.5s increments and retry until accumulated elapsed time >= timeout.
        """
        # Simple mode: delegate to the strategy (blocks up to 30s on base queue)
        if getattr(self, "_simple_mode", False):
            return self._strategy_impl.try_once(client, self.queues, self._non_immediate_order)

        start = time.monotonic()
        while True:
            # Strategy-based attempt (strategy may include immediate priority internally)
            job = self._strategy_impl.try_once(client, self.queues, self._non_immediate_order)
            if job is not None:
                return job

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
