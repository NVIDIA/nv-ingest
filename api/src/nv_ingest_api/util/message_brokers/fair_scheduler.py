# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


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

    def __init__(self, base_queue: str, starvation_cycles: int = 10) -> None:
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

    # ---------------------------- Public API ----------------------------
    def fetch_next(self, client, timeout: float = 0.0) -> Optional[dict]:
        """
        Attempt to fetch the next job following fairness rules.
        Returns the first job found or None if no job is available across all queues for this cycle.
        """
        # 1) Drain immediate
        job = self._drain_immediate(client, timeout)
        if job is not None:
            return job

        # 2) Starvation override (only when immediate is empty)
        starving_queue = self._choose_starving_queue()
        if starving_queue is not None:
            job = self._try_fetch_category(client, starving_queue, timeout)
            if job is not None:
                return job

        # 3) Fair quotas
        for category in ["micro", "small", "medium", "large", "default"]:
            remaining = self._cycle_quotas.get(category, 0)
            if remaining <= 0:
                continue

            # Try up to remaining pulls for this category
            while self._cycle_quotas[category] > 0:
                job = self._try_fetch_category(client, category, timeout)
                if job is not None:
                    return job
                # If empty, stop trying this category for now (skip if empty)
                break

        # 4) End of cycle – update starvation and reset quotas
        self._end_cycle_update()
        return None

    # ---------------------------- Internal helpers ----------------------------
    def _drain_immediate(self, client, timeout: float) -> Optional[dict]:
        while True:
            try:
                job = client.fetch_message(self.queues["immediate"], timeout)
            except TimeoutError:
                job = None
            if job is None:
                return None
            return job  # return one-at-a-time to caller; repeated calls will continue draining

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
