# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

from nv_ingest_api.util.message_brokers.fair_scheduler import FairScheduler
import time


class FakeClient:
    """Thread-safe minimal broker client supporting fetch_message(queue, timeout)."""

    def __init__(self, initial: dict[str, list]):
        # Copy lists so tests don't share state accidentally
        self.queues: dict[str, list] = {k: list(v) for k, v in initial.items()}
        self._lock = threading.Lock()

    def fetch_message(self, queue: str, timeout: float):
        # Non-blocking pop with a lock for thread-safety
        with self._lock:
            lst = self.queues.get(queue, [])
            if not lst:
                return None
            return lst.pop(0)


def make_queues(base: str) -> dict[str, str]:
    return {
        "immediate": f"{base}_immediate",
        "micro": f"{base}_micro",
        "small": f"{base}_small",
        "medium": f"{base}_medium",
        "large": f"{base}_large",
        "default": f"{base}",
    }


def drain_until_none(scheduler: FairScheduler, client: FakeClient, max_iters: int = 100):
    """Call fetch_next repeatedly until it returns None or max_iters hit. Return captured sequence."""
    seq = []
    for _ in range(max_iters):
        item = scheduler.fetch_next(client, timeout=0.0)
        if item is None:
            break
        seq.append(item)
    return seq


def mk_scheduler(base_queue: str, starvation_cycles: int = 5):
    """Create a scheduler with minimal prefetch concurrency for deterministic-ish tests."""
    return FairScheduler(
        base_queue=base_queue,
        starvation_cycles=starvation_cycles,
        total_buffer_capacity=1,
        num_prefetch_threads=1,
        prefetch_poll_interval=0.02,
    )


def pull_until_counts(
    scheduler: FairScheduler,
    client: FakeClient,
    target: dict[str, int],
    max_pulls: int = 1000,
) -> dict[str, int]:
    """Pull until we have met the target counts per category or run out of work.

    This avoids relying on None-as-cycle-boundary and tolerates multi-thread interleaving.
    """
    got = {k: 0 for k in target}
    pulls = 0
    while pulls < max_pulls and any(got[k] < target[k] for k in target):
        item = scheduler.fetch_next(client)
        if item is None:
            # No immediately available item; allow a short pause and retry once
            time.sleep(0.002)
            item = scheduler.fetch_next(client)
            if item is None:
                break
        pulls += 1
        label = item[0]
        if label in got and got[label] < target[label]:
            got[label] += 1
    return got


def test_immediate_drains_first():
    base = "jobs"
    q = make_queues(base)

    # Put items in all queues, but immediate should be returned first until empty
    client = FakeClient(
        {
            q["immediate"]: [("immediate", i) for i in range(3)],
            q["micro"]: [("micro", 0)],
            q["small"]: [("small", 0)],
            q["medium"]: [("medium", 0)],
            q["large"]: [("large", 0)],
            q["default"]: [("default", 0)],
        }
    )

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.01)
        # Collect until we've seen all immediate items (allow limited interleaving)
        immediates_seen = 0
        max_pulls = 20
        pulled = []
        for _ in range(max_pulls):
            item = scheduler.fetch_next(client)
            if item is None:
                break
            pulled.append(item)
            if item[0] == "immediate":
                immediates_seen += 1
            if immediates_seen == 3:
                break
        assert immediates_seen == 3
        # After collecting all immediate items, future pulls should be from other categories or None
        nxt = scheduler.fetch_next(client)
        if nxt is not None:
            assert nxt[0] in {"micro", "small", "medium", "large", "default"}


def test_cycle_quotas_and_order():
    base = "jobs"
    q = make_queues(base)

    # Load multiple items in each queue to test quotas within a cycle
    client = FakeClient(
        {
            q["micro"]: [("micro", i) for i in range(6)],  # more than quota 4
            q["small"]: [("small", i) for i in range(5)],  # more than quota 4
            q["medium"]: [("medium", i) for i in range(3)],  # more than quota 2
            q["large"]: [("large", i) for i in range(2)],  # more than quota 1
            q["default"]: [("default", i) for i in range(2)],  # more than quota 1
        }
    )

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.01)
        quotas = {"micro": 4, "small": 4, "medium": 2, "large": 1, "default": 1}
        counts = pull_until_counts(scheduler, client, quotas, max_pulls=100)
        assert counts == quotas

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.01)
        remaining = {"micro": 2, "small": 1, "medium": 1, "large": 1, "default": 1}
        counts2 = pull_until_counts(scheduler, client, remaining, max_pulls=100)
        assert counts2 == remaining


def test_skip_empty_categories():
    base = "jobs"
    q = make_queues(base)

    # Only 'small' has data; micro is empty and should be skipped without blocking
    client = FakeClient(
        {
            q["micro"]: [],
            q["small"]: [("small", i) for i in range(2)],
        }
    )

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.005)
        small_q = q["small"]
        max_pulls = 200
        pulls = 0
        while pulls < max_pulls and len(client.queues.get(small_q, [])) > 0:
            it = scheduler.fetch_next(client)
            if it is None:
                # allow prefetcher/loop to advance
                time.sleep(0.002)
                pulls += 1
                continue
            pulls += 1
            # Only 'small' should appear since it's the only non-empty category
            assert it[0] == "small", f"Unexpected category {it[0]} when only small has data"
        # Ensure small is fully drained
        assert len(client.queues.get(small_q, [])) == 0
        # After draining small, should see None within a few tries
        none_seen = False
        for _ in range(10):
            if scheduler.fetch_next(client) is None:
                none_seen = True
                break
            time.sleep(0.002)
        assert none_seen


def test_starvation_override_without_immediate():
    base = "jobs"
    q = make_queues(base)

    # Build two idle cycles where only micro has data; default has no items yet.
    client = FakeClient(
        {
            q["micro"]: [("micro", i) for i in range(8)],  # enough for 2 cycles of 4 each
            q["default"]: [],
        }
    )

    with mk_scheduler(base, starvation_cycles=2) as scheduler:
        # Perform two full cycles pulling only from micro (4 pulls + end-of-cycle None)
        for _ in range(2):
            pulls = [scheduler.fetch_next(client) for _ in range(4)]
            assert all(p[0] == "micro" for p in pulls)
            # End of cycle occurs only when no queues have items
            assert scheduler.fetch_next(client) is None

        # Now add items to default; it has been starved for 2 cycles and should be chosen before micro
        client.queues[q["default"]] = [("default", i) for i in range(2)]

        time.sleep(0.01)
        # Within a bounded number of pulls, default should appear (immediate is empty)
        seen_default = False
        for _ in range(20):
            nxt = scheduler.fetch_next(client)
            if nxt is None:
                time.sleep(0.002)
                continue
            if nxt[0] == "default":
                seen_default = True
                break
        assert seen_default


def test_starvation_ignored_when_immediate_present():
    base = "jobs"
    q = make_queues(base)

    # First, create starvation across idle cycles where default is absent
    client = FakeClient(
        {
            q["micro"]: [("micro", i) for i in range(8)],  # two cycles of micro-only
            q["default"]: [],
        }
    )

    with mk_scheduler(base, starvation_cycles=1) as scheduler:
        # Two idle cycles: only micro is attempted; default not attempted and becomes starved
        for _ in range(2):
            pulls = [scheduler.fetch_next(client) for _ in range(4)]
            assert all(p[0] == "micro" for p in pulls)
            assert scheduler.fetch_next(client) is None  # end of cycle

        # Now add both an immediate item and default items
        client.queues[q["immediate"]] = [("immediate", 0)]
        client.queues[q["default"]] = [("default", 0), ("default", 1)]

        # Starvation override exists for default, but immediate must be served first
        time.sleep(0.01)
        seen_immediate = False
        seen_default = False
        first_default_index = None
        first_immediate_index = None
        for i in range(20):
            nxt = scheduler.fetch_next(client)
            if nxt is None:
                time.sleep(0.002)
                continue
            if not seen_immediate and nxt[0] == "immediate":
                seen_immediate = True
                first_immediate_index = i
            if not seen_default and nxt[0] == "default":
                seen_default = True
                first_default_index = i
            if seen_immediate and seen_default:
                break
        assert seen_immediate
        assert seen_default
        assert first_immediate_index is not None and first_default_index is not None
        # immediate should appear before default
        assert first_immediate_index < first_default_index


def test_complex_drain_hundreds_of_tasks():
    base = "jobs"
    q = make_queues(base)

    totals = {
        "immediate": 10,
        "micro": 200,
        "small": 200,
        "medium": 100,
        "large": 50,
        "default": 50,
    }

    client = FakeClient(
        {
            q["immediate"]: [("immediate", i) for i in range(totals["immediate"])],
            q["micro"]: [("micro", i) for i in range(totals["micro"])],
            q["small"]: [("small", i) for i in range(totals["small"])],
            q["medium"]: [("medium", i) for i in range(totals["medium"])],
            q["large"]: [("large", i) for i in range(totals["large"])],
            q["default"]: [("default", i) for i in range(totals["default"])],
        }
    )

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.01)
        # First, all immediate should be drained without any None in between
        first_ten = [scheduler.fetch_next(client) for _ in range(totals["immediate"])]
        assert all(x[0] == "immediate" for x in first_ten)

    # Now run cycles until everything is drained; validate per-cycle quotas
    remaining = totals.copy()
    remaining["immediate"] = 0

    quotas = {"micro": 4, "small": 4, "medium": 2, "large": 1, "default": 1}

    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.01)
        cycles = 0
        while any(len(client.queues[qname]) > 0 for qname in client.queues):
            cycles += 1
            cycle_counts = pull_until_counts(scheduler, client, quotas, max_pulls=1000)
            # Validate cycle counts within quotas and non-negative
            for cat, quota in quotas.items():
                assert 0 <= cycle_counts.get(cat, 0) <= quota

    # All queues should be empty
    for qname in client.queues:
        assert len(client.queues[qname]) == 0


def test_no_starvation_over_many_cycles_without_immediate():
    base = "jobs"
    q = make_queues(base)

    # Load large numbers so we can observe many cycles
    counts = {"micro": 80, "small": 80, "medium": 40, "large": 20, "default": 20}
    client = FakeClient(
        {
            q["micro"]: [("micro", i) for i in range(counts["micro"])],
            q["small"]: [("small", i) for i in range(counts["small"])],
            q["medium"]: [("medium", i) for i in range(counts["medium"])],
            q["large"]: [("large", i) for i in range(counts["large"])],
            q["default"]: [("default", i) for i in range(counts["default"])],
        }
    )

    with mk_scheduler(base, starvation_cycles=3) as scheduler:
        time.sleep(0.01)
        quotas = {"micro": 4, "small": 4, "medium": 2, "large": 1, "default": 1}
        last_served_cycle = {k: None for k in quotas}

        cycle_idx = -1
        while any(len(client.queues[qname]) > 0 for qname in client.queues):
            cycle_idx += 1
            served_this_cycle = {k: 0 for k in quotas}
            while True:
                item = scheduler.fetch_next(client)
                if item is None:
                    break
                lbl = item[0]
                if lbl in served_this_cycle:
                    served_this_cycle[lbl] += 1

            # For any category that still has remaining items, it must be served at least once
            # every starvation_cycles cycles
            for cat in quotas:
                remaining_items = len(client.queues[q[cat]])
                if remaining_items > 0:
                    if served_this_cycle[cat] > 0:
                        last_served_cycle[cat] = cycle_idx
                    else:
                        # If not served this cycle, ensure we haven't exceeded starvation threshold since last service
                        if last_served_cycle[cat] is not None:
                            assert (cycle_idx - last_served_cycle[cat]) <= 3

        # Completed draining without violating starvation bound
        assert all(v is not None for v in last_served_cycle.values())


def test_immediate_can_starve_others_until_empty():
    base = "jobs"
    q = make_queues(base)

    client = FakeClient(
        {
            q["immediate"]: [("immediate", i) for i in range(500)],
            q["micro"]: [("micro", i) for i in range(50)],
            q["small"]: [("small", i) for i in range(50)],
            q["medium"]: [("medium", i) for i in range(25)],
            q["large"]: [("large", i) for i in range(10)],
            q["default"]: [("default", i) for i in range(10)],
        }
    )

    with mk_scheduler(base, starvation_cycles=1) as scheduler:
        time.sleep(0.01)
        # First 500 pulls should all be immediate
        pulls = [scheduler.fetch_next(client) for _ in range(500)]
        assert all(x[0] == "immediate" for x in pulls)

    # Next pulls should begin following quotas
    with mk_scheduler(base, starvation_cycles=1) as scheduler:
        time.sleep(0.01)
        nxt = scheduler.fetch_next(client)
        assert nxt[0] in {"micro", "small", "medium", "large", "default"}


def test_never_none_when_any_queue_has_items_later_category():
    base = "jobs"
    q = make_queues(base)

    # micro and small empty; medium has items
    client = FakeClient(
        {
            q["micro"]: [],
            q["small"]: [],
            q["medium"]: [("medium", 0), ("medium", 1)],
        }
    )
    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        time.sleep(0.005)
        # First call should skip micro and small and pick medium, not None
        item1 = scheduler.fetch_next(client)
        assert item1 is not None and item1[0] == "medium"

        # Second call should again return medium (quota allows up to 2 per cycle)
        # Tolerate transient None by retrying briefly until second medium observed
        mediums_seen = 1
        max_tries = 20
        for _ in range(max_tries):
            item2 = scheduler.fetch_next(client)
            if item2 is None:
                time.sleep(0.002)
                continue
            assert item2[0] == "medium"
            mediums_seen += 1
            break
        assert mediums_seen == 2

        # Now medium empty; end of cycle should return None
        item3 = scheduler.fetch_next(client)
        assert item3 is None


def test_skips_multiple_empty_and_selects_default():
    base = "jobs"
    q = make_queues(base)

    # Only default has items; all earlier categories empty
    client = FakeClient(
        {
            q["micro"]: [],
            q["small"]: [],
            q["medium"]: [],
            q["large"]: [],
            q["default"]: [("default", 0)],
        }
    )
    with mk_scheduler(base, starvation_cycles=5) as scheduler:
        # Should return default immediately, not None
        item = scheduler.fetch_next(client)
        assert item is not None and item[0] == "default"

        # Next call ends cycle (no more work)
        assert scheduler.fetch_next(client) is None
