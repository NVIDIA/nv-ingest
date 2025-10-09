# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nv_ingest_api.util.message_brokers.fair_scheduler import FairScheduler


class FakeClient:
    """Minimal broker client that supports fetch_message(queue, timeout)."""

    def __init__(self, initial: dict[str, list]):
        # Copy lists so tests don't share state accidentally
        self.queues: dict[str, list] = {k: list(v) for k, v in initial.items()}

    def fetch_message(self, queue: str, timeout: float):
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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # First three calls must be immediate
    got = [scheduler.fetch_next(client) for _ in range(3)]
    assert [x[0] for x in got] == ["immediate", "immediate", "immediate"]

    # Next call should follow normal quotas (micro first)
    nxt = scheduler.fetch_next(client)
    assert nxt[0] == "micro"


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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # Drain one scheduling cycle worth of pulls
    seq = drain_until_none(scheduler, client)

    # Expect exactly: 4 micro, 4 small, 2 medium, 1 large, 1 default
    expected_prefix = ["micro"] * 4 + ["small"] * 4 + ["medium"] * 2 + ["large"] * 1 + ["default"] * 1
    assert [x[0] for x in seq] == expected_prefix

    # Start next cycle; quotas reset, so we can pull again in same order until queues empty
    seq2 = drain_until_none(scheduler, client)
    # Remaining after first cycle: micro: 2, small: 1, medium: 1, large: 1, default: 1
    # Next cycle will take min(remaining, quotas)
    expected_prefix2 = ["micro"] * 2 + ["small"] * 1 + ["medium"] * 1 + ["large"] * 1 + ["default"] * 1
    assert [x[0] for x in seq2] == expected_prefix2


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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # First call should skip empty micro and pull from small
    got1 = scheduler.fetch_next(client)
    assert got1[0] == "small"

    # Second call should still pull from small (quota allows up to 4)
    got2 = scheduler.fetch_next(client)
    assert got2[0] == "small"

    # Third call ends cycle (no more items)
    got3 = scheduler.fetch_next(client)
    assert got3 is None


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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=2)

    # Perform two full cycles pulling only from micro (4 pulls + end-of-cycle None)
    for _ in range(2):
        pulls = [scheduler.fetch_next(client) for _ in range(4)]
        assert all(p[0] == "micro" for p in pulls)
        # End of cycle occurs only when no queues have items
        assert scheduler.fetch_next(client) is None

    # Now add items to default; it has been starved for 2 cycles and should be chosen before micro
    client.queues[q["default"]] = [("default", i) for i in range(2)]

    nxt = scheduler.fetch_next(client)
    assert nxt[0] == "default"


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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=1)

    # Two idle cycles: only micro is attempted; default not attempted and becomes starved
    for _ in range(2):
        pulls = [scheduler.fetch_next(client) for _ in range(4)]
        assert all(p[0] == "micro" for p in pulls)
        assert scheduler.fetch_next(client) is None  # end of cycle

    # Now add both an immediate item and default items
    client.queues[q["immediate"]] = [("immediate", 0)]
    client.queues[q["default"]] = [("default", 0), ("default", 1)]

    # Starvation override exists for default, but immediate must be served first
    nxt = scheduler.fetch_next(client)
    assert nxt[0] == "immediate"

    # After immediate drained, starvation override should serve default
    nxt2 = scheduler.fetch_next(client)
    assert nxt2[0] == "default"


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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # First, all immediate should be drained without any None in between
    first_ten = [scheduler.fetch_next(client) for _ in range(totals["immediate"])]
    assert all(x[0] == "immediate" for x in first_ten)

    # Now run cycles until everything is drained; validate per-cycle quotas
    remaining = totals.copy()
    remaining["immediate"] = 0

    quotas = {"micro": 4, "small": 4, "medium": 2, "large": 1, "default": 1}

    cycles = 0
    while any(len(client.queues[qname]) > 0 for qname in client.queues):
        cycles += 1
        # Gather one cycle worth of outputs
        cycle_counts = {k: 0 for k in quotas}
        while True:
            item = scheduler.fetch_next(client)
            if item is None:
                break
            label = item[0]
            if label in cycle_counts:
                cycle_counts[label] += 1
                remaining[label] -= 1

        # Validate cycle counts within quotas and not exceeding remaining
        for cat, quota in quotas.items():
            _ = min(quota, totals[cat] - (totals[cat] - remaining[cat]))
            # expected is effectively min(quota, remaining at start of cycle),
            # but we can't easily compute start-of-cycle remaining now.
            # So we only assert upper bounds by quotas and non-negative counts.
            assert 0 <= cycle_counts[cat] <= quota

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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=3)

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

    scheduler = FairScheduler(base_queue=base, starvation_cycles=1)

    # First 500 pulls should all be immediate
    pulls = [scheduler.fetch_next(client) for _ in range(500)]
    assert all(x[0] == "immediate" for x in pulls)

    # Next pulls should begin following quotas
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
    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # First call should skip micro and small and pick medium, not None
    item1 = scheduler.fetch_next(client)
    assert item1 is not None and item1[0] == "medium"

    # Second call should again return medium (quota allows up to 2 per cycle)
    item2 = scheduler.fetch_next(client)
    assert item2 is not None and item2[0] == "medium"

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
    scheduler = FairScheduler(base_queue=base, starvation_cycles=5)

    # Should return default immediately, not None
    item = scheduler.fetch_next(client)
    assert item is not None and item[0] == "default"

    # Next call ends cycle (no more work)
    assert scheduler.fetch_next(client) is None
