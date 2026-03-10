# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

from nv_ingest_api.util.message_brokers.qos_scheduler import QosScheduler


class FakeClient:
    """
    Thread-safe minimal broker client supporting fetch_message(queue, timeout).
    Implements blocking semantics using a Condition to allow testing of simple strategy.
    """

    def __init__(self, initial: dict[str, list] | None = None):
        self.queues: dict[str, list] = {k: list(v) for k, v in (initial or {}).items()}
        self._cv = threading.Condition()

    def fetch_message(self, queue: str, timeout: float):
        deadline = time.time() + max(0.0, float(timeout))
        with self._cv:
            while True:
                lst = self.queues.setdefault(queue, [])
                if lst:
                    return lst.pop(0)
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    return None
                self._cv.wait(timeout=remaining)

    def put_message(self, queue: str, item):
        with self._cv:
            self.queues.setdefault(queue, []).append(item)
            self._cv.notify_all()


def make_queues(base: str) -> dict[str, str]:
    return {
        "immediate": f"{base}_immediate",
        "micro": f"{base}_micro",
        "small": f"{base}_small",
        "medium": f"{base}_medium",
        "large": f"{base}_large",
        "default": f"{base}",
    }


def mk_scheduler(base_queue: str, strategy: str = "lottery", prioritize_immediate: bool = True) -> QosScheduler:
    return QosScheduler(
        base_queue=base_queue,
        total_buffer_capacity=1,
        num_prefetch_threads=0,
        prefetch_poll_interval=0.0,
        prefetch_non_immediate=False,
        strategy=strategy,
        prioritize_immediate=prioritize_immediate,
    )


def test_simple_strategy_uses_default_and_ignores_immediate():
    base = "jobs"
    q = make_queues(base)

    client = FakeClient(
        {
            q["immediate"]: [("immediate", 0)],
            q["default"]: [("default", 42)],
        }
    )

    with mk_scheduler(base, strategy="simple") as scheduler:
        item = scheduler.fetch_next(client)
        assert item is not None and item[0] == "default"


def test_simple_strategy_blocks_until_item_arrives():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({})

    def producer():
        time.sleep(0.05)
        client.put_message(q["default"], ("default", 1))

    with mk_scheduler(base, strategy="simple") as scheduler:
        t = threading.Thread(target=producer, daemon=True)
        t.start()
        t0 = time.time()
        item = scheduler.fetch_next(client)
        dt = time.time() - t0
        assert item is not None and item[0] == "default"
        assert dt >= 0.05


def test_nonblocking_returns_none():
    base = "jobs"
    client = FakeClient({})
    with mk_scheduler(base, strategy="lottery") as scheduler:
        assert scheduler.fetch_next(client, timeout=0.0) is None


def test_round_robin_smoke_nonblocking():
    # Not asserting strict order; just that it returns from available queues and never blocks with timeout=0.0
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({q["small"]: [("small", 1)], q["micro"]: [("micro", 1)]})
    with mk_scheduler(base, strategy="round_robin") as scheduler:
        a = scheduler.fetch_next(client, timeout=0.0)
        b = scheduler.fetch_next(client, timeout=0.0)
        assert a is not None and b is not None
        assert {a[0], b[0]} == {"small", "micro"}


def test_weighted_round_robin_smoke_nonblocking():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({q["micro"]: [("micro", 1)], q["default"]: [("default", 1)]})
    with mk_scheduler(base, strategy="weighted_round_robin") as scheduler:
        a = scheduler.fetch_next(client, timeout=0.0)
        b = scheduler.fetch_next(client, timeout=0.0)
        assert a is not None and b is not None
        assert {a[0], b[0]} == {"micro", "default"}

    # (Complex quota/cycle tests removed; no starvation/quotas in current scheduler)


def test_immediate_arrival_is_seen_quickly_when_prioritized():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({q["micro"]: [("micro", 1), ("micro", 2)]})

    with mk_scheduler(base, strategy="lottery", prioritize_immediate=True) as scheduler:
        # First pull from non-immediate
        first = scheduler.fetch_next(client, timeout=0.0)
        assert first is not None and first[0] in {"micro"}

        # Now an immediate job arrives
        client.put_message(q["immediate"], ("immediate", 99))

        # Repeated non-blocking pulls should see immediate within a few tries
        seen_immediate = False
        for _ in range(10):
            nxt = scheduler.fetch_next(client, timeout=0.0)
            if nxt is not None and nxt[0] == "immediate":
                seen_immediate = True
                break
        assert seen_immediate


def test_large_immediate_backlog_served_first_when_prioritized():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient(
        {
            q["immediate"]: [("immediate", i) for i in range(50)],
            q["small"]: [("small", 0)],
        }
    )
    with mk_scheduler(base, strategy="lottery", prioritize_immediate=True) as scheduler:
        pulls = [scheduler.fetch_next(client, timeout=0.0) for _ in range(50)]
        assert all(p is not None and p[0] == "immediate" for p in pulls)


def test_skips_empty_and_returns_available_non_immediate():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({q["micro"]: [], q["small"]: [("small", 1)]})
    with mk_scheduler(base, strategy="round_robin") as scheduler:
        item = scheduler.fetch_next(client, timeout=0.0)
        assert item is not None and item[0] == "small"


def test_selects_default_when_others_empty():
    base = "jobs"
    q = make_queues(base)
    client = FakeClient({q["default"]: [("default", 0)]})
    with mk_scheduler(base, strategy="lottery") as scheduler:
        item = scheduler.fetch_next(client, timeout=0.0)
        assert item is not None and item[0] == "default"


def test_constructor_params_and_close_idempotent():
    base = "jobs"
    with QosScheduler(
        base_queue=base,
        total_buffer_capacity=7,
        num_prefetch_threads=3,
        prefetch_poll_interval=0.01,
        strategy="round_robin",
        prioritize_immediate=True,
    ) as scheduler:
        assert scheduler._total_buffer_capacity == 7
        assert scheduler._num_prefetch_threads == 3

    fs = QosScheduler(base_queue=base)
    fs.close()
    fs.close()


def test_immediate_arrival_preempts_within_bounded_pulls():
    base = "jobs"
    q = make_queues(base)
    # Start with micro items only
    client = FakeClient({q["micro"]: [("micro", i) for i in range(8)]})

    with mk_scheduler(base, strategy="lottery", prioritize_immediate=True) as scheduler:
        # Pull one micro first
        first = scheduler.fetch_next(client, timeout=0.0)
        assert first and first[0] == "micro"

        # Now, an immediate job arrives
        client.put_message(q["immediate"], ("immediate", 999))

        # Within bounded non-blocking pulls, we should observe the immediate item
        seen_immediate = False
        for _ in range(20):
            nxt = scheduler.fetch_next(client, timeout=0.0)
            if nxt is not None and nxt[0] == "immediate":
                seen_immediate = True
                break
        assert seen_immediate


def test_returns_none_when_all_empty():
    base = "jobs"
    client = FakeClient({})
    with mk_scheduler(base) as scheduler:
        assert scheduler.fetch_next(client, timeout=0.0) is None
