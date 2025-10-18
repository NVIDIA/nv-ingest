# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
import random
from collections import defaultdict, Counter

import pytest

from nv_ingest_api.util.message_brokers.qos_scheduler import QosScheduler


class MockClient:
    """
    Minimal black-box client for FairScheduler tests.

    - fetch_message(queue_name, timeout): returns a job dict or raises TimeoutError
    - Jobs are stored per-queue in FIFO lists
    - Call order is recorded for assertions where needed
    """

    def __init__(self):
        self._queues = defaultdict(list)
        self.call_log = []

    def add_job(self, queue_name: str, payload=None):
        job = {"queue": queue_name, "payload": payload if payload is not None else f"job@{queue_name}"}
        self._queues[queue_name].append(job)

    def add_job_after(self, queue_name: str, delay_sec: float, payload=None):
        def _later():
            time.sleep(delay_sec)
            self.add_job(queue_name, payload=payload)

        t = threading.Thread(target=_later, daemon=True)
        t.start()

    def fetch_message(self, queue_name: str, timeout: float):
        # Record call order to validate immediate-first access if needed
        self.call_log.append(queue_name)
        bucket = self._queues.get(queue_name, [])
        if bucket:
            return bucket.pop(0)
        # Non-blocking behavior expected by FairScheduler strategies:
        # raise TimeoutError when no job is available immediately
        raise TimeoutError("No job ready")


def make_scheduler(strategy: str = "lottery") -> QosScheduler:
    # Base queue name for deriving sub-queues
    return QosScheduler(base_queue="base_q", strategy=strategy)


def queues_for(sched: QosScheduler):
    return sched.queues  # {"default": ..., "immediate": ..., "micro": ..., "small": ..., "medium": ..., "large": ...}


# ----------------------- Core behavior tests -----------------------


def test_immediate_preempts_others():
    sched = make_scheduler(strategy="round_robin")
    client = MockClient()
    q = queues_for(sched)

    # Jobs exist in both immediate and others; immediate must win
    client.add_job(q["immediate"], payload="immediate_job")
    client.add_job(q["micro"], payload="micro_job")
    client.add_job(q["default"], payload="default_job")

    job = sched.fetch_next(client, timeout=0)
    assert job is not None
    assert job["queue"] == q["immediate"]


@pytest.mark.parametrize("strategy", ["lottery", "round_robin", "weighted_round_robin"])
def test_immediate_preempts_others_all_strategies(strategy):
    sched = make_scheduler(strategy=strategy)
    client = MockClient()
    q = queues_for(sched)

    # Jobs exist in both immediate and others; immediate must win for every strategy
    client.add_job(q["immediate"], payload="immediate_job")
    client.add_job(q["micro"], payload="micro_job")
    client.add_job(q["default"], payload="default_job")

    job = sched.fetch_next(client, timeout=0)
    assert job is not None
    assert job["queue"] == q["immediate"]


def test_nonblocking_returns_none_when_empty():
    sched = make_scheduler(strategy="lottery")
    client = MockClient()

    start = time.monotonic()
    job = sched.fetch_next(client, timeout=0)
    elapsed = time.monotonic() - start

    assert job is None
    # Should be fast return (no enforced sleep)
    assert elapsed < 0.1


@pytest.mark.parametrize("strategy", ["lottery", "round_robin", "weighted_round_robin"])
def test_immediate_polled_first_even_when_empty_all_strategies(strategy):
    sched = make_scheduler(strategy=strategy)
    client = MockClient()
    q = queues_for(sched)

    # No jobs anywhere; ensure first poll is always the immediate queue
    job = sched.fetch_next(client, timeout=0)
    assert job is None
    assert len(client.call_log) >= 1
    assert client.call_log[0] == q["immediate"]


def test_single_sweep_non_immediate_returns_job():
    # No immediate job, only one non-immediate has a job
    sched = make_scheduler(strategy="round_robin")
    client = MockClient()
    q = queues_for(sched)

    client.add_job(q["default"], payload="d1")

    job = sched.fetch_next(client, timeout=0)
    assert job is not None
    assert job["queue"] == q["default"]


def test_timeout_waits_and_eventual_job_arrives():
    sched = make_scheduler(strategy="lottery")
    client = MockClient()
    q = queues_for(sched)

    # No job initially; one appears on 'default' after ~0.2s
    client.add_job_after(q["default"], delay_sec=0.2, payload="late_job")

    start = time.monotonic()
    job = sched.fetch_next(client, timeout=1.0)  # allow up to 1s
    elapsed = time.monotonic() - start

    assert job is not None
    assert job["queue"] == q["default"]
    # Should not return immediately; should be >= 0.2s but not exceed 1.0s by much
    assert elapsed >= 0.2
    assert elapsed < 1.0


def test_returns_none_when_timeout_expires():
    sched = make_scheduler(strategy="weighted_round_robin")
    client = MockClient()

    start = time.monotonic()
    job = sched.fetch_next(client, timeout=0.6)  # will do at least one 0.5s sleep cycle
    elapsed = time.monotonic() - start

    assert job is None
    # Should be roughly the timeout window (allow small slop)
    assert 0.45 <= elapsed <= 0.8


# ----------------------- Strategy-specific tests -----------------------


def test_round_robin_rotation_persists():
    sched = make_scheduler(strategy="round_robin")
    client = MockClient()
    q = queues_for(sched)

    # Fill all non-immediate queues with abundant jobs
    non_immediate = ["micro", "small", "large", "medium", "default"]
    for name in non_immediate:
        for i in range(3):
            client.add_job(q[name], payload=f"{name}-{i}")

    picks = []
    for _ in range(7):
        job = sched.fetch_next(client, timeout=0)
        assert job is not None
        # Map queue path back to its logical name by reverse lookup
        picked_name = next(k for k, v in q.items() if v == job["queue"])
        picks.append(picked_name)

    # Expected rotation over non-immediate list, starting at "micro"
    # 7 picks: micro, small, large, medium, default, micro, small
    assert picks == ["micro", "small", "large", "medium", "default", "micro", "small"]


def compute_swrr_sequence(order, weights, picks):
    current = {k: 0 for k in order}
    total = sum(weights[k] for k in order)
    seq = []
    for _ in range(picks):
        for k in order:
            current[k] += weights[k]
        chosen = max(order, key=lambda k: current[k])
        current[chosen] -= total
        seq.append(chosen)
    return seq


def test_weighted_round_robin_smooth_sequence_matches_swr():
    sched = make_scheduler(strategy="weighted_round_robin")
    client = MockClient()
    q = queues_for(sched)

    order = ["micro", "small", "large", "medium", "default"]
    weights = {"micro": 4, "small": 2, "large": 1, "medium": 1, "default": 1}

    # Ensure every queue always has a job
    for name in order:
        for i in range(5):
            client.add_job(q[name], payload=f"{name}-{i}")

    expected = compute_swrr_sequence(order, weights, picks=10)
    actual = []
    for _ in range(10):
        job = sched.fetch_next(client, timeout=0)
        assert job is not None
        picked_name = next(k for k, v in q.items() if v == job["queue"])
        actual.append(picked_name)

        # Replenish to keep availability high
        client.add_job(job["queue"], payload="replenish")

    assert actual == expected


def test_lottery_weight_bias_distribution():
    # Make randomness reproducible
    random.seed(12345)

    sched = make_scheduler(strategy="lottery")
    client = MockClient()
    q = queues_for(sched)

    non_immediate = ["micro", "small", "large", "medium", "default"]
    # Always-available jobs in all non-immediate queues
    for name in non_immediate:
        for i in range(3):
            client.add_job(q[name], payload=f"{name}-{i}")

    # Run many picks and replenish to maintain availability
    counts = Counter()
    trials = 900
    for _ in range(trials):
        job = sched.fetch_next(client, timeout=0)
        assert job is not None
        picked_name = next(k for k, v in q.items() if v == job["queue"])
        counts[picked_name] += 1
        client.add_job(job["queue"], payload="replenish")

    # Expected weights: micro=4, small=2, others=1 each (total=9)
    # Check rough proportionality bands with slack
    micro_ratio = counts["micro"] / trials
    small_ratio = counts["small"] / trials
    large_ratio = counts["large"] / trials
    medium_ratio = counts["medium"] / trials
    default_ratio = counts["default"] / trials

    # micro should dominate
    assert 0.38 <= micro_ratio <= 0.50
    # small ~ 0.22
    assert 0.17 <= small_ratio <= 0.28
    # others ~ 0.11 each
    for r in [large_ratio, medium_ratio, default_ratio]:
        assert 0.06 <= r <= 0.18
    # monotonic trend micro > small > the rest (likely)
    assert counts["micro"] > counts["small"]
    assert counts["small"] > counts["large"]
    assert counts["small"] > counts["medium"]
    assert counts["small"] > counts["default"]
