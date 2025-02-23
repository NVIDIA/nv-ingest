from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue

import pytest

from nv_ingest_api.primitives.nim.model_interface.decorators import multiprocessing_cache


@pytest.fixture
def shared_manager():
    """Fixture to create a shared multiprocessing manager."""
    return Manager()


def test_global_cache_with_same_arguments(shared_manager):
    queue = Queue()

    @multiprocessing_cache(3)
    def add(x, y):
        queue.put(1)  # Track each function call
        return x + y

    def worker(val1, val2):
        add(val1, val2)

    processes = [
        Process(target=worker, args=(1, 2)),  # called 1st time
        Process(target=worker, args=(1, 2)),
        Process(target=worker, args=(1, 2)),
        Process(target=worker, args=(1, 2)),  # called 2nd time
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    total_calls = 0
    while not queue.empty():
        total_calls += queue.get()

    assert total_calls == 2


def test_global_cache_with_different_arguments(shared_manager):
    queue = Queue()

    @multiprocessing_cache(3)
    def add(x, y):
        queue.put(1)  # Track each function call
        return x + y

    def worker(val1, val2):
        add(val1, val2)

    processes = [
        Process(target=worker, args=(1, 2)),  # called 1st time
        Process(target=worker, args=(3, 4)),  # called 2nd time
        Process(target=worker, args=(1, 2)),
        Process(target=worker, args=(3, 4)),
        Process(target=worker, args=(1, 2)),
        Process(target=worker, args=(3, 4)),
        Process(target=worker, args=(3, 4)),  # called 3rd time
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    total_calls = 0
    while not queue.empty():
        total_calls += queue.get()

    assert total_calls == 3
