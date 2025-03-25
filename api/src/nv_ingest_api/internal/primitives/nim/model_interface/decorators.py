# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import wraps
from multiprocessing import Lock
from multiprocessing import Manager

logger = logging.getLogger(__name__)

# Create a shared manager and lock for thread-safe access
manager = Manager()
global_cache = manager.dict()
lock = Lock()


def multiprocessing_cache(max_calls):
    """
    A decorator that creates a global cache shared between multiple processes.
    The cache is invalidated after `max_calls` number of accesses.

    Args:
        max_calls (int): The number of calls after which the cache is cleared.

    Returns:
        function: The decorated function with global cache and invalidation logic.
    """

    def decorator(func):
        call_count = manager.Value("i", 0)  # Shared integer for call counting

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, frozenset(kwargs.items()))

            with lock:
                call_count.value += 1

                if call_count.value > max_calls:
                    global_cache.clear()
                    call_count.value = 0

                if key in global_cache:
                    return global_cache[key]

            result = func(*args, **kwargs)

            with lock:
                global_cache[key] = result

            return result

        return wrapper

    return decorator
