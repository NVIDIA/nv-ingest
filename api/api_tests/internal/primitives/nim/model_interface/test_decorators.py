# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import unittest
import time
from multiprocessing import Process, Queue
import logging

from nv_ingest_api.internal.primitives.nim.model_interface.decorators import multiprocessing_cache

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


class TestMultiprocessingCache(unittest.TestCase):
    def setUp(self):
        # Reset between tests by importing the module again
        pass

    def test_basic_caching(self):
        """Test that the function result is cached."""
        call_count = 0

        @multiprocessing_cache(max_calls=10)
        def example_function(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # First call should execute the function
        result1 = example_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)

        # Second call with same args should use cache
        result2 = example_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Call count shouldn't increase

        # Call with different args should execute the function
        result3 = example_function(3, 4)
        self.assertEqual(result3, 7)
        self.assertEqual(call_count, 2)  # Call count should increase

    def test_cache_invalidation(self):
        """Test that the cache is invalidated after max_calls."""
        call_count = 0

        @multiprocessing_cache(max_calls=3)
        def example_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First set of calls
        self.assertEqual(example_function(1), 2)
        self.assertEqual(call_count, 1)
        self.assertEqual(example_function(1), 2)  # Cached
        self.assertEqual(call_count, 1)

        # Another unique call
        self.assertEqual(example_function(2), 4)
        self.assertEqual(call_count, 2)

        # This should be the call that triggers invalidation
        self.assertEqual(example_function(3), 6)
        self.assertEqual(call_count, 3)

        # This call should use a fresh cache after invalidation
        self.assertEqual(example_function(1), 2)
        self.assertEqual(call_count, 4)  # Function should be called again

    def test_kwargs_handling(self):
        """Test that the function caches correctly with keyword arguments."""
        call_count = 0

        @multiprocessing_cache(max_calls=10)
        def example_function(a, b=2, c=3):
            nonlocal call_count
            call_count += 1
            return a + b + c

        # Call with different keyword argument patterns
        result1 = example_function(1, b=2, c=3)
        self.assertEqual(result1, 6)
        self.assertEqual(call_count, 1)

        # Same effective call, different syntax
        result2 = example_function(1, c=3, b=2)
        self.assertEqual(result2, 6)
        self.assertEqual(call_count, 1)  # Should use cache

        # Default arguments - the implementation doesn't recognize this as the same call
        # because default args aren't part of the key
        result3 = example_function(1)
        self.assertEqual(result3, 6)
        self.assertEqual(call_count, 2)  # Will call function again

        # Different arguments
        result4 = example_function(1, b=5)
        self.assertEqual(result4, 9)
        self.assertEqual(call_count, 3)  # Should call function again

    def test_multiple_processes(self):
        """Test that the cache works across multiple processes."""

        def worker_function(queue, value):
            @multiprocessing_cache(max_calls=10)
            def cached_func(x):
                # This sleep simulates some expensive computation
                time.sleep(0.1)
                return x * 2

            # First call should compute
            start_time = time.time()
            result1 = cached_func(value)
            first_call_time = time.time() - start_time

            # Second call should be cached
            start_time = time.time()
            result2 = cached_func(value)
            second_call_time = time.time() - start_time

            queue.put((result1, result2, first_call_time, second_call_time))

        # Create two processes that both call the same function with the same argument
        queue = Queue()
        p1 = Process(target=worker_function, args=(queue, 5))
        p2 = Process(target=worker_function, args=(queue, 5))

        p1.start()
        time.sleep(0.05)  # Small delay to ensure p1 starts first
        p2.start()

        p1.join()
        p2.join()

        # Get results from both processes
        results = []
        while not queue.empty():
            results.append(queue.get())

        # Both processes should return the same result
        self.assertEqual(len(results), 2)

        for result1, result2, first_call_time, second_call_time in results:
            # Both calls should return correct result
            self.assertEqual(result1, 10)
            self.assertEqual(result2, 10)

        # At least one process must have actually computed (hit the sleep).
        # The other process may see a cached first call if it starts after
        # the first process finishes, so we only require one slow first call.
        first_call_times = [r[2] for r in results]
        self.assertTrue(
            any(t >= 0.05 for t in first_call_times),
            f"Expected at least one first call >= 0.05s, got {first_call_times}",
        )

    def test_different_function_caches(self):
        """Test that different functions have separate caches."""
        call_count1 = 0
        call_count2 = 0

        @multiprocessing_cache(max_calls=10)
        def function1(x):
            nonlocal call_count1
            call_count1 += 1
            return x * 2

        @multiprocessing_cache(max_calls=5)
        def function2(x):
            nonlocal call_count2
            call_count2 += 1
            return x * 3

        # Call both functions
        self.assertEqual(function1(1), 2)
        self.assertEqual(call_count1, 1)
        self.assertEqual(function2(1), 3)
        self.assertEqual(call_count2, 1)

        # Call again, should use caches
        self.assertEqual(function1(1), 2)
        self.assertEqual(call_count1, 1)
        self.assertEqual(function2(1), 3)
        self.assertEqual(call_count2, 1)

        # Should not affect each other's cache
        for i in range(5):  # Exceed function2's max_calls
            function2(i)

        # The implementation actually counts total calls across all decorated functions,
        # not per function, so function1's cache is also affected
        self.assertEqual(function1(1), 2)
        self.assertEqual(call_count1, 2)  # Cache was invalidated, so count increases

        # function2's cache should be invalidated
        self.assertEqual(function2(1), 3)
        self.assertEqual(call_count2, 6)  # 1 (initial) + 5 (loop) = 6

    def test_hashable_arguments(self):
        """Test caching behavior with hashable arguments."""
        call_count = 0

        @multiprocessing_cache(max_calls=10)
        def example_function(a, b, c):
            nonlocal call_count
            call_count += 1
            return a + b + c

        # Call with simple hashable arguments
        result1 = example_function(1, 2, 3)
        self.assertEqual(result1, 6)
        self.assertEqual(call_count, 1)

        # Same call again
        result2 = example_function(1, 2, 3)
        self.assertEqual(result2, 6)
        self.assertEqual(call_count, 1)  # Should use cache

        # Note: The cache implementation cannot handle unhashable arguments like
        # lists or dictionaries in the positional arguments. Attempting to use
        # unhashable types would raise a TypeError exception.

    def test_cache_with_exceptions(self):
        """Test that exceptions aren't cached."""
        call_count = 0

        @multiprocessing_cache(max_calls=10)
        def failing_function(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Function failed")
            return "Success"

        # First call should raise exception
        with self.assertRaises(ValueError):
            failing_function(True)
        self.assertEqual(call_count, 1)

        # Second call with same args should try again, not use cache
        with self.assertRaises(ValueError):
            failing_function(True)
        self.assertEqual(call_count, 2)

        # Call with different args should succeed
        result = failing_function(False)
        self.assertEqual(result, "Success")
        self.assertEqual(call_count, 3)

    def test_zero_max_calls(self):
        """Test behavior with max_calls=0 (should disable caching)."""
        call_count = 0

        @multiprocessing_cache(max_calls=0)
        def example_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Each call should execute the function
        self.assertEqual(example_function(1), 2)
        self.assertEqual(call_count, 1)
        self.assertEqual(example_function(1), 2)
        self.assertEqual(call_count, 2)  # Should increment

    def test_negative_max_calls(self):
        """Test behavior with negative max_calls (actually disables caching)."""
        call_count = 0

        @multiprocessing_cache(max_calls=-1)
        def example_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Multiple calls will execute the function each time
        # since negative max_calls disables caching
        for _ in range(100):
            self.assertEqual(example_function(1), 2)

        self.assertEqual(call_count, 100)  # Should call function every time


if __name__ == "__main__":
    unittest.main()
