# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import math
import os
import sys
import multiprocessing as mp
from threading import Lock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SimpleFuture:
    """
    A simplified future object that uses a multiprocessing Pipe to receive its result.

    When the result() method is called, it blocks until the worker sends a tuple
    (result, error) over the pipe.
    """

    def __init__(self, parent_conn: mp.connection.Connection) -> None:
        """
        Parameters
        ----------
        parent_conn : mp.connection.Connection
            The parent end of the multiprocessing Pipe used to receive the result.
        """
        self._parent_conn: mp.connection.Connection = parent_conn

    def result(self) -> Any:
        """
        Retrieve the result from the future, blocking until it is available.

        Returns
        -------
        Any
            The result returned by the worker function.

        Raises
        ------
        Exception
            If the worker function raised an exception, it is re-raised here.
        """
        result, error = self._parent_conn.recv()
        if error is not None:
            raise error
        return result


class ProcessWorkerPoolSingleton:
    """
    A singleton process worker pool using a dual-queue implementation.

    Instead of a global result queue, each submitted task gets its own Pipe.
    The submit_task() method returns a SimpleFuture, whose result() call blocks
    until the task completes.
    """

    _instance: Optional["ProcessWorkerPoolSingleton"] = None
    _lock: Lock = Lock()
    _total_workers: int = 0

    def __new__(cls) -> "ProcessWorkerPoolSingleton":
        """
        Create or return the singleton instance of ProcessWorkerPoolSingleton.

        Returns
        -------
        ProcessWorkerPoolSingleton
            The singleton instance.
        """
        logger.debug("Creating ProcessWorkerPoolSingleton instance...")
        with cls._lock:
            if cls._instance is None:
                max_worker_limit: int = int(os.environ.get("MAX_INGEST_PROCESS_WORKERS", -1))
                instance = super().__new__(cls)
                # Determine available CPU count using affinity if possible
                available: Optional[int] = (
                    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
                )
                # Use 40% of available CPUs, ensuring at least one worker
                max_workers: int = math.floor(max(1, available * 0.4))
                if (max_worker_limit > 0) and (max_workers > max_worker_limit):
                    max_workers = max_worker_limit
                logger.debug("Creating ProcessWorkerPoolSingleton instance with max workers: %d", max_workers)
                instance._initialize(max_workers)
                logger.debug("ProcessWorkerPoolSingleton instance created: %s", instance)
                cls._instance = instance
            else:
                logger.debug("ProcessWorkerPoolSingleton instance already exists: %s", cls._instance)
        return cls._instance

    def _initialize(self, total_max_workers: int) -> None:
        """
        Initialize the worker pool with the specified number of worker processes.

        Parameters
        ----------
        total_max_workers : int
            The total number of worker processes to start.
        """
        self._total_workers = total_max_workers

        start_method = "fork"
        if sys.platform.lower() == "darwin":
            start_method = "spawn"
        self._context: mp.context.ForkContext = mp.get_context(start_method)

        # Bounded task queue: maximum tasks queued = 2 * total_max_workers.
        self._task_queue: mp.Queue = self._context.Queue(maxsize=2 * total_max_workers)
        self._next_task_id: int = 0
        self._processes: list[mp.Process] = []
        logger.debug(
            "Initializing ProcessWorkerPoolSingleton with %d workers and queue size %d.",
            total_max_workers,
            2 * total_max_workers,
        )
        for i in range(total_max_workers):
            p: mp.Process = self._context.Process(target=self._worker, args=(self._task_queue,))
            p.start()
            self._processes.append(p)
            logger.debug("Started worker process %d/%d: PID %d", i + 1, total_max_workers, p.pid)
        logger.debug("Initialized with max workers: %d", total_max_workers)

    @staticmethod
    def _worker(task_queue: mp.Queue) -> None:
        """
        Worker process that continuously processes tasks from the task queue.

        Parameters
        ----------
        task_queue : mp.Queue
            The queue from which tasks are retrieved.
        """
        logger.debug("Worker process started: PID %d", os.getpid())
        while True:
            task = task_queue.get()
            if task is None:
                # Stop signal received; exit the loop.
                logger.debug("Worker process %d received stop signal.", os.getpid())
                break
            # Unpack task: (task_id, process_fn, args, child_conn)
            task_id, process_fn, args, child_conn = task
            try:
                result = process_fn(*args)
                child_conn.send((result, None))
            except Exception as e:
                logger.error("Task %d error in worker %d: %s", task_id, os.getpid(), e)
                child_conn.send((None, e))
            finally:
                child_conn.close()

    def submit_task(self, process_fn: Callable, *args: Any) -> SimpleFuture:
        """
        Submits a task to the worker pool for asynchronous execution.

        If a single tuple is passed as the only argument, it is unpacked.

        Parameters
        ----------
        process_fn : Callable
            The function to be executed asynchronously.
        *args : Any
            The arguments to pass to the process function. If a single argument is a tuple,
            it will be unpacked as the function arguments.

        Returns
        -------
        SimpleFuture
            A future object that can be used to retrieve the result of the task.
        """
        # Unpack tuple if a single tuple argument is provided.
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        parent_conn, child_conn = mp.Pipe(duplex=False)
        task_id: int = self._next_task_id
        self._next_task_id += 1
        self._task_queue.put((task_id, process_fn, args, child_conn))
        return SimpleFuture(parent_conn)

    def close(self) -> None:
        """
        Closes the worker pool and terminates all worker processes.

        Sends a stop signal to each worker and waits for them to terminate.
        """
        logger.debug("Closing ProcessWorkerPoolSingleton...")
        # Send a stop signal (None) for each worker.
        for _ in range(self._total_workers):
            self._task_queue.put(None)
            logger.debug("Sent stop signal to worker.")
        # Wait for all processes to finish.
        for i, p in enumerate(self._processes):
            p.join()
            logger.debug("Worker process %d/%d joined: PID %d", i + 1, self._total_workers, p.pid)
        logger.debug("ProcessWorkerPoolSingleton closed.")
