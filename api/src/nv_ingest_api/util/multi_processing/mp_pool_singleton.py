# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import math
import multiprocessing as mp
import os
from multiprocessing import Manager
from threading import Lock
from typing import Any
from typing import Callable
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleFuture:
    """
    A simplified future object for handling asynchronous task results.

    This class allows the storage and retrieval of the result or exception from an asynchronous task,
    using multiprocessing primitives for inter-process communication.

    Parameters
    ----------
    manager : multiprocessing.Manager
        A multiprocessing manager that provides shared memory for the result and exception.

    Attributes
    ----------
    _result : multiprocessing.Value
        A shared memory object to store the result of the asynchronous task.
    _exception : multiprocessing.Value
        A shared memory object to store any exception raised during task execution.
    _done : multiprocessing.Event
        An event that signals the completion of the task.

    Methods
    -------
    set_result(result)
        Sets the result of the task and marks the task as done.
    set_exception(exception)
        Sets the exception of the task and marks the task as done.
    result()
        Waits for the task to complete and returns the result, or raises the exception if one occurred.
    """

    def __init__(self, manager: Manager):
        self._result = manager.Value("i", None)
        self._exception = manager.Value("i", None)
        self._done = manager.Event()

    def set_result(self, result: Any) -> None:
        """
        Sets the result of the asynchronous task and signals task completion.

        Parameters
        ----------
        result : Any
            The result of the asynchronous task.

        Returns
        -------
        None
        """
        self._result.value = result
        self._done.set()

    def set_exception(self, exception: Exception) -> None:
        """
        Sets the exception raised by the asynchronous task and signals task completion.

        Parameters
        ----------
        exception : Exception
            The exception raised during task execution.

        Returns
        -------
        None
        """
        self._exception.value = exception
        self._done.set()

    def result(self) -> Any:
        """
        Retrieves the result of the asynchronous task or raises the exception if one occurred.

        This method blocks until the task is complete.

        Returns
        -------
        Any
            The result of the asynchronous task.

        Raises
        ------
        Exception
            The exception raised during task execution, if any.
        """
        self._done.wait()
        if self._exception.value is not None:
            raise self._exception.value
        return self._result.value


class ProcessWorkerPoolSingleton:
    """
    A singleton process worker pool for managing a fixed number of worker processes.

    This class implements a process pool using the singleton pattern, ensuring that only one instance
    of the pool exists. It manages worker processes that can execute tasks asynchronously.

    Attributes
    ----------
    _instance : ProcessWorkerPoolSingleton or None
        The singleton instance of the class.
    _lock : threading.Lock
        A lock to ensure thread-safe initialization of the singleton instance.
    _total_workers : int
        The total number of worker processes.

    Methods
    -------
    __new__(cls)
        Ensures only one instance of the class is created.
    _initialize(total_max_workers)
        Initializes the worker pool with the specified number of workers.
    submit_task(process_fn, *args)
        Submits a task to the worker pool for asynchronous execution.
    close()
        Closes the worker pool and terminates all worker processes.
    """

    _instance: Optional["ProcessWorkerPoolSingleton"] = None
    _lock: Lock = Lock()
    _total_workers: int = 0

    def __new__(cls):
        """
        Ensures that only one instance of the ProcessWorkerPoolSingleton is created.

        Returns
        -------
        ProcessWorkerPoolSingleton
            The singleton instance of the class.
        """
        logger.debug("Creating ProcessWorkerPoolSingleton instance...")
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProcessWorkerPoolSingleton, cls).__new__(cls)
                max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * 0.4))
                cls._instance._initialize(max_workers)
                logger.debug(f"ProcessWorkerPoolSingleton instance created: {cls._instance}")
            else:
                logger.debug(f"ProcessWorkerPoolSingleton instance already exists: {cls._instance}")
        return cls._instance

    def _initialize(self, total_max_workers: int) -> None:
        """
        Initializes the worker pool with the specified number of worker processes.

        Parameters
        ----------
        total_max_workers : int
            The maximum number of worker processes to create.

        Returns
        -------
        None
        """
        self._total_max_workers = total_max_workers
        self._context = mp.get_context("fork")
        self._task_queue = self._context.Queue()
        self._manager = mp.Manager()
        self._processes = []
        logger.debug(f"Initializing ProcessWorkerPoolSingleton with {total_max_workers} workers.")
        for i in range(total_max_workers):
            p = self._context.Process(target=self._worker, args=(self._task_queue, self._manager))
            p.start()
            self._processes.append(p)
            logger.debug(f"Started worker process {i + 1}/{total_max_workers}: PID {p.pid}")
        logger.debug(f"Initialized with max workers: {total_max_workers}")

    @staticmethod
    def _worker(task_queue: mp.Queue, manager: mp.Manager) -> None:
        """
        The worker process function that executes tasks from the queue.

        Parameters
        ----------
        task_queue : multiprocessing.Queue
            The queue from which tasks are retrieved.
        manager : multiprocessing.Manager
            The manager providing shared memory for inter-process communication.

        Returns
        -------
        None
        """
        logger.debug(f"Worker process started: PID {os.getpid()}")
        while True:
            task = task_queue.get()
            if task is None:  # Stop signal
                logger.debug(f"Worker process {os.getpid()} received stop signal.")
                break

            future, process_fn, args = task
            args, *kwargs = args
            try:
                result = process_fn(*args, **{k: v for kwarg in kwargs for k, v in kwarg.items()})
                future.set_result(result)
            except Exception as e:
                logger.error(f"Future result failure - {e}\n")
                future.set_exception(e)

    def submit_task(self, process_fn: Callable, *args: Any) -> SimpleFuture:
        """
        Submits a task to the worker pool for asynchronous execution.

        Parameters
        ----------
        process_fn : callable
            The function to be executed by the worker process.
        args : tuple
            The arguments to pass to the function.

        Returns
        -------
        SimpleFuture
            A future object representing the result of the task.
        """
        future = SimpleFuture(self._manager)
        self._task_queue.put((future, process_fn, args))
        return future

    def close(self) -> None:
        """
        Closes the worker pool and terminates all worker processes.

        This method sends a stop signal to each worker and waits for them to terminate.

        Returns
        -------
        None
        """
        logger.debug("Closing ProcessWorkerPoolSingleton...")
        for _ in range(self._total_max_workers):
            self._task_queue.put(None)  # Send stop signal to all workers
            logger.debug("Sent stop signal to worker.")
        for i, p in enumerate(self._processes):
            p.join()
            logger.debug(f"Worker process {i + 1}/{self._total_max_workers} joined: PID {p.pid}")
        logger.debug("ProcessWorkerPoolSingleton closed.")
