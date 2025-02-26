# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import multiprocessing as mp
import os
import threading
from ctypes import py_object
from threading import RLock
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class SimpleFuture:
    """
    A simplified future object for handling asynchronous task results.

    This class uses multiprocessing primitives to store and retrieve the result or exception
    from an asynchronous task, and it pins the manager used to create the shared proxies so that
    they remain valid until the future is resolved.

    Attributes
    ----------
    _manager : mp.Manager
        The Manager instance used to create shared objects. It is "pinned" (kept alive) until
        the future is resolved.
    _result : mp.Value
        A proxy holding the result of the asynchronous task.
    _exception : mp.Value
        A proxy holding any exception raised during task execution.
    _done : mp.Event
        A synchronization event that signals task completion.
    """

    def __init__(self, manager: mp.Manager) -> None:
        """
        Initialize a SimpleFuture.

        Parameters
        ----------
        manager : mp.Manager
            The Manager instance used to create shared objects.
        """
        self._manager = manager  # Pin the manager until this future is resolved.
        self._result = manager.Value(py_object, None)
        self._exception = manager.Value(py_object, None)
        self._done = manager.Event()

    def set_result(self, result: Any) -> None:
        """
        Set the result of the asynchronous task.

        Parameters
        ----------
        result : Any
            The result produced by the task.
        """
        self._result.value = result
        self._done.set()

    def set_exception(self, exception: Exception) -> None:
        """
        Set an exception raised during the execution of the asynchronous task.

        Parameters
        ----------
        exception : Exception
            The exception encountered during task execution.
        """
        self._exception.value = exception
        self._done.set()

    def result(self) -> Any:
        """
        Block until the task completes and return the result.

        Returns
        -------
        Any
            The result of the asynchronous task.

        Raises
        ------
        Exception
            Re-raises any exception encountered during task execution.
        """
        self._done.wait()
        if self._exception.value is not None:
            raise self._exception.value
        return self._result.value

    def __getstate__(self) -> dict:
        """
        Return the state for pickling, excluding the _manager to avoid pickling errors.

        Returns
        -------
        dict
            The object's state without the _manager attribute.
        """
        state = self.__dict__.copy()
        state.pop("_manager", None)
        return state


class ProcessWorkerPoolSingleton:
    """
    A singleton process worker pool for managing a fixed number of worker processes.

    This class implements a process pool using the singleton pattern, ensuring that only one
    instance exists. It manages worker processes that execute tasks asynchronously. A background
    thread periodically checks if the task queue is empty; if so, it refreshes the entire pool:
      - Closes (and optionally joins) all current worker processes (without shutting down the active Manager).
      - Creates a new Manager.
      - Re-creates all worker processes using the new Manager.
      - Swaps in the new Manager as the active manager, allowing the old Manager to eventually be garbage collected.

    The public task submission interface (submit_task) remains unchanged.

    Attributes
    ----------
    _instance : Optional[ProcessWorkerPoolSingleton]
        The singleton instance.
    _lock : RLock
        A reentrant lock to ensure thread-safe access.
    _total_workers : int
        The total number of worker processes.
    """

    _instance: Optional["ProcessWorkerPoolSingleton"] = None
    _lock: RLock = RLock()  # Use reentrant lock to avoid deadlocks in nested acquisitions.
    _total_workers: int = 0

    def __new__(cls) -> "ProcessWorkerPoolSingleton":
        """
        Create or return the singleton instance.

        Returns
        -------
        ProcessWorkerPoolSingleton
            The singleton instance.
        """
        logger.debug("Creating ProcessWorkerPoolSingleton instance...")
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProcessWorkerPoolSingleton, cls).__new__(cls)
                max_workers = math.floor(max(1, len(os.sched_getaffinity(0)) * 0.4))
                cls._instance._initialize(max_workers)
                cls._instance._start_manager_monitor()
                logger.info(f"ProcessWorkerPoolSingleton instance created: {cls._instance}")
            else:
                logger.info(f"ProcessWorkerPoolSingleton instance already exists: {cls._instance}")
        return cls._instance

    def _initialize(self, total_max_workers: int, new_manager: Optional[mp.Manager] = None) -> None:
        """
        Initialize the worker pool with a specified number of worker processes.

        Parameters
        ----------
        total_max_workers : int
            The number of worker processes to create.
        new_manager : Optional[mp.Manager], optional
            A new Manager to use for shared objects. If None, a new Manager is created.
        """
        self._total_workers = total_max_workers
        self._context = mp.get_context("fork")
        self._task_queue = self._context.Queue()
        self._manager = new_manager if new_manager is not None else mp.Manager()
        self._active_manager = self._manager
        self._processes = []

        logger.debug(f"Initializing ProcessWorkerPoolSingleton with {total_max_workers} workers.")
        for i in range(total_max_workers):
            process = self._context.Process(target=self._worker, args=(self._task_queue, self._manager))
            process.start()
            self._processes.append(process)
            logger.debug(f"Started worker process {i + 1}/{total_max_workers}: PID {process.pid}")
        logger.debug(f"Initialized with max workers: {total_max_workers}")

    def _start_manager_monitor(self) -> None:
        """
        Start a background thread that periodically checks if the task queue is empty.
        """
        self._stop_manager_monitor = False
        self._monitor_thread = threading.Thread(target=self._monitor_manager, daemon=True)
        self._monitor_thread.start()
        logger.debug("Started Manager monitoring thread.")

    def _monitor_manager(self) -> None:
        """
        Periodically check whether the task queue is empty. If so, refresh the pool.

        Notes
        -----
        Consider adding exception handling in this loop to prevent unexpected thread termination.
        """
        import time

        check_interval = 2 * 60  # 5 minute Manager cache rotation interval
        while not self._stop_manager_monitor:
            time.sleep(check_interval)
            with self._lock:
                self._refresh_manager()

    def _refresh_manager(self) -> None:
        """
        Refresh the Manager and re-create all worker processes.

        This method performs the following steps:
          1. Closes current worker processes without shutting down the active Manager.
          3. reinitializes the worker pool using the new manager.
          4. swaps in the new manager as the active manager.
          2. Creates a new Manager.
        """
        logger.warning("Cycling ProcessWorkerPoolSingleton workers...")

        # Close current workers without waiting (join=False).
        self.close(join=False)

        # Create a new Manager and reinitialize the worker pool.
        new_manager = mp.Manager()
        self._initialize(self._total_workers, new_manager=new_manager)

        # Swap in the new Manager.
        self._active_manager = new_manager
        logger.warning("ProcessWorkerPoolSingleton workers cycled.")

    @staticmethod
    def _worker(task_queue: mp.Queue, manager: mp.Manager) -> None:
        """
        Worker process function that executes tasks from the queue.

        Parameters
        ----------
        task_queue : mp.Queue
            The queue from which tasks are retrieved.
        manager : mp.Manager
            The Manager instance used to create shared objects.
        """
        logger.debug(f"Worker process started: PID {os.getpid()}")
        while True:
            task = task_queue.get()
            if task is None:
                logger.debug(f"Worker process {os.getpid()} received stop signal.")
                break

            future, process_fn, args = task
            args, *kwargs = args
            try:
                # Flatten kwargs from list of dictionaries.
                kwargs_dict = {k: v for kwarg in kwargs for k, v in kwarg.items()}
                result = process_fn(*args, **kwargs_dict)
                future.set_result(result)
            except Exception as e:
                logger.error(f"Future result failure - {e}")
                future.set_exception(e)

    def submit_task(self, process_fn: Callable, *args: Any) -> SimpleFuture:
        """
        Submit a task to the worker pool for asynchronous execution.

        Parameters
        ----------
        process_fn : Callable
            The function to be executed by a worker.
        *args : Any
            Positional arguments for the function.

        Returns
        -------
        SimpleFuture
            A future representing the asynchronous execution of the task.
        """
        with self._lock:
            future = SimpleFuture(self._active_manager)
            self._task_queue.put((future, process_fn, args))
            return future

    def close(self, join: bool = True) -> None:
        """
        Close the worker pool by sending stop signals to all workers.
        Optionally waits for them to terminate (join).

        The active Manager is not shut down so that outstanding references remain valid.

        Parameters
        ----------
        join : bool, optional
            If True (default), waits for the worker processes to terminate.
            If False, sends stop signals and returns immediately.
        """
        logger.debug("Closing ProcessWorkerPoolSingleton workers...")
        for _ in range(self._total_workers):
            self._task_queue.put(None)
            logger.debug("Sent stop signal to worker.")
        if join:
            for i, process in enumerate(self._processes):
                process.join()
                logger.debug(f"Worker process {i + 1}/{self._total_workers} joined: PID {process.pid}")
        logger.debug("Worker pool closed.")

    def shutdown_manager_monitor(self) -> None:
        """
        Stop the background Manager monitoring thread.
        """
        self._stop_manager_monitor = True
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=5)
            logger.debug("Manager monitoring thread stopped.")
