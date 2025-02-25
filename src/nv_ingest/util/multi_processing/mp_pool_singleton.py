# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent
import logging
import math
import multiprocessing as mp
import os
import pickle
from threading import Lock
from typing import Any
from typing import Callable
from typing import Optional

logger = logging.getLogger(__name__)


class ProcessWorkerPoolSingleton:
    """
    A singleton process worker pool using concurrent.futures.ProcessPoolExecutor with
    an explicit 'fork' context. This version attempts to pre-check if the submitted task
    function is pickleable and raises a clear exception if not. It also wraps task submission
    so that submission failures are robustly reported.

    Usage Example:
        pool = ProcessWorkerPoolSingleton()
        # The process function must be defined at the module level for picklability.
        future = pool.submit_task(process_fn, (df, task_props))
        result = future.result()
    """

    _instance: Optional["ProcessWorkerPoolSingleton"] = None
    _lock: Lock = Lock()

    def __new__(cls):
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
        Initializes the process pool with the specified number of worker processes,
        using the 'fork' context to match the original design.
        """
        self._total_max_workers = total_max_workers
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=total_max_workers, mp_context=mp.get_context("fork")
        )
        logger.debug(f"Initialized ProcessWorkerPoolSingleton with {total_max_workers} workers.")

    def submit_task(self, process_fn: Callable, *args: Any, **kwargs: Any) -> concurrent.futures.Future:
        """
        Submits a task to the worker pool for asynchronous execution.

        Before submission, this method attempts to pickle the provided function to verify that it is
        pickleable. This is required for ProcessPoolExecutor to work properly.

        Parameters
        ----------
        process_fn : Callable
            The function to execute. It must be defined at the module level.
        *args
            Positional arguments for the process function. If a single tuple is passed as the only argument,
            it will be unpacked.
        **kwargs
            Keyword arguments for the process function.

        Returns
        -------
        concurrent.futures.Future
            A Future representing the asynchronous execution of the task.

        Raises
        ------
        ValueError
            If process_fn cannot be pickled.
        RuntimeError
            If the task submission fails.
        """
        # If a single tuple is passed as the only positional argument, unpack it.
        if len(args) == 1 and isinstance(args[0], tuple) and not kwargs:
            args = args[0]

        # Verify picklability of the function early.
        try:
            pickle.dumps(process_fn)
        except Exception as e:
            message = f"process_fn is not pickleable: {e}"
            logger.exception(message)
            raise ValueError(message) from e

        logger.debug("Submitting task to ProcessWorkerPoolSingleton.")
        try:
            future = self._executor.submit(process_fn, *args, **kwargs)
        except Exception as e:
            message = f"Task submission failed: {e}"
            logger.exception(message)
            raise RuntimeError(message) from e

        return future

    def close(self) -> None:
        """
        Shuts down the worker pool and waits for all worker processes to terminate.
        """
        logger.debug("Shutting down ProcessWorkerPoolSingleton.")
        self._executor.shutdown(wait=True)
        logger.debug("ProcessWorkerPoolSingleton shut down.")
