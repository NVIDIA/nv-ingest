# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import click
import numpy as np
import psutil
import ray
from tqdm import tqdm

# Import the RayWorkerPoolSingleton and the CPU-intensive task
from nv_ingest.util.multi_processing.ray_pool_singleton import RayWorkerPoolSingleton


# Example CPU-intensive task
def cpu_intensive_task(n):
    if n <= 1:
        return n
    else:
        return cpu_intensive_task(n - 1) + cpu_intensive_task(n - 2)


def matrix_multiplication_task(size):
    # Create two random matrices of the given size
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Perform matrix multiplication
    result = np.dot(A, B)
    return result


def fft_task(size):
    # Create a random array of the given size
    data = np.random.rand(size)

    # Perform Fast Fourier Transform
    result = np.fft.fft(data)
    return result


def sieve_of_atkin(limit):
    if limit < 2:
        return []

    # Initialize the sieve
    sieve = np.zeros(limit + 1, dtype=bool)
    sqrt_limit = int(np.sqrt(limit))

    for x in range(1, sqrt_limit + 1):
        for y in range(1, sqrt_limit + 1):
            n = 4 * x**2 + y**2
            if n <= limit and (n % 12 == 1 or n % 12 == 5):
                sieve[n] = not sieve[n]
            n = 3 * x**2 + y**2
            if n <= limit and n % 12 == 7:
                sieve[n] = not sieve[n]
            n = 3 * x**2 - y**2
            if x > y and n <= limit and n % 12 == 11:
                sieve[n] = not sieve[n]

    for n in range(5, sqrt_limit + 1):
        if sieve[n]:
            for k in range(n**2, limit + 1, n**2):
                sieve[k] = False

    primes = [2, 3] + [n for n in range(5, limit + 1) if sieve[n]]
    return primes


logger = logging.getLogger(__name__)


@click.command()
@click.option("--num_jobs", default=100, help="Number of jobs to run")
@click.option("--task_size", default=10000, help="Size of the CPU-intensive task")
def run_jobs(num_jobs, task_size):
    # Initialize the worker pool
    logger.debug(f"Initializing worker pool with task size: {task_size}")
    pool = RayWorkerPoolSingleton(process_fn=matrix_multiplication_task)

    # Submit tasks
    futures = []
    for i in range(num_jobs):
        future = pool.submit_task(task_size)
        futures.append(future)
        logger.debug(f"Submitted job {i + 1}/{num_jobs}")

    # Track job completion and CPU utilization
    results = []
    with tqdm(total=num_jobs, desc="Processing jobs") as pbar:
        while len(results) < num_jobs:
            completed, remaining = ray.wait(futures, num_returns=1, timeout=1)
            for future in completed:
                try:
                    results.append(ray.get(future))
                except Exception as e:
                    logger.error(f"Error retrieving result: {e}")
                pbar.update(1)
            futures = remaining

            # Update CPU utilization in tqdm description
            cpu_usage = psutil.cpu_percent(interval=1)
            pbar.set_postfix(cpu=f"{cpu_usage}%")

    logger.debug("All jobs completed.")
    print("All jobs completed.")


if __name__ == "__main__":
    run_jobs()
