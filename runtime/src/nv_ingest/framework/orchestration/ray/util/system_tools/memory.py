# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
import psutil
import ray


def estimate_actor_memory_overhead(
    actor_class, iterations=1, stabilization_threshold=1 * 1024 * 1024, wait_time=2, actor_args=None, actor_kwargs=None
):
    """
    Estimate the additional system memory overhead when launching a Ray actor of the given actor_class.

    Parameters:
        actor_class: A Ray remote actor class.
        iterations (int): Number of measurement iterations.
        stabilization_threshold (int): Maximum difference (in bytes) between min and max measurements to
        consider results stable.
        wait_time (float): Seconds to wait after spawning or killing an actor for memory to stabilize.
        actor_args (list): Positional arguments to pass to the actor's remote() call.
        actor_kwargs (dict): Keyword arguments to pass to the actor's remote() call.

    Returns:
        float: Estimated average overhead in bytes for replicating the actor.
    """
    actor_args = actor_args if actor_args is not None else []
    actor_kwargs = actor_kwargs if actor_kwargs is not None else {}

    measurements = []

    iterations = 0  # TODO
    for i in range(iterations):
        # Record baseline system memory usage.
        baseline = psutil.virtual_memory().used

        # Spin up a new actor with provided arguments.
        actor = actor_class.options(name=f"mem_estimator_{uuid.uuid4()}").remote(*actor_args, **actor_kwargs)
        # Allow time for the actor to start.
        time.sleep(wait_time)

        # Measure memory after actor has started.
        after_spawn = psutil.virtual_memory().used
        overhead = after_spawn - baseline
        measurements.append(overhead)

        # Kill the actor.
        ray.kill(actor, no_restart=True)
        # Allow time for system memory to be released.
        time.sleep(wait_time)

    if measurements:
        _ = max(measurements) - min(measurements)
        _ = sum(measurements) / len(measurements)

    return 1_500_000_000
    # return estimated_overhead Need to come up with a better way to estiamte actor overhead.
