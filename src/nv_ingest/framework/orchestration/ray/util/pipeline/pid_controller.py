# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Deque

logger = logging.getLogger(__name__)


class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_replicas: int,
        memory_threshold: int,
        stage_cost_estimates: Dict[str, int],
        window_size: int = 10,
        memory_rate_threshold: float = 0.05,
        penalty_factor: float = 1.0,
    ):
        """
        Initialize the PID controller for multi-stage replica management.

        Parameters
        ----------
        kp : float
            Proportional gain.
        ki : float
            Integral gain.
        kd : float
            Derivative gain (not used for memory, reserved for future use).
        max_replicas : int
            Maximum allowed replicas across all stages.
        memory_threshold : int
            Maximum allowed memory usage in MB.
        stage_cost_estimates : dict of str to int
            Estimated memory cost (in MB) per replica for each stage.
        window_size : int
            Number of recent samples to use for dynamic memory estimation.
        memory_rate_threshold : float
            Maximum acceptable rate of memory growth per replica (MB/s or MB/interval).
        penalty_factor : float
            Factor to multiply with the idle cycle count to compute a downscale penalty.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_error: Dict[str, float] = {}
        self.prev_error: Dict[str, float] = {}

        self.max_replicas = max_replicas
        self.memory_threshold = memory_threshold
        self.stage_cost_estimates = stage_cost_estimates

        self.memory_history: Dict[str, Deque[int]] = {}
        self.memory_rate_history: Dict[str, float] = {}
        self.window_size = window_size
        self.memory_rate_threshold = memory_rate_threshold

        # New: track idle cycles per stage and penalty factor.
        self.idle_cycles: Dict[str, int] = {}
        self.penalty_factor = penalty_factor

    def initialize_stage(self, stage: str) -> None:
        self.integral_error[stage] = 0.0
        self.prev_error[stage] = 0.0
        self.memory_history[stage] = deque(maxlen=self.window_size)
        self.memory_rate_history[stage] = 0.0
        # Initialize idle cycle counter.
        self.idle_cycles[stage] = 0

    def estimate_dynamic_memory(self, stage: str, memory_samples: Deque[int]) -> float:
        if memory_samples:
            return np.mean(memory_samples)
        return self.stage_cost_estimates[stage]

    def estimate_memory_rate(self, stage: str, new_value: float) -> float:
        old_value = self.memory_rate_history.get(stage, new_value)
        rate = new_value - old_value
        self.memory_rate_history[stage] = new_value
        return rate

    def update(self, stage_metrics: Dict[str, Dict[str, Any]], current_memory_usage: int) -> Dict[str, int]:
        total_replicas = sum(m["replicas"] for m in stage_metrics.values())
        memory_budget_remaining = self.memory_threshold - current_memory_usage

        logger.debug(
            f"[PIDController] Total replicas: {total_replicas}, "
            f"Current memory usage: {current_memory_usage} MB, "
            f"Memory budget remaining: {memory_budget_remaining} MB"
        )

        stage_adjustments: Dict[str, int] = {}

        for stage, metrics in stage_metrics.items():
            queue_depth = metrics["queue_depth"]
            throughput = metrics["throughput"]
            replicas = metrics["replicas"]
            memory_usage = metrics["memory_usage"]
            target_queue_depth = metrics.get("target_queue_depth", 10)

            logger.debug(
                f"[PIDController] Stage '{stage}': queue_depth={queue_depth}, "
                f"target_queue_depth={target_queue_depth}, replicas={replicas}, "
                f"memory_usage={memory_usage} MB, throughput={throughput}"
            )

            # Track idle cycles: if queue is zero, increment idle counter; else reset.
            if queue_depth == 0:
                self.idle_cycles[stage] += 1
            else:
                self.idle_cycles[stage] = 0

            # Compute penalty based on idle cycles.
            penalty = self.penalty_factor * self.idle_cycles[stage]
            logger.debug(f"[PIDController] Stage '{stage}': idle_cycles={self.idle_cycles[stage]}, penalty={penalty}")

            current_memory_per_replica = memory_usage / max(replicas, 1)
            self.memory_history[stage].append(current_memory_per_replica)
            dynamic_cost = self.estimate_dynamic_memory(stage, self.memory_history[stage])
            memory_rate = self.estimate_memory_rate(stage, current_memory_per_replica)

            logger.debug(
                f"[PIDController] Stage '{stage}': current_memory_per_replica={current_memory_per_replica:.2f} MB, "
                f"dynamic_cost={dynamic_cost:.2f} MB, memory_rate={memory_rate:.4f}"
            )

            # Proactive downscaling if memory is growing too fast.
            if memory_rate > self.memory_rate_threshold:
                new_replicas = max(1, replicas - 1)
                logger.debug(
                    f"[PIDController] Stage '{stage}': memory_rate {memory_rate:.4f} exceeds threshold "
                    f"{self.memory_rate_threshold}. Reducing replicas from {replicas} to {new_replicas}"
                )
                stage_adjustments[stage] = new_replicas
                continue

            # Compute error with penalty: subtract penalty from the typical error.
            error = (queue_depth - target_queue_depth) - penalty
            self.integral_error[stage] += error
            derivative = error - self.prev_error[stage]
            self.prev_error[stage] = error

            logger.debug(
                f"[PIDController] Stage '{stage}': error (after penalty)={error},"
                f" integral_error={self.integral_error[stage]}, "
                f"derivative={derivative}"
            )

            delta = self.kp * error + self.ki * self.integral_error[stage] + self.kd * derivative

            # Apply a boost if error is positive to scale up more aggressively initially.
            if error > 0:
                aggressive_delta = delta * 2
                logger.debug(
                    f"[PIDController] Stage '{stage}': Positive error detected. "
                    f"Boosting delta from {delta:.2f} to {aggressive_delta:.2f}"
                )
                delta = aggressive_delta

            delta = int(round(delta))
            logger.debug(f"[PIDController] Stage '{stage}': Final delta (after rounding) is {delta}")

            new_replicas = max(1, replicas + delta)
            logger.debug(f"[PIDController] Stage '{stage}': Tentative new_replicas={new_replicas}")

            projected_memory_usage = new_replicas * dynamic_cost
            if projected_memory_usage > memory_budget_remaining:
                max_allowed = memory_budget_remaining // dynamic_cost
                new_replicas = max(1, min(new_replicas, int(max_allowed)))
                logger.debug(
                    f"[PIDController] Stage '{stage}': projected_memory_usage ({projected_memory_usage:.2f} MB) "
                    f"exceeds memory_budget_remaining. Adjusting new_replicas to {new_replicas}"
                )

            if total_replicas - replicas + new_replicas > self.max_replicas:
                allowed = self.max_replicas - (total_replicas - replicas)
                new_replicas = max(1, min(new_replicas, allowed))
                logger.debug(
                    f"[PIDController] Stage '{stage}': Total replicas constraint exceeded. "
                    f"Limiting new_replicas to {new_replicas}"
                )

            logger.debug(f"[PIDController] Stage '{stage}': Final replica adjustment: {replicas} -> {new_replicas}")
            stage_adjustments[stage] = new_replicas

        return stage_adjustments
