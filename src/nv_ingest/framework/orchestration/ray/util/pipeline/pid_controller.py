# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Deque, Optional

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
        # Initialize with None so that the first measurement returns 0 rate.
        self.memory_rate_history: Dict[str, Optional[float]] = {}
        self.window_size = window_size
        self.memory_rate_threshold = memory_rate_threshold

        self.idle_cycles: Dict[str, int] = {}
        self.penalty_factor = penalty_factor

    def initialize_stage(self, stage: str) -> None:
        """Private helper: Initialize controller state for a given stage."""
        self.integral_error[stage] = 0.0
        self.prev_error[stage] = 0.0
        self.memory_history[stage] = deque(maxlen=self.window_size)
        self.memory_rate_history[stage] = None  # No previous measurement.
        self.idle_cycles[stage] = 0

    def _estimate_dynamic_memory(self, stage: str, memory_samples: Deque[int]) -> float:
        """Private helper: Estimate dynamic memory cost based on recent samples."""
        if memory_samples:
            return np.mean(memory_samples)
        return self.stage_cost_estimates[stage]

    def _enforce_replica_bounds(
        self,
        tentative_replicas: int,
        queue_depth: int,
        min_replicas_metric: int,
        max_replicas_metric: int,
        pipeline_in_flight: int,
    ) -> int:
        """
        Private helper to enforce per-stage replica bounds.

        If any tasks are in-flight in the pipeline (pipeline_in_flight > 0),
        the stage must maintain at least one replica.
        Otherwise, if the stage is idle (queue_depth is 0) and min_replicas is 0, it may scale down to 0.
        In all cases, the result is capped by max_replicas_metric.

        Parameters
        ----------
        tentative_replicas : int
            Computed tentative new replica count.
        queue_depth : int
            Current queue depth for this stage.
        min_replicas_metric : int
            Minimum replicas allowed for the stage.
        max_replicas_metric : int
            Maximum replicas allowed for the stage.
        pipeline_in_flight : int
            Total in-flight tasks across the pipeline.

        Returns
        -------
        int
            Final new replica count after enforcing bounds.
        """
        if pipeline_in_flight > 0:
            lower_bound = 1
        elif queue_depth == 0 and min_replicas_metric == 0:
            lower_bound = 0
        else:
            lower_bound = max(1, min_replicas_metric)
        new_replicas = max(lower_bound, tentative_replicas)
        new_replicas = min(new_replicas, max_replicas_metric)
        return new_replicas

    def _estimate_memory_rate(self, stage: str, new_value: float) -> float:
        """
        Private helper: Estimate the memory growth rate for a stage.

        Returns 0 if no previous measurement exists.
        """
        if self.memory_rate_history.get(stage) is None:
            self.memory_rate_history[stage] = new_value
            return 0.0
        old_value = self.memory_rate_history[stage]
        rate = new_value - old_value
        self.memory_rate_history[stage] = new_value
        return rate

    def update(self, stage_metrics: Dict[str, Dict[str, Any]], current_memory_usage: int) -> Dict[str, int]:
        """
        Update the PID controller based on stage metrics and current memory usage.

        The update process follows these steps:

        STEP 1: Global Calculations.
          - Compute total replicas and the remaining memory budget.

        STEP 2: Per-Stage Processing.
          For each stage:
          a) Extract stage-specific metrics: queue depth, throughput, replica count,
             memory usage, target queue depth, min/max replicas, in-flight tasks, and
             pipeline_in_flight (global in-flight count).
          b) Update the idle cycle counter and compute a penalty.
          c) Estimate the current memory cost per replica and calculate dynamic memory cost.
          d) Compute the memory growth rate.
          e) If the memory rate exceeds the threshold, propose a one-replica reduction (subject to bounds).
          f) Otherwise, compute the error as (queue_depth - target) adjusted by the penalty.
          g) Update integral and derivative errors, and compute the PID delta.
          h) If the error is positive, boost the delta to scale up more aggressively.
          i) Compute a tentative new replica count.
          j) Adjust the tentative count based on the memory budget and global maximum replicas.
          k) Enforce per-stage replica bounds using the helper function, which uses the global
             pipeline_in_flight value.

        STEP 3: Global Consistency.
          - If any stage with 0 replicas is scaling up (i.e. new count > 0), force all stages that are at 0
            to have at least 1 replica, ensuring the pipeline remains active.

        Parameters
        ----------
        stage_metrics : Dict[str, Dict[str, Any]]
            Dictionary of stage metrics. Each stage metric must include:
                - queue_depth: Current queue depth.
                - throughput: Throughput (currently unused).
                - replicas: Current number of replicas.
                - memory_usage: Current memory usage in MB.
                - target_queue_depth: Desired queue depth (default: 10).
                - min_replicas: Minimum number of replicas allowed.
                - max_replicas: Maximum number of replicas allowed.
                - in_flight: Number of tasks in-flight for this stage.
                - pipeline_in_flight: Total in-flight tasks across the entire pipeline.
        current_memory_usage : int
            Current global memory usage in MB.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping stage names to new replica counts.
        """
        # STEP 1: Global calculations.
        total_replicas = sum(m["replicas"] for m in stage_metrics.values())
        memory_budget_remaining = self.memory_threshold - current_memory_usage
        logger.debug(
            f"[PIDController] Total replicas: {total_replicas}, "
            f"Current memory usage: {current_memory_usage} MB, "
            f"Memory budget remaining: {memory_budget_remaining} MB"
        )

        stage_adjustments: Dict[str, int] = {}

        # STEP 2: Process each stage individually.
        for stage, metrics in stage_metrics.items():
            # a) Extract metrics.
            queue_depth = metrics["queue_depth"]
            throughput = metrics["throughput"]  # Currently unused.
            replicas = metrics["replicas"]
            memory_usage = metrics["memory_usage"]
            target_queue_depth = metrics.get("target_queue_depth", 10)
            min_replicas_metric = metrics.get("min_replicas", 1)
            max_replicas_metric = metrics.get("max_replicas", 1)
            in_flight = metrics.get("in_flight", 0)
            pipeline_in_flight = metrics.get("pipeline_in_flight", 0)

            logger.debug(
                f"[PIDController] Stage '{stage}': queue_depth={queue_depth}, target_queue_depth={target_queue_depth}, "
                f"replicas={replicas}, memory_usage={memory_usage} MB, throughput={throughput}, "
                f"min_replicas={min_replicas_metric}, max_replicas={max_replicas_metric}, "
                f"in_flight={in_flight}, pipeline_in_flight={pipeline_in_flight}"
            )

            # b) Update idle cycle counter.
            if queue_depth == 0:
                self.idle_cycles[stage] += 1
            else:
                self.idle_cycles[stage] = 0
            penalty = self.penalty_factor * self.idle_cycles[stage]
            logger.debug(f"[PIDController] Stage '{stage}': idle_cycles={self.idle_cycles[stage]}, penalty={penalty}")

            # c) Estimate memory cost.
            current_memory_per_replica = memory_usage / max(replicas, 1)
            self.memory_history[stage].append(current_memory_per_replica)
            dynamic_cost = self._estimate_dynamic_memory(stage, self.memory_history[stage])
            # d) Compute memory growth rate.
            memory_rate = self._estimate_memory_rate(stage, current_memory_per_replica)
            logger.debug(
                f"[PIDController] Stage '{stage}': current_memory_per_replica={current_memory_per_replica:.2f} MB, "
                f"dynamic_cost={dynamic_cost:.2f} MB, memory_rate={memory_rate:.4f}"
            )

            # e) If memory is growing too fast, propose a one-replica reduction.
            if memory_rate > self.memory_rate_threshold:
                tentative_replicas = replicas - 1
                new_replicas = self._enforce_replica_bounds(
                    tentative_replicas, queue_depth, min_replicas_metric, max_replicas_metric, pipeline_in_flight
                )
                logger.warning(
                    f"[PIDController] Stage '{stage}': memory_rate {memory_rate:.4f} exceeds threshold "
                    f"{self.memory_rate_threshold}. Reducing replicas from {replicas} to {new_replicas}"
                )
                stage_adjustments[stage] = new_replicas
                continue

            # f) Compute error based on (queue_depth - target) adjusted by the penalty.
            error = (queue_depth - target_queue_depth) - penalty
            self.integral_error[stage] += error
            derivative = error - self.prev_error[stage]
            self.prev_error[stage] = error
            logger.debug(
                f"[PIDController] Stage '{stage}': error (after penalty)={error}, "
                f"integral_error={self.integral_error[stage]}, derivative={derivative}"
            )

            # g) Compute the PID delta.
            delta = self.kp * error + self.ki * self.integral_error[stage] + self.kd * derivative

            # h) Boost positive error to scale up more aggressively.
            if error > 0:
                aggressive_delta = delta * 2
                logger.debug(
                    f"[PIDController] Stage '{stage}': Positive error detected. "
                    f"Boosting delta from {delta:.2f} to {aggressive_delta:.2f}"
                )
                delta = aggressive_delta
            delta = int(round(delta))
            logger.debug(f"[PIDController] Stage '{stage}': Final delta (after rounding) is {delta}")

            # i) Compute tentative new replica count.
            tentative_replicas = replicas + delta
            logger.debug(f"[PIDController] Stage '{stage}': Tentative new_replicas={tentative_replicas}")

            # j) Adjust for memory budget.
            projected_memory_usage = tentative_replicas * dynamic_cost
            if projected_memory_usage > memory_budget_remaining:
                max_allowed = memory_budget_remaining // dynamic_cost
                tentative_replicas = min(tentative_replicas, int(max_allowed))
                logger.debug(
                    f"[PIDController] Stage '{stage}': "
                    f"projected_memory_usage ({projected_memory_usage:.2f} MB) exceeds "
                    f"memory_budget_remaining. Adjusting new_replicas to {tentative_replicas}"
                )

            # k) Enforce global maximum replicas constraint.
            if total_replicas - replicas + tentative_replicas > self.max_replicas:
                allowed = self.max_replicas - (total_replicas - replicas)
                tentative_replicas = min(tentative_replicas, allowed)
                logger.debug(
                    f"[PIDController] Stage '{stage}': Total replicas constraint exceeded. "
                    f"Limiting new_replicas to {tentative_replicas}"
                )

            # l) Enforce per-stage replica bounds using the helper (which uses pipeline_in_flight).
            new_replicas = self._enforce_replica_bounds(
                tentative_replicas, queue_depth, min_replicas_metric, max_replicas_metric, pipeline_in_flight
            )
            logger.debug(f"[PIDController] Stage '{stage}': Final replica adjustment: {replicas} -> {new_replicas}")
            stage_adjustments[stage] = new_replicas

        # STEP 3: Global consistency.
        # If any stage with 0 replicas is scaling up (i.e. its new value > 0),
        # then force all stages with 0 replicas to have at least 1 replica.
        scale_up_triggered = any(
            (metrics["replicas"] == 0 and stage_adjustments.get(stage, 0) > 0)
            for stage, metrics in stage_metrics.items()
        )
        if scale_up_triggered:
            for stage, metrics in stage_metrics.items():
                if metrics["replicas"] == 0 and stage_adjustments.get(stage, 0) == 0:
                    logger.debug(
                        f"[PIDController] Stage '{stage}' currently has 0 replicas but pipeline has in-flight tasks. "
                        "Forcing minimum 1 replica."
                    )
                    stage_adjustments[stage] = 1

        return stage_adjustments
