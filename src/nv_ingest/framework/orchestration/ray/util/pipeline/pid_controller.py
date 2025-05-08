# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass

import numpy as np
from collections import deque
from typing import Dict, Any, Deque, List, Tuple, Optional

from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STAGE_COST_MB = 5000.0  # Fallback memory cost


@dataclass
class StagePIDProposal:
    """Holds the initial proposal from the PID controller for a single stage."""

    name: str
    current_replicas: int
    proposed_replicas: int  # Initial proposal based on PID / stage rate limit
    # Conservative cost estimate (max(dynamic_avg, static)) used for projections
    conservative_cost_estimate: float
    metrics: Dict[str, Any]  # Original metrics for context


class PIDController:
    """
    Calculates initial replica adjustment proposals based on PID control logic.

    This controller focuses on the core PID algorithm reacting to the error
    between the current state (queue depth) and the desired state (target depth),
    adjusted by an idle penalty. It tracks memory usage per replica to provide
    a dynamic cost estimate for the ResourceConstraintManager.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,  # Currently unused in delta calculation
        stage_cost_estimates: Dict[str, int],  # Static estimates (MB)
        target_queue_depth: int = 0,
        window_size: int = 10,
        penalty_factor: float = 0.0005,
        error_boost_factor: float = 1.5,
    ):
        """
        Initializes the PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain. Reacts to the current error magnitude.
        ki : float
            Integral gain. Accumulates past errors to eliminate steady-state offsets.
        kd : float
            Derivative gain. Reacts to the rate of change of the error.
            (Currently set to 0 in internal calculations).
        stage_cost_estimates : Dict[str, int]
            Static estimated memory cost (in MB) per replica for each stage.
            Used as a fallback and minimum for dynamic estimates.
        target_queue_depth : int, optional
            Default target queue depth for stages if not specified in metrics,
            by default 0. The PID loop tries to drive the queue depth towards
            this value.
        window_size : int, optional
            Number of recent samples used for dynamic memory cost estimation
            per replica, by default 10.
        penalty_factor : float, optional
            Multiplier applied to the number of consecutive idle cycles for a
            stage. The resulting penalty effectively lowers the target queue
            depth for idle stages, encouraging scale-down, by default 0.5.
        error_boost_factor : float, optional
            Factor to multiply the raw PID delta when the error is positive
            (queue > target), potentially speeding up scale-up response,
            by default 1.5.
        """
        self.kp = kp
        self.ki = ki
        self.kd = 0.0  # Explicitly disable derivative term for now
        self.target_queue_depth = target_queue_depth
        self.error_boost_factor = error_boost_factor

        # Per-Stage State
        self.stage_cost_estimates = {
            name: float(max(cost, 1.0)) for name, cost in stage_cost_estimates.items()  # Ensure float and min 1MB
        }
        self.integral_error: Dict[str, float] = {}
        self.prev_error: Dict[str, float] = {}
        self.memory_history: Dict[str, Deque[float]] = {}  # Per-replica memory history (MB)
        self.idle_cycles: Dict[str, int] = {}

        # Per-Stage Config
        self.window_size = window_size
        self.penalty_factor = penalty_factor

    # --- Private Methods ---

    def _initialize_stage_state(self, stage: str) -> None:
        """Initializes controller state variables for a newly seen stage."""
        if stage not in self.integral_error:
            logger.debug(f"[PID-{stage}] Initializing state.")
            self.integral_error[stage] = 0.0
            self.prev_error[stage] = 0.0
            self.memory_history[stage] = deque(maxlen=self.window_size)
            self.idle_cycles[stage] = 0
            # Ensure static cost estimate exists, provide default if missing
            if stage not in self.stage_cost_estimates:
                logger.warning(f"[PID-{stage}] Missing static cost estimate. Using default {DEFAULT_STAGE_COST_MB}MB.")
                self.stage_cost_estimates[stage] = DEFAULT_STAGE_COST_MB

    def _get_conservative_cost_estimate(self, stage: str) -> float:
        """
        Estimates dynamic memory cost, using static estimate as a floor/max.

        Returns the maximum of the recent average dynamic cost per replica
        and the static estimate provided during initialization. This provides
        a conservative value for resource projection.

        Parameters
        ----------
        stage : str
            The name of the stage.

        Returns
        -------
        float
            The conservative memory cost estimate in MB per replica.
        """
        static_cost = self.stage_cost_estimates.get(stage, DEFAULT_STAGE_COST_MB)
        memory_samples = self.memory_history.get(stage)

        # Use numpy.mean if samples exist, otherwise fallback to static
        if memory_samples and len(memory_samples) > 0:
            try:
                dynamic_avg = float(np.mean(memory_samples))
                # Use max(dynamic, static) for projection, enforce min 1MB
                cost = max(dynamic_avg, static_cost, 1.0)
                return cost
            except Exception as e:
                logger.error(
                    f"[PID-{stage}] Error calculating mean of memory samples: {e}. Falling back to static cost.",
                    exc_info=False,
                )
                return max(static_cost, 1.0)  # Fallback safely
        return max(static_cost, 1.0)  # Fallback to static estimate if no history

    # --- Public Method ---

    def calculate_initial_proposals(self, stage_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, StagePIDProposal]:
        """
        Calculates initial, unconstrained replica proposals for each stage.

        Iterates through each stage, calculates its PID error and delta based
        on queue depth and target, and returns the initial proposals
        without considering global constraints. Includes dynamic cost estimates.

        Parameters
        ----------
        stage_metrics : Dict[str, Dict[str, Any]]
            Dictionary mapping stage names to their current metrics. Expected keys
            per stage: 'replicas', 'queue_depth'. Optional: 'memory_usage',
            'target_queue_depth', 'processing', 'min_replicas', 'max_replicas'.

        Returns
        -------
        Dict[str, StagePIDProposal]
            Dictionary mapping stage names to their initial proposals, including
            current/proposed replicas, cost estimates, and original metrics.
        """
        logger.debug("--- PID Controller: Calculating Initial Proposals ---")
        proposals: Dict[str, StagePIDProposal] = {}

        for stage, metrics in stage_metrics.items():
            # Ensure state exists and initialize if necessary
            self._initialize_stage_state(stage)

            # --- Extract data and calculate current memory state ---
            replicas = metrics.get("replicas", 0)
            # Start with static cost as initial guess if no memory_usage provided
            initial_cost_guess = self.stage_cost_estimates.get(stage, DEFAULT_STAGE_COST_MB)
            memory_usage = metrics.get("memory_usage", initial_cost_guess * max(replicas, 1))
            # Calculate memory per replica safely (avoid division by zero)
            current_memory_per_replica = memory_usage / max(replicas, 1.0)

            # Update memory history *before* calculating the conservative cost for *this* cycle's proposal
            self.memory_history[stage].append(current_memory_per_replica)
            # Recalculate conservative cost *after* updating history for the proposal
            conservative_cost = self._get_conservative_cost_estimate(stage)

            # --- PID Calculation ---
            queue_depth = metrics.get("queue_depth", 0)
            # Allow target override per stage, else use controller default
            target_queue_depth = metrics.get("target_queue_depth", self.target_queue_depth)
            min_replicas_metric = metrics.get("min_replicas", 0)
            max_replicas_metric = metrics.get("max_replicas", 1)  # Default max should likely be higher

            # Idle penalty calculation
            if queue_depth == 0 and metrics.get("processing", 0) == 0:
                self.idle_cycles[stage] += 1
            else:
                self.idle_cycles[stage] = 0

            # Limit how much penalty can reduce the effective target below zero
            penalty = self.penalty_factor * (self.idle_cycles[stage] ** 2.0)

            # Error calculation (Queue deviation from target, adjusted by idle penalty)
            error = (queue_depth - target_queue_depth) - penalty

            # Integral term update with basic anti-windup
            # Don't accumulate integral if already at boundary AND error pushes further past boundary
            should_accumulate_integral = True
            if replicas >= max_replicas_metric and error > 0:  # At max replicas, still have backlog
                should_accumulate_integral = False
                logger.debug(
                    f"[PID-{stage}] At max replicas ({replicas}) with positive error ({error:.2f}), pausing integral."
                )
            elif (
                replicas <= min_replicas_metric and error < 0
            ):  # At min replicas, queue is below target (or penalty active)
                should_accumulate_integral = False
                logger.debug(
                    f"[PID-{stage}] At min replicas ({replicas}) with negative error ({error:.2f}), pausing integral."
                )

            if should_accumulate_integral:
                self.integral_error[stage] += error

            # Update previous error state for potential future derivative use
            self.prev_error[stage] = error

            # --- Delta Calculation ---
            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error[stage]
            # derivative_term = self.kd * derivative # Still disabled

            # Combine terms
            raw_delta = proportional_term + integral_term  # + derivative_term

            # Boost scale-up signals (positive error means queue > target)
            if error > 0:
                boosted_delta = raw_delta * self.error_boost_factor
                logger.debug(f"[PID-{stage}] Boosting positive error delta: {raw_delta:.3f} -> {boosted_delta:.3f}")
                raw_delta = boosted_delta

            # Round to get integer replica change
            delta_replicas = int(round(raw_delta))
            proposed_replicas = replicas + delta_replicas

            logger.debug(
                f"[PID-{stage}] R={replicas}, Q={queue_depth}, Tgt={target_queue_depth},"
                f" Idle={self.idle_cycles[stage]}, Pen={penalty:.2f} -> "
                f"Err={error:.2f}, P={proportional_term:.2f}, I={integral_term:.2f}"
                f" (Acc={self.integral_error[stage]:.2f}) -> "
                f"DeltaR={delta_replicas}, RawProp={proposed_replicas}"
            )

            # --- Create Final Proposal Object for this Stage ---
            proposal = StagePIDProposal(
                name=stage,
                current_replicas=replicas,
                proposed_replicas=proposed_replicas,
                conservative_cost_estimate=conservative_cost,  # Use updated cost
                metrics=metrics,  # Pass along original metrics
            )

            proposals[stage] = proposal

        logger.debug("--- PID Controller: Initial Proposals Calculated ---")
        return proposals


class ResourceConstraintManager:
    """
    Applies global resource constraints and safety checks to initial proposals.

    Takes the initial replica proposals generated by the PIDController and
    adjusts them based on global limits (max replicas, available CPU cores based
    on affinity, memory budget with safety buffer), and ensures pipeline
    consistency (zero-replica safety). It allocates limited resources
    proportionally if multiple stages request scale-ups simultaneously.

    If current global memory usage exceeds the effective limit, it aggressively
    scales down stages starting with the highest replica counts.
    """

    def __init__(
        self,
        max_replicas: int,
        memory_threshold: int,
        estimated_edge_cost_mb: int,
        memory_safety_buffer_fraction: float,
    ):
        """
        Initializes the Resource Constraint Manager using CoreCountDetector.

        Parameters are the same as before.
        """
        if not (0.0 <= memory_safety_buffer_fraction < 1.0):
            raise ValueError("memory_safety_buffer_fraction must be between 0.0 and 1.0")

        self.max_replicas = max_replicas
        self.memory_threshold_mb = memory_threshold
        self.estimated_edge_cost_mb = estimated_edge_cost_mb  # Keep track, though unused
        self.memory_safety_buffer_fraction = memory_safety_buffer_fraction
        self.effective_memory_limit_mb = self.memory_threshold_mb * (1.0 - self.memory_safety_buffer_fraction)

        core_detector = SystemResourceProbe()  # Instantiate the detector
        self.available_cores: Optional[float] = core_detector.get_effective_cores()
        self.core_detection_details: Dict[str, Any] = core_detector.get_details()

        # Determine a practical replica limit based on cores (optional, but often useful)
        self.core_based_replica_limit: Optional[int] = None
        if self.available_cores is not None and self.available_cores > 0:
            self.core_based_replica_limit = math.floor(self.available_cores)
        else:
            self.core_based_replica_limit = None  # Treat as unlimited if detection failed

        logger.info(
            f"[ConstraintMgr] Initialized. MaxReplicas={max_replicas}, "
            f"EffectiveCoreLimit={self.available_cores:.2f} "  # Log the potentially fractional value
            f"(Method: {self.core_detection_details.get('detection_method')}), "
            f"CoreBasedReplicaLimit={self.core_based_replica_limit}, "  # Log the derived integer limit
            f"MemThreshold={memory_threshold}MB, "
            f"EffectiveLimit={self.effective_memory_limit_mb:.1f}MB "
        )
        logger.debug(f"[ConstraintMgr] Core detection details: {self.core_detection_details}")

    # --- Private Methods ---

    @staticmethod
    def _get_effective_min_replicas(stage_name: str, metrics: Dict[str, Any], pipeline_in_flight: int) -> int:
        """Helper to calculate the effective minimum replicas for a stage."""
        min_replicas_metric = metrics.get("min_replicas", 0)
        # If the pipeline is active globally, enforce a minimum of 1 replica,
        # unless min_replicas dictates higher.
        if pipeline_in_flight > 0:
            return max(1, min_replicas_metric)
        else:  # Pipeline is globally idle
            # Allow scaling down to zero ONLY if the pipeline is idle AND min_replicas allows it.
            return min_replicas_metric

    def _apply_aggressive_memory_scale_down(
        self,
        current_proposals: Dict[str, int],
        initial_proposals_meta: Dict[str, StagePIDProposal],
        current_global_memory_usage: int,
        pipeline_in_flight_global: int,
    ) -> Dict[str, int]:
        """
        If current memory exceeds the effective limit, force scale-downs.

        Reduces replicas from stages with the highest counts first, respecting
        their effective minimum replicas, until memory is below the limit or
        no more reductions are possible.

        Returns:
            Dict[str, int]: Updated replica proposals after aggressive scale-down.
        """
        if current_global_memory_usage <= self.effective_memory_limit_mb:
            return current_proposals

        memory_overrun = current_global_memory_usage - self.effective_memory_limit_mb
        logger.warning(
            f"[ConstraintMgr] Aggressive Scale-Down Triggered: "
            f"Current Mem ({current_global_memory_usage:.1f}MB) > Effective Limit"
            f" ({self.effective_memory_limit_mb:.1f}MB). "
            f"Need to reduce by {memory_overrun:.1f}MB."
        )

        adjusted_proposals = current_proposals.copy()

        # Identify candidates for scale-down
        candidates = []
        for name, current_replicas in adjusted_proposals.items():
            proposal_meta = initial_proposals_meta.get(name)
            if not proposal_meta:
                logger.error(f"[ConstraintMgr] Missing metadata for stage {name} during aggressive scale-down.")
                continue

            effective_min = self._get_effective_min_replicas(name, proposal_meta.metrics, pipeline_in_flight_global)
            cost_estimate = proposal_meta.conservative_cost_estimate

            if current_replicas > effective_min:
                candidates.append(
                    {
                        "name": name,
                        "replicas": current_replicas,
                        "cost": cost_estimate if cost_estimate > 0 else 1e-6,
                        "effective_min": effective_min,
                    }
                )

        # Sort candidates: primarily by replica count desc, secondarily by cost desc
        candidates.sort(key=lambda x: (x["replicas"], x["cost"]), reverse=True)

        if not candidates:
            logger.warning("[ConstraintMgr] Aggressive Scale-Down: No eligible stages found to reduce replicas.")
            return adjusted_proposals

        # Iteratively reduce replicas
        memory_reduced = 0.0
        stages_reduced = []
        while memory_overrun > 0 and candidates:
            target_stage = candidates[0]
            name = target_stage["name"]
            new_replica_count = target_stage["replicas"] - 1
            mem_saved_this_step = target_stage["cost"]

            logger.debug(
                f"[ConstraintMgr-{name}] Aggressive Scale-Down: Reducing replica from {target_stage['replicas']} ->"
                f" {new_replica_count} (saves ~{mem_saved_this_step:.2f}MB)"
            )
            adjusted_proposals[name] = new_replica_count
            memory_overrun -= mem_saved_this_step
            memory_reduced += mem_saved_this_step
            stages_reduced.append(name)

            target_stage["replicas"] = new_replica_count
            if new_replica_count <= target_stage["effective_min"]:
                candidates.pop(0)
            else:
                candidates.sort(key=lambda x: (x["replicas"], x["cost"]), reverse=True)

        if memory_overrun > 0:
            logger.warning(
                f"[ConstraintMgr] Aggressive Scale-Down: Completed, but still over memory limit by"
                f" {memory_overrun:.1f}MB. Reduced total {memory_reduced:.1f}MB from stages:"
                f" {list(set(stages_reduced))}."
            )
        else:
            logger.info(
                f"[ConstraintMgr] Aggressive Scale-Down: Completed. Reduced total {memory_reduced:.1f}MB from stages:"
                f" {list(set(stages_reduced))}. Projected memory now below limit."
            )

        return adjusted_proposals

    def _apply_global_constraints_proportional(
        self,
        tentative_proposals: Dict[str, int],
        initial_proposals_meta: Dict[str, StagePIDProposal],
        current_global_memory_usage: int,
    ) -> Dict[str, int]:
        """Applies global replica/memory/core limits proportionally to scale-up requests."""
        upscale_requests: List[Tuple[str, int, float]] = []
        total_requested_increase_replicas = 0
        total_projected_mem_increase = 0.0
        baseline_total_replicas = sum(prop.current_replicas for prop in initial_proposals_meta.values())

        # Separate proposals based on ORIGINAL intent vs current state
        for name, proposal_meta in initial_proposals_meta.items():
            initial_delta = proposal_meta.proposed_replicas - proposal_meta.current_replicas
            if initial_delta > 0:
                cost_estimate = proposal_meta.conservative_cost_estimate
                upscale_requests.append((name, initial_delta, cost_estimate))
                total_requested_increase_replicas += initial_delta
                total_projected_mem_increase += initial_delta * cost_estimate

        logger.debug(
            f"[ConstraintMgr-Proportional] Baseline replicas: {baseline_total_replicas}. "
            f"Initial upscale intent: {len(upscale_requests)} stages, "
            f"ΔR={total_requested_increase_replicas}, ΔMem={total_projected_mem_increase:.2f}MB."
        )

        # Check global constraints for the original requested increase
        scale_up_allowed = True
        reduction_factor = 1.0

        if total_requested_increase_replicas <= 0:
            scale_up_allowed = False
            logger.debug("[ConstraintMgr-Proportional] No scale-up originally requested.")
        else:
            projected_total_replicas = baseline_total_replicas + total_requested_increase_replicas

            if projected_total_replicas > self.max_replicas:
                allowed_increase_max_r = max(0, self.max_replicas - baseline_total_replicas)
                factor_max_r = (
                    allowed_increase_max_r / total_requested_increase_replicas
                    if total_requested_increase_replicas > 0
                    else 0.0
                )
                reduction_factor = min(reduction_factor, factor_max_r)
                logger.warning(
                    f"[ConstraintMgr-Proportional] Max replicas constraint potentially hit by initial request "
                    f"({projected_total_replicas}/{self.max_replicas}). Limiting factor: {factor_max_r:.3f}"
                )

            if self.available_cores is not None and projected_total_replicas > self.available_cores:
                # Allowed increase is float, based on fractional limit
                allowed_increase_cores = max(0.0, self.available_cores - baseline_total_replicas)
                factor_cores = (
                    allowed_increase_cores / total_requested_increase_replicas
                    if total_requested_increase_replicas > 0
                    else 0.0
                )
                reduction_factor = min(reduction_factor, factor_cores)
                logger.warning(
                    f"[ConstraintMgr-Proportional] Effective core limit potentially hit by initial request "
                    # Log the comparison accurately
                    f"({projected_total_replicas} replicas vs {self.available_cores:.2f} effective cores)."
                    f" Limiting factor: {factor_cores:.3f}"
                )

            # Check Effective Memory Limit Constraint (Unchanged)
            projected_total_global_memory = current_global_memory_usage + total_projected_mem_increase
            if projected_total_global_memory > self.effective_memory_limit_mb:
                allowed_mem_increase = max(0.0, self.effective_memory_limit_mb - current_global_memory_usage)
                factor_mem = (
                    allowed_mem_increase / total_projected_mem_increase if total_projected_mem_increase > 0 else 0.0
                )
                reduction_factor = min(reduction_factor, factor_mem)
                logger.warning(
                    f"[ConstraintMgr-Proportional] Effective memory limit potentially hit by initial request "
                    f"({projected_total_global_memory:.1f}/{self.effective_memory_limit_mb:.1f}MB). "
                    f"Limiting factor: {factor_mem:.3f}"
                )

            # Determine if any scale up is possible after all checks
            scale_up_allowed = reduction_factor > 0.001  # Epsilon for floating point comparison

        # Apply reduction factor to the original requested increase (Logic Unchanged, relies on math.floor later)
        final_proposals = tentative_proposals.copy()

        if not scale_up_allowed or reduction_factor <= 0.001:
            logger.debug("[ConstraintMgr-Proportional] Blocking/zeroing scale-up potential based on constraints.")
            # Reset proposals for stages that initially requested scale-up back to their
            # count from the tentative_proposals input (after aggressive scale-down).
            for name, requested_increase, _ in upscale_requests:
                final_proposals[name] = tentative_proposals[name]

        elif reduction_factor < 1.0:
            logger.info(
                f"[ConstraintMgr-Proportional] Reducing scale-up potential by factor {reduction_factor:.3f}"
                f" relative to original request due to global limits."
            )
            for name, requested_increase, _ in upscale_requests:
                proposal_meta = initial_proposals_meta[name]
                # Floor ensures we only add whole replicas, respecting the proportional reduction
                allowed_increase = math.floor(requested_increase * reduction_factor)
                target_replicas = proposal_meta.current_replicas + allowed_increase  # Target relative to original count
                current_val_after_aggressive = tentative_proposals.get(name, proposal_meta.current_replicas)

                # Final replicas should be the minimum of the proportionally calculated target
                # and the value after any aggressive scale-down, ensuring we don't scale up past
                # what aggressive scale-down might have enforced.
                final_replicas_for_stage = min(target_replicas, current_val_after_aggressive)

                if final_replicas_for_stage != current_val_after_aggressive:
                    logger.debug(
                        f"[ConstraintMgr-{name}] Proportional Adjustment: OrigReq={requested_increase}, "
                        f"AllowedInc={allowed_increase}, CalcTarget={target_replicas}, "
                        f"AggressiveVal={current_val_after_aggressive} -> Final={final_replicas_for_stage}"
                    )
                final_proposals[name] = final_replicas_for_stage

        return final_proposals

    def _enforce_replica_bounds(
        self, stage_name: str, tentative_replicas: int, metrics: Dict[str, Any], pipeline_in_flight: int
    ) -> int:
        """Enforces per-stage min/max replica bounds and zero-replica safety logic."""
        max_replicas_metric = metrics.get("max_replicas", 1)
        lower_bound = self._get_effective_min_replicas(stage_name, metrics, pipeline_in_flight)
        bounded_replicas = max(lower_bound, tentative_replicas)
        final_replicas = min(bounded_replicas, max_replicas_metric)

        if final_replicas != tentative_replicas:
            min_replicas_metric = metrics.get("min_replicas", 0)
            logger.debug(
                f"[ConstraintMgr-{stage_name}] Bounds Applied: Tentative={tentative_replicas} ->"
                f" Final={final_replicas} "
                f"(MinConfig={min_replicas_metric}, MaxConfig={max_replicas_metric}, "
                f"EffectiveLowerBound={lower_bound}, PipeInFlight={pipeline_in_flight})"
            )
        elif final_replicas == 0 and lower_bound == 0:
            logger.debug(f"[ConstraintMgr-{stage_name}] Allowing scale to 0: Pipeline Idle and MinReplicas=0.")

        return final_replicas

    @staticmethod
    def _apply_global_consistency(
        final_adjustments: Dict[str, int], initial_proposals: Dict[str, StagePIDProposal]
    ) -> None:
        """Ensures pipeline doesn't get stuck if one stage scales up from zero."""
        scale_up_from_zero_triggered = any(
            (prop.current_replicas == 0 and final_adjustments.get(name, 0) > 0)
            for name, prop in initial_proposals.items()
        )

        if scale_up_from_zero_triggered:
            logger.debug("[ConstraintMgr] Wake-up consistency: Ensuring no stages stuck at zero.")
            for name, prop in initial_proposals.items():
                if prop.current_replicas == 0 and final_adjustments.get(name, 0) == 0:
                    min_r = prop.metrics.get("min_replicas", 0)
                    max_r = prop.metrics.get("max_replicas", 1)
                    target = max(1, min_r)
                    final_target = min(target, max_r)
                    if final_target > 0:
                        logger.info(
                            f"[ConstraintMgr-{name}] Forcing minimum {final_target} replica due to global wake-up."
                        )
                        final_adjustments[name] = final_target

    # --- Public Method ---

    def apply_constraints(
        self,
        initial_proposals: Dict[str, "StagePIDProposal"],
        global_in_flight: int,
        current_global_memory_usage_mb: int,
        num_edges: int,
    ) -> Dict[str, int]:
        """
        Applies all configured constraints to initial replica proposals.

        This is the main entry point for the constraint manager. It orchestrates
        the application of various constraint phases to the raw proposals
        received from a scaling controller (e.g., PID controller).

        Order of Operations:
        1.  Log initial state and received proposals.
        2.  (Optionally) Apply Aggressive Memory Scale-Down: If current global
            memory usage exceeds the effective limit, replicas are reduced,
            prioritizing stages with higher replica counts.
        3.  Apply Proportional Allocation/Reduction for Scale-Up Requests:
            Global limits (max total replicas, available cores, memory budget)
            are checked against the *original combined scale-up intent*. If limits
            would be breached, the originally requested *increases* are scaled
            down proportionally. Results are capped by any adjustments from
            the aggressive memory scale-down.
        4.  Enforce Per-Stage Bounds: Each stage's replica count is clamped
            between its effective minimum (considering pipeline activity) and
            its configured maximum.
        5.  Apply Global Consistency (Wake-up Safety): Ensures that if any
            stage scales up from zero, other stages that could also start are
            not left at zero if their minimums allow.
        6.  Log a detailed summary of the decision-making process and final outcomes.

        Parameters
        ----------
        initial_proposals : Dict[str, StagePIDProposal]
            A dictionary mapping stage names to `StagePIDProposal` objects,
            each containing the current replica count, the PID-proposed replica
            count, and other metrics like min/max replicas and memory cost.
        global_in_flight : int
            The total number of tasks currently in flight across the entire pipeline.
            Used to determine effective minimum replicas (e.g., to keep stages
            alive if there's pending work).
        current_global_memory_usage_mb : int
            The current total memory usage by all replicas in the pipeline, in MB.
        num_edges : int
            The number of edges (queues) in the pipeline. Currently used for logging context.

        Returns
        -------
        Dict[str, int]
            A dictionary mapping stage names to their final target replica counts
            after all constraints have been applied.
        """
        _ = num_edges  # Mark as used if only for logging/context, otherwise remove if truly unused.
        logger.info(
            f"[ConstraintMgr] --- Applying Constraints START --- "
            f"GlobalInFlight={global_in_flight}, "
            f"CurrentGlobalMemMB={current_global_memory_usage_mb}, "
            f"NumEdges={num_edges}."
        )
        logger.debug("[ConstraintMgr] Initial Proposals:")
        for name, prop in initial_proposals.items():
            logger.debug(
                f"[ConstraintMgr]   Stage '{name}': Current={prop.current_replicas}, "
                f"PIDProposed={prop.proposed_replicas}, CostMB={prop.conservative_cost_estimate:.2f}, "
                f"MinCfg={prop.metrics.get('min_replicas', 'N/A')}, MaxCfg={prop.metrics.get('max_replicas', 'N/A')}"
            )

        # --- Phase 1: Initialize adjustments from PID proposals ---
        # These are the raw numbers from the PID controller before any constraints.
        intermediate_adjustments: Dict[str, int] = {
            name: prop.proposed_replicas for name, prop in initial_proposals.items()
        }
        logger.debug(f"[ConstraintMgr] Intermediate Adjustments (Phase 1 - From PID): {intermediate_adjustments}")

        # --- Phase 2: Aggressive Memory Scale-Down ---
        # This step modifies `intermediate_adjustments` in place if memory limits are breached.
        # try:
        #     intermediate_adjustments = self._apply_aggressive_memory_scale_down(
        #        intermediate_adjustments, initial_proposals,
        #        current_global_memory_usage_mb, global_in_flight_tasks
        #     )
        #     logger.debug(f"[ConstraintMgr] Intermediate Adjustments (Phase 2 - After Aggressive Mem Scale-Down):
        #     {intermediate_adjustments}")
        # except Exception as e_agg:
        #     logger.error(f"[ConstraintMgr] Error during aggressive memory scale-down: {e_agg}", exc_info=True)
        #     # Fallback: revert to current replicas if aggressive scaling fails critically
        #     intermediate_adjustments = {name: prop.current_replicas for name, prop in initial_proposals.items()}
        # logger.info("[ConstraintMgr] Phase 2: Aggressive Memory Scale-Down (Currently Bypassed/To-Implement).")

        # --- Phase 3: Apply Global Constraints & Proportional Allocation ---
        # This step calculates `tentative_adjustments` based on global limits,
        # using `intermediate_adjustments` as input (which might have been modified by aggressive scale-down).
        try:
            tentative_adjustments = self._apply_global_constraints_proportional(
                intermediate_adjustments,  # Input from previous phase
                initial_proposals,  # Original proposals for calculating upscale intent
                current_global_memory_usage_mb,
            )
            logger.debug(
                f"[ConstraintMgr] Tentative Adjustments (Phase 3 - After Proportional Allocation): "
                f"{tentative_adjustments}"
            )
        except Exception as e_prop:
            logger.error(f"[ConstraintMgr] Error during global proportional allocation: {e_prop}", exc_info=True)
            # Fallback: use results from aggressive scale-down (or PID if aggressive was skipped)
            tentative_adjustments = intermediate_adjustments

        # --- Phase 4: Enforce Per-Stage Min/Max Replica Bounds ---
        # This step iterates through `tentative_adjustments` and applies individual stage bounds.
        final_adjustments: Dict[str, int] = {}
        for stage_name, proposal_meta in initial_proposals.items():
            # Get the replica count for this stage after the proportional allocation phase.
            # Default to its current_replicas if somehow missing (should not happen).
            replicas_after_proportional = tentative_adjustments.get(stage_name, proposal_meta.current_replicas)
            try:
                bounded_replicas = self._enforce_replica_bounds(
                    stage_name, replicas_after_proportional, proposal_meta.metrics, global_in_flight
                )
                final_adjustments[stage_name] = bounded_replicas
            except Exception as e_bounds:
                logger.error(
                    f"[ConstraintMgr-{stage_name}] Error during per-stage bound enforcement: {e_bounds}", exc_info=True
                )
                # Fallback: use the stage's current replica count if bound enforcement fails.
                final_adjustments[stage_name] = proposal_meta.current_replicas
        logger.debug(f"[ConstraintMgr] Final Adjustments (Phase 4 - After Per-Stage Bounds): {final_adjustments}")

        # --- Phase 5: Apply Global Consistency (e.g., Wake-up Safety) ---
        # This step modifies `final_adjustments` in place to ensure pipeline consistency.
        try:
            self._apply_global_consistency(final_adjustments, initial_proposals)
            logger.debug(f"[ConstraintMgr] Final Adjustments (Phase 5 - After Global Consistency): {final_adjustments}")
        except Exception as e_gc:
            logger.error(f"[ConstraintMgr] Error during global consistency application: {e_gc}", exc_info=True)
            # No specific fallback here; modifications by _apply_global_consistency might be partial.

        # --- Summary and Final Logging ---
        logger.info("--- Constraint Manager: Decision Summary ---")

        # Detailed log for each stage's journey
        for stage_name in initial_proposals.keys():
            prop = initial_proposals[stage_name]
            pid_proposed = prop.proposed_replicas
            after_aggressive = intermediate_adjustments.get(
                stage_name, pid_proposed
            )  # Value after aggressive (or pid if skipped)
            after_proportional = tentative_adjustments.get(stage_name, after_aggressive)  # Value after proportional
            final_decision = final_adjustments.get(stage_name, prop.current_replicas)  # Final value

            # Build a string describing the journey and reasons for change
            journey = f"Current={prop.current_replicas}, PID={pid_proposed}"
            if pid_proposed != after_aggressive:  # Assuming aggressive scale-down is phase 2
                journey += f" -> AggressiveSD={after_aggressive}"
            if after_aggressive != after_proportional:
                journey += f" -> Proportional={after_proportional}"
            # Bound enforcement and global consistency modify the 'final_decision'
            # So we compare after_proportional with final_decision
            if after_proportional != final_decision:
                # Determine if change was due to bounds or global consistency by checking logs or specific flags
                # if available
                # For now, a generic "FinalBounds/Consistency"
                journey += f" -> Bounds/Consistency={final_decision}"
            elif (
                final_decision == after_proportional
                and final_decision == pid_proposed
                and final_decision == prop.current_replicas
            ):
                # No change from start to end, but good to confirm
                pass  # No change, journey already reflects this if PID = current
            elif final_decision == after_proportional:  # No change in last steps
                pass

            # logger.info(
            #    f"[ConstraintMgr-Decision] Stage '{stage_name}': {journey} => FINAL={final_decision}. "
            #    f"(MinCfg={prop.metrics.get('min_replicas', 'N/A')},
            #       MaxCfg={prop.metrics.get('max_replicas', 'N/A')}, "
            #    f"EffectiveMinActual={self._get_effective_min_replicas(stage_name, prop.metrics, global_in_flight)})"
            # )

        # Log global summaries
        final_total_replicas = sum(final_adjustments.values())
        core_limit_display = (
            f"{self.available_cores:.1f}" if self.available_cores is not None else "N/A (Detection Failed or Zero)"
        )
        core_based_limit_display = (
            str(self.core_based_replica_limit)
            if self.core_based_replica_limit is not None
            else "N/A (No Core-Based Limit)"
        )

        # Calculate projected memory usage based on final decisions
        projected_final_memory_mb = sum(
            final_adjustments.get(name, 0) * initial_proposals[name].conservative_cost_estimate
            for name in final_adjustments
        )

        # --- Summary and Final Logging (Pretty Printed) ---
        logger.info("[ConstraintMgr] --- Final State & Limit Checks ---")

        # Overall Replica Summary
        logger.info(f"[ConstraintMgr] Final Total Replicas         : {final_total_replicas}")

        # Configured Global Limits
        logger.info(f"[ConstraintMgr]   Configured MaxTotalReplicas: {self.max_replicas}")
        logger.info(
            f"[ConstraintMgr]   Configured MemThresholdMB  : {self.memory_threshold_mb:.1f}"
        )  # Assuming you have self.memory_threshold_mb
        logger.info(f"[ConstraintMgr]   Configured MemSafetyBuffer : {self.memory_safety_buffer_fraction*100:.1f}%")
        logger.info(f"[ConstraintMgr]   Effective MemLimitMB       : {self.effective_memory_limit_mb:.1f}")

        # System-Derived Constraints
        core_limit_display = (
            f"{self.available_cores:.1f}" if self.available_cores is not None else "N/A (Detection Failed or Zero)"
        )
        core_based_limit_display = (
            str(self.core_based_replica_limit)
            if self.core_based_replica_limit is not None
            else "N/A (Not Applicable or Undefined)"
        )
        logger.info(f"[ConstraintMgr]   System AvailableEffCores : {core_limit_display}")
        logger.info(f"[ConstraintMgr]   System CoreBasedRepLimit : {core_based_limit_display}")

        # Memory Usage Summary
        logger.info("[ConstraintMgr] Memory Usage (MB):")
        logger.info(f"[ConstraintMgr]   Current Global           : {current_global_memory_usage_mb:.1f}")
        logger.info(f"[ConstraintMgr]   Projected Final          : {projected_final_memory_mb:.1f}")
        logger.info(f"[ConstraintMgr]   Target                   : < {self.effective_memory_limit_mb:.1f}")

        # Limit Breach Warnings
        breached_limits_count = 0
        if final_total_replicas > self.max_replicas:
            logger.warning(
                f"[ConstraintMgr]   WARNING: MaxTotalReplicas BREACHED! "
                f"(Final: {final_total_replicas} > Limit: {self.max_replicas})"
            )
            breached_limits_count += 1
        if self.core_based_replica_limit is not None and final_total_replicas > self.core_based_replica_limit:
            logger.warning(
                f"[ConstraintMgr]   WARNING: CoreBasedReplicaLimit BREACHED! "
                f"(Final: {final_total_replicas} > Limit: {self.core_based_replica_limit})"
            )
            breached_limits_count += 1
        if projected_final_memory_mb > self.effective_memory_limit_mb:
            # Add a small tolerance for floating point comparisons if necessary
            tolerance = 0.01  # e.g., 0.01 MB
            if projected_final_memory_mb > (self.effective_memory_limit_mb + tolerance):
                logger.warning(
                    f"[ConstraintMgr]   WARNING: EffectiveMemLimit BREACHED! "
                    f"(Projected: {projected_final_memory_mb:.1f}MB > Limit: {self.effective_memory_limit_mb:.1f}MB)"
                )
                breached_limits_count += 1

        if breached_limits_count == 0:
            logger.info("[ConstraintMgr] All global limits appear to be respected by final decisions.")

        # Final decisions per stage (can be lengthy, so consider if always needed at INFO or DEBUG)
        # Using a more structured format for the final decisions dictionary if it's large
        if len(final_adjustments) > 5:  # Example threshold
            logger.info("[ConstraintMgr] Final Replica Decisions (Per Stage):")
            for stage_name, count in sorted(final_adjustments.items()):  # Sort for consistent output
                logger.info(f"[ConstraintMgr]     '{stage_name}': {count}")
        else:
            logger.info(f"[ConstraintMgr] Final Replica Decisions (Per Stage): {final_adjustments}")

        logger.info("[ConstraintMgr] --- Applying Constraints END ---")
        return final_adjustments
