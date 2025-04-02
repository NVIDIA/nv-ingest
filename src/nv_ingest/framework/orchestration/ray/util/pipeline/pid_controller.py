# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass

import numpy as np
from collections import deque
from typing import Dict, Any, Deque, List, Tuple

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
        penalty_factor: float = 0.5,
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
            # Example: If target is 5, penalty shouldn't effectively make it < -2.5
            max_penalty_contribution = target_queue_depth + max(1, 0.5 * abs(target_queue_depth))
            penalty = min(self.penalty_factor * self.idle_cycles[stage], max_penalty_contribution)

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
    adjusts them based on global limits (max replicas, memory budget with
    safety buffer), and ensures pipeline consistency (zero-replica safety).
    It allocates limited resources proportionally if multiple stages request
    scale-ups simultaneously.

    If current global memory usage exceeds the effective limit, it aggressively
    scales down stages starting with the highest replica counts.

    """

    def __init__(
        self,
        max_replicas: int,
        memory_threshold: int,
        estimated_edge_cost_mb: int,  # Note: Still unused in current logic
        memory_safety_buffer_fraction: float,
    ):
        """
        Initializes the Resource Constraint Manager.

        Parameters
        ----------
        max_replicas : int
            Absolute maximum number of replicas allowed across *all* stages combined.
        memory_threshold : int
            The total system memory usage (e.g., from psutil.virtual_memory().used,
            in MB) that the pipeline should aim to stay under.
        estimated_edge_cost_mb : int
            An estimate of the memory footprint (in MB) for each inter-stage
            queue (edge) actor. Used for rough pipeline memory estimation.
            (Currently unused in constraint logic).
        memory_safety_buffer_fraction : float
            A fraction (0.0 to <1.0) of `memory_threshold` to reserve as headroom.
            The manager aims to keep the *projected* memory usage below
            `memory_threshold * (1 - memory_safety_buffer_fraction)`. This buffer
            accounts for estimation errors and initialization overhead of new actors.
        """
        if not (0.0 <= memory_safety_buffer_fraction < 1.0):
            raise ValueError("memory_safety_buffer_fraction must be between 0.0 and 1.0")

        self.max_replicas = max_replicas
        self.memory_threshold = memory_threshold
        self.estimated_edge_cost_mb = estimated_edge_cost_mb
        self.memory_safety_buffer_fraction = memory_safety_buffer_fraction
        # Calculate the actual memory target considering the buffer
        self.effective_memory_limit = self.memory_threshold * (1.0 - self.memory_safety_buffer_fraction)
        logger.info(
            f"[ConstraintMgr] Initialized. MaxReplicas={max_replicas}, MemThreshold={memory_threshold}MB, "
            f"EffectiveLimit={self.effective_memory_limit:.1f}MB "
            # f"EdgeCost={estimated_edge_cost_mb}MB" # Commented out as unused
        )

    # --- Private Methods ---

    def _get_effective_min_replicas(self, stage_name: str, metrics: Dict[str, Any], pipeline_in_flight: int) -> int:
        """Helper to calculate the effective minimum replicas for a stage."""
        min_replicas_metric = metrics.get("min_replicas", 0)
        # --- FIXED RULE ---
        # If the pipeline is active globally, enforce a minimum of 1 replica,
        # unless min_replicas dictates higher.
        if pipeline_in_flight > 0:
            return max(1, min_replicas_metric)
        else:  # Pipeline is globally idle
            # Allow scaling down to zero ONLY if the pipeline is idle AND min_replicas allows it.
            return min_replicas_metric  # This will be 0 if min_replicas was 0 or not set

    def _apply_aggressive_memory_scale_down(
        self,
        current_proposals: Dict[str, int],  # Current replica counts (potentially modified from initial PID)
        initial_proposals_meta: Dict[str, StagePIDProposal],  # Original proposals for metadata
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
        if current_global_memory_usage <= self.effective_memory_limit:
            # Memory is fine, no aggressive action needed.
            return current_proposals

        memory_overrun = current_global_memory_usage - self.effective_memory_limit
        logger.warning(
            f"[ConstraintMgr] Aggressive Scale-Down Triggered: "
            f"Current Mem ({current_global_memory_usage:.1f}MB) > Effective Limit"
            f" ({self.effective_memory_limit:.1f}MB). "
            f"Need to reduce by {memory_overrun:.1f}MB."
        )

        adjusted_proposals = current_proposals.copy()  # Work on a copy

        # --- Identify candidates for scale-down ---
        candidates = []
        for name, current_replicas in adjusted_proposals.items():
            proposal_meta = initial_proposals_meta.get(name)
            if not proposal_meta:
                logger.error(f"[ConstraintMgr] Missing metadata for stage {name} during aggressive scale-down.")
                continue  # Should not happen if inputs are consistent

            effective_min = self._get_effective_min_replicas(name, proposal_meta.metrics, pipeline_in_flight_global)
            cost_estimate = proposal_meta.conservative_cost_estimate

            if current_replicas > effective_min:
                # Sort by: 1. Highest replica count (desc), 2. Highest memory cost (desc, to save more per step)
                candidates.append(
                    {
                        "name": name,
                        "replicas": current_replicas,
                        "cost": cost_estimate if cost_estimate > 0 else 1e-6,  # Avoid zero cost for sorting/division
                        "effective_min": effective_min,
                    }
                )

        # Sort candidates: primarily by replica count desc, secondarily by cost desc
        candidates.sort(key=lambda x: (x["replicas"], x["cost"]), reverse=True)

        if not candidates:
            logger.warning("[ConstraintMgr] Aggressive Scale-Down: No eligible stages found to reduce replicas.")
            return adjusted_proposals  # Return the proposals as they are

        # --- Iteratively reduce replicas ---
        memory_reduced = 0.0
        stages_reduced = []

        while memory_overrun > 0 and candidates:
            # Take the top candidate (highest replicas/cost)
            target_stage = candidates[0]
            name = target_stage["name"]

            # Reduce by one replica
            new_replica_count = target_stage["replicas"] - 1
            mem_saved_this_step = target_stage["cost"]

            logger.debug(
                f"[ConstraintMgr-{name}] Aggressive Scale-Down: Reducing replica from {target_stage['replicas']} ->"
                f" {new_replica_count} (saves ~{mem_saved_this_step:.2f}MB)"
            )
            adjusted_proposals[name] = new_replica_count
            memory_overrun -= mem_saved_this_step
            memory_reduced += mem_saved_this_step
            stages_reduced.append(name)  # Track which stages were hit

            # Update the candidate's state
            target_stage["replicas"] = new_replica_count

            # Remove candidate if it reached its effective minimum
            if new_replica_count <= target_stage["effective_min"]:
                candidates.pop(0)
            else:
                # Re-sort required if replica counts change significantly relative to others
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
        # Takes the proposals possibly already modified by aggressive scale-down
        tentative_proposals: Dict[str, int],
        # Still needs original proposal metadata for costs, current replicas etc.
        initial_proposals_meta: Dict[str, StagePIDProposal],
        current_global_memory_usage: int,
    ) -> Dict[str, int]:
        """Applies global replica/memory limits proportionally to scale-up requests."""
        # --- Base calculations on tentative_proposals vs *original* current replicas ---
        upscale_requests: List[Tuple[str, int, float]] = []  # name, requested_increase, cost_per_replica
        downscale_or_no_change: List[Tuple[str, int]] = []  # name, final_replicas
        total_requested_increase_replicas = 0
        total_projected_mem_increase = 0.0
        current_total_replicas_effective = 0  # Sum of replicas *before* this proportional step

        # --- Separate proposals based on ORIGINAL intent vs current state ---
        for name, proposal_meta in initial_proposals_meta.items():
            current_replicas_for_stage = tentative_proposals[name]
            current_total_replicas_effective += current_replicas_for_stage
            initial_delta = proposal_meta.proposed_replicas - proposal_meta.current_replicas

            if initial_delta > 0:
                cost_estimate = proposal_meta.conservative_cost_estimate
                upscale_requests.append((name, initial_delta, cost_estimate))
                total_requested_increase_replicas += initial_delta
                total_projected_mem_increase += initial_delta * cost_estimate
            else:
                downscale_or_no_change.append((name, current_replicas_for_stage))

        logger.debug(
            f"[ConstraintMgr-Proportional] Initial upscale intent: {len(upscale_requests)} stages, "
            f"ΔR={total_requested_increase_replicas}, ΔMem={total_projected_mem_increase:.2f}MB. "
            f"Current effective replicas before proportional step: {current_total_replicas_effective}"
        )

        # --- Check global constraints for the original *requested* increase ---
        scale_up_allowed = True
        reduction_factor = 1.0

        if total_requested_increase_replicas <= 0:
            scale_up_allowed = False
            logger.debug("[ConstraintMgr-Proportional] No scale-up originally requested.")
        else:
            # Check Max Replicas
            baseline_total_replicas = sum(prop.current_replicas for prop in initial_proposals_meta.values())
            projected_total_replicas = baseline_total_replicas + total_requested_increase_replicas
            if projected_total_replicas > self.max_replicas:
                allowed_increase = max(0, self.max_replicas - baseline_total_replicas)
                factor = (
                    allowed_increase / total_requested_increase_replicas
                    if total_requested_increase_replicas > 0
                    else 0.0
                )
                reduction_factor = min(reduction_factor, factor)
                logger.warning(
                    f"[ConstraintMgr-Proportional] Max replicas constraint potentially hit by initial request "
                    f"({projected_total_replicas}/{self.max_replicas}). Limiting factor: {factor:.3f}"
                )

            # Check Effective Memory Limit
            projected_total_global_memory = current_global_memory_usage + total_projected_mem_increase
            if projected_total_global_memory > self.effective_memory_limit:
                allowed_mem_increase = max(0.0, self.effective_memory_limit - current_global_memory_usage)
                factor = (
                    allowed_mem_increase / total_projected_mem_increase if total_projected_mem_increase > 0 else 0.0
                )
                reduction_factor = min(reduction_factor, factor)
                logger.warning(
                    f"[ConstraintMgr-Proportional] Effective memory limit potentially hit by initial request "
                    f"({projected_total_global_memory:.1f}/{self.effective_memory_limit:.1f}MB). "
                    f"Limiting factor: {factor:.3f}"
                )

            scale_up_allowed = reduction_factor > 0.001

        # --- Apply reduction factor to the *original requested increase* ---
        final_proposals = tentative_proposals.copy()

        if not scale_up_allowed or reduction_factor <= 0.001:
            logger.debug("[ConstraintMgr-Proportional] Blocking/zeroing scale-up potential based on constraints.")
            pass  # Values already correct due to initialization from tentative_proposals
        elif reduction_factor < 1.0:
            logger.debug(
                f"[ConstraintMgr-Proportional] Reducing scale-up potential by factor {reduction_factor:.3f}"
                f" relative to original request."
            )
            for name, requested_increase, _ in upscale_requests:
                proposal_meta = initial_proposals_meta[name]
                allowed_increase = math.floor(requested_increase * reduction_factor)
                target_replicas = proposal_meta.current_replicas + allowed_increase
                current_val_after_aggressive = final_proposals[name]  # Value possibly set by aggressive step

                # Apply the proportionally calculated target
                final_replicas_for_stage = target_replicas

                if final_replicas_for_stage != current_val_after_aggressive:
                    logger.debug(
                        f"[ConstraintMgr-{name}] Proportional Adjustment: Original Req={requested_increase}, "
                        f"Allowed Inc={allowed_increase}, Target={final_replicas_for_stage}. "
                        f"(Value before this step was {current_val_after_aggressive})"
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
            min_replicas_metric = metrics.get("min_replicas", 0)  # For logging
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
        self, initial_proposals: Dict[str, StagePIDProposal], current_global_memory_usage: int, num_edges: int
    ) -> Dict[str, int]:
        """
        Applies all constraints to initial proposals and returns final adjustments.

        Order of Operations:
        1. Calculate global pipeline activity.
        2. Apply Aggressive Memory Scale-Down if over memory limit.
        3. Apply Proportional Allocation/Reduction for scale-up requests based on limits.
        4. Enforce Per-Stage Bounds (min/max replicas, zero-replica safety).
        5. Apply Global Consistency (wake-up safety).

        Returns:
            Dict[str, int]: Final target replica counts per stage.
        """

        _ = num_edges  # Unused for now
        logger.debug("--- Constraint Manager: Applying Constraints ---")

        # --- Phase 1: Calculate Global State ---
        pipeline_in_flight_global = sum(prop.metrics.get("in_flight", 0) for prop in initial_proposals.values())
        logger.debug(f"[ConstraintMgr] Pipeline In-Flight: {pipeline_in_flight_global}")

        # Intermediate proposals start as the initially proposed values
        intermediate_adjustments: Dict[str, int] = {
            name: prop.proposed_replicas for name, prop in initial_proposals.items()
        }

        # --- Phase 2: Aggressive Memory Scale-Down ---
        try:
            intermediate_adjustments = self._apply_aggressive_memory_scale_down(
                intermediate_adjustments, initial_proposals, current_global_memory_usage, pipeline_in_flight_global
            )
        except Exception as e:
            logger.error(f"[ConstraintMgr] Error during aggressive memory scale-down: {e}", exc_info=True)
            intermediate_adjustments = {
                name: prop.current_replicas for name, prop in initial_proposals.items()
            }  # Fallback

        # --- Phase 3: Apply Global Constraints Proportional Allocation ---
        try:
            tentative_adjustments = self._apply_global_constraints_proportional(
                intermediate_adjustments,
                initial_proposals,
                current_global_memory_usage,
            )
        except Exception as e:
            logger.error(f"[ConstraintMgr] Error during global constraint proportional allocation: {e}", exc_info=True)
            tentative_adjustments = intermediate_adjustments  # Fallback to output of previous step

        # --- Phase 4: Enforce Per-Stage Bounds ---
        final_adjustments: Dict[str, int] = {}
        for name, proposal in initial_proposals.items():
            replicas_after_proportional = tentative_adjustments.get(name, proposal.current_replicas)
            try:
                final_replicas = self._enforce_replica_bounds(
                    name, replicas_after_proportional, proposal.metrics, pipeline_in_flight_global
                )
                final_adjustments[name] = final_replicas
            except Exception as e:
                logger.error(f"[ConstraintMgr-{name}] Error during bound enforcement: {e}", exc_info=True)
                final_adjustments[name] = proposal.current_replicas  # Fallback

        # --- Phase 5: Apply Global Consistency (Zero-Replica Safety / Wake-up) ---
        try:
            self._apply_global_consistency(final_adjustments, initial_proposals)
        except Exception as e:
            logger.error(f"[ConstraintMgr] Error during global consistency check: {e}", exc_info=True)

        logger.debug("--- Constraint Manager: Constraints Applied ---")
        final_decision_log = {name: count for name, count in final_adjustments.items()}
        logger.info(f"[ConstraintMgr] Final Replica Decisions: {final_decision_log}")

        return final_adjustments
