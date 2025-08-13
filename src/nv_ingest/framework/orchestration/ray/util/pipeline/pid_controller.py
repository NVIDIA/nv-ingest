# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from dataclasses import dataclass

from typing import Dict, Any, List, Tuple, Optional

from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_STAGE_COST_MB = 5_000.0  # Fallback memory cost


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
        target_queue_depth: int = 0,
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
        target_queue_depth : int, optional
            Default target queue depth for stages if not specified in metrics,
            by default 0. The PID loop tries to drive the queue depth towards
            this value.
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
        self.integral_error: Dict[str, float] = {}
        self.prev_error: Dict[str, float] = {}
        self.idle_cycles: Dict[str, int] = {}

        # Per-Stage Config
        self.penalty_factor = penalty_factor

    # --- Private Methods ---

    def _initialize_stage_state(self, stage: str) -> None:
        """Initializes controller state variables for a newly seen stage."""
        if stage not in self.integral_error:
            logger.debug(f"[PID-{stage}] Initializing state.")
            self.integral_error[stage] = 0.0
            self.prev_error[stage] = 0.0
            self.idle_cycles[stage] = 0

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
            per stage: 'replicas', 'queue_depth', 'ema_memory_per_replica'.
            Optional: 'target_queue_depth', 'processing', 'min_replicas', 'max_replicas'.

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
            # The conservative cost is now the EMA memory passed in from the stats collector.
            # Fallback to a default if not present.
            conservative_cost = metrics.get("ema_memory_per_replica", DEFAULT_STAGE_COST_MB)

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
            penalty = min(8, self.penalty_factor * (self.idle_cycles[stage] ** 2.0))

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
        self.memory_safety_buffer_fraction = memory_safety_buffer_fraction  # Unused
        self.effective_memory_limit_mb = self.memory_threshold_mb

        core_detector = SystemResourceProbe()  # Instantiate the detector
        self.available_cores: Optional[float] = core_detector.get_effective_cores()
        self.core_detection_details: Dict[str, Any] = core_detector.get_details()

        # Determine a practical replica limit based on cores (optional, but often useful)
        self.core_based_replica_limit: Optional[int] = None
        if self.available_cores is not None and self.available_cores > 0:
            self.core_based_replica_limit = math.floor(self.available_cores)
        else:
            self.core_based_replica_limit = None  # Treat as unlimited if detection failed

        logger.debug(
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
        initial_proposals_meta: Dict[str, "StagePIDProposal"],  # Assuming StagePIDProposal type hint
        current_global_memory_usage: int,
        pipeline_in_flight_global: int,
    ) -> Dict[str, int]:
        """
        If current memory exceeds the effective limit, force scale-downs.

        Reduces replicas for all stages with > 1 replica
        by 25% (rounded down), ensuring they don't go below their effective minimum
        or 1 replica. This is done in a single pass.

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
        total_memory_reduced = 0.0
        stages_affected_details = {}  # To store details of changes

        # Iterate through all proposals to apply the 25% reduction if applicable
        for name, current_replicas in current_proposals.items():
            proposal_meta = initial_proposals_meta.get(name)
            if not proposal_meta:
                logger.error(f"[ConstraintMgr] Missing metadata for stage {name} during aggressive scale-down.")
                continue

            # Determine the effective minimum for this stage (ensuring at least 1)
            effective_min = self._get_effective_min_replicas(name, proposal_meta.metrics, pipeline_in_flight_global)

            # Cost per replica (assuming proposal_meta.conservative_cost_estimate is for ONE replica)
            # If it's for all current_replicas, you'd divide by current_replicas here.
            cost_per_replica = float(
                proposal_meta.conservative_cost_estimate
                if proposal_meta.conservative_cost_estimate and proposal_meta.conservative_cost_estimate > 0
                else 1e-6
            )

            if current_replicas > 1:  # Only consider stages with more than 1 replica
                # Calculate 25% reduction
                reduction_amount = math.floor(current_replicas * 0.25)

                # Ensure reduction_amount is at least 1 if current_replicas > 1 and 25% is < 1
                # (e.g., for 2 or 3 replicas, 25% is 0, but we want to reduce by 1 if possible)
                if reduction_amount == 0 and current_replicas > 1:
                    reduction_amount = 1

                if reduction_amount > 0:
                    proposed_new_replicas = current_replicas - reduction_amount

                    # Ensure new count doesn't go below the effective minimum (which is at least 1)
                    final_new_replicas = max(effective_min, proposed_new_replicas)

                    # Only apply if this actually results in a reduction
                    if final_new_replicas < current_replicas:
                        replicas_actually_reduced = current_replicas - final_new_replicas
                        memory_saved_for_stage = replicas_actually_reduced * cost_per_replica

                        logger.info(
                            f"[ConstraintMgr-{name}] Aggressive Scale-Down: Reducing from "
                            f"{current_replicas} -> {final_new_replicas} "
                            f"(by {replicas_actually_reduced} replicas, target 25% of "
                            f"{current_replicas} was {reduction_amount}). "
                            f"Est. memory saved: {memory_saved_for_stage:.2f}MB."
                        )
                        adjusted_proposals[name] = final_new_replicas
                        total_memory_reduced += memory_saved_for_stage
                        stages_affected_details[name] = {
                            "from": current_replicas,
                            "to": final_new_replicas,
                            "saved_mem": memory_saved_for_stage,
                        }
                    else:
                        logger.debug(
                            f"[ConstraintMgr-{name}] Aggressive Scale-Down: No reduction applied. "
                            f"Current: {current_replicas}, Target 25% reduction: {reduction_amount}, "
                            f"Proposed: {proposed_new_replicas}, Effective Min: {effective_min}."
                        )
                else:
                    logger.debug(
                        f"[ConstraintMgr-{name}] Aggressive Scale-Down: Calculated 25% reduction is 0 for "
                        f"{current_replicas} replicas. No change."
                    )
            else:
                logger.debug(
                    f"[ConstraintMgr-{name}] Aggressive Scale-Down: Stage has {current_replicas} "
                    f"replica(s), not eligible for 25% reduction."
                )

        # After applying reductions, check the new memory overrun
        # This is a projection based on our cost estimates.
        projected_new_global_memory_usage = current_global_memory_usage - total_memory_reduced
        new_memory_overrun = projected_new_global_memory_usage - self.effective_memory_limit_mb

        if not stages_affected_details:
            logger.warning("[ConstraintMgr] Aggressive Scale-Down: No stages were eligible or changed replicas.")
        elif new_memory_overrun > 0:
            logger.warning(
                f"[ConstraintMgr] Aggressive Scale-Down: Completed. Reduced total {total_memory_reduced:.1f}MB. "
                f"Stages affected: {len(stages_affected_details)}. "
                f"Projected memory still over limit by {new_memory_overrun:.1f}MB."
                # f"Details: {stages_affected_details}" # Potentially too verbose for warning
            )
        else:
            logger.info(
                f"[ConstraintMgr] Aggressive Scale-Down: Completed. Reduced total {total_memory_reduced:.1f}MB. "
                f"Stages affected: {len(stages_affected_details)}. "
                f"Projected memory now below limit (overrun {new_memory_overrun:.1f}MB)."
                # f"Details: {stages_affected_details}" # Potentially too verbose for info
            )
        if stages_affected_details:
            logger.debug(f"[ConstraintMgr] Aggressive Scale-Down Details: {stages_affected_details}")

        return adjusted_proposals

    def _apply_global_constraints_proportional(
        self,
        proposals_after_aggressive_sd: Dict[str, int],  # Values from PID or after AggressiveMemSD
        initial_proposals_meta: Dict[str, "StagePIDProposal"],  # Contains original .current_replicas
        current_global_memory_usage_mb: int,
        current_effective_mins: Dict[str, int],  # Effective minimum for each stage
        room_to_scale_up_to_global_caps: bool,
    ) -> Dict[str, int]:
        """
        Applies global replica, core, and memory limits to scale-up intentions.
        (Docstring from previous correct version summarizing the logic is fine)
        """
        final_proposals_this_step = {}

        if not room_to_scale_up_to_global_caps:
            logger.debug(
                "[ConstraintMgr-Proportional] Global scaling beyond effective minimums is RESTRICTED "
                "as SumOfEffectiveMins likely meets/exceeds a global Core/MaxReplica cap. "
                "Proposed increases from initial current values will be nullified."
            )
            for name, prop_meta in initial_proposals_meta.items():
                val_from_prior_phases = proposals_after_aggressive_sd.get(name, prop_meta.current_replicas)
                original_current_replicas = prop_meta.current_replicas

                if val_from_prior_phases > original_current_replicas:
                    final_proposals_this_step[name] = original_current_replicas
                    if val_from_prior_phases != original_current_replicas:
                        logger.debug(
                            f"[ConstraintMgr-{name}] Proportional: Scaling restricted. "
                            f"Nullified proposed increase from {original_current_replicas} to {val_from_prior_phases}. "
                            f"Setting to {original_current_replicas}."
                        )
                else:
                    final_proposals_this_step[name] = val_from_prior_phases
            return final_proposals_this_step

        # --- ELSE: room_to_scale_up_to_global_caps is TRUE ---
        # We can proportionally scale *increases above each stage's effective minimum*,
        # up to the global caps. The baseline sum for headroom is sum_of_effective_mins.

        # Stores (stage_name, proposed_increase_above_eff_min, cost_per_replica)
        upscale_deltas_above_eff_min: List[Tuple[str, int, float]] = []
        total_requested_increase_replicas_above_eff_mins = 0
        total_projected_mem_increase_for_deltas_mb = 0.0

        # Initialize final_proposals_this_step: each stage starts at its effective minimum,
        # but not less than what aggressive_sd might have proposed (e.g., if agg_sd proposed 0 and eff_min is 0).
        # And not more than what PID/agg_sd proposed if that was already below effective_min.
        # Essentially, the base is max(eff_min, value_from_agg_sd_if_value_is_for_scale_down_or_no_change).
        # More simply: start each stage at its effective_min. The "delta" is how much PID wants *above* that.

        sum_of_effective_mins_for_baseline = 0
        for name, prop_meta in initial_proposals_meta.items():
            eff_min_for_stage = current_effective_mins[name]
            final_proposals_this_step[name] = eff_min_for_stage  # Initialize with effective min
            sum_of_effective_mins_for_baseline += eff_min_for_stage

            # What did PID (after aggressive_sd) propose for this stage?
            pid_proposed_val = proposals_after_aggressive_sd.get(name, prop_meta.current_replicas)

            if pid_proposed_val > eff_min_for_stage:
                # This stage wants to scale up beyond its effective minimum.
                increase_delta = pid_proposed_val - eff_min_for_stage
                cost = prop_meta.conservative_cost_estimate
                upscale_deltas_above_eff_min.append((name, increase_delta, cost))
                total_requested_increase_replicas_above_eff_mins += increase_delta
                total_projected_mem_increase_for_deltas_mb += increase_delta * cost

        logger.debug(
            f"[ConstraintMgr-Proportional] Room to scale. BaselineSum "
            f"(SumOfEffMins)={sum_of_effective_mins_for_baseline}. "
            f"NumStagesRequestingUpscaleAboveEffMin={len(upscale_deltas_above_eff_min)}. "
            f"TotalReplicaIncreaseReqAboveEffMin={total_requested_increase_replicas_above_eff_mins}. "
            f"TotalMemIncreaseForTheseDeltas={total_projected_mem_increase_for_deltas_mb:.2f}MB."
        )

        reduction_factor = 1.0
        limiting_reasons = []

        if total_requested_increase_replicas_above_eff_mins <= 0:
            logger.debug(
                "[ConstraintMgr-Proportional] No upscale request beyond effective minimums. "
                "Proposals remain at effective minimums (or prior phase values if lower and valid)."
            )
            # final_proposals_this_step already contains effective minimums.
            # We need to ensure if PID proposed *lower* than effective_min (and eff_min was 0), that's respected.
            # This should be: max(pid_proposed_value, eff_min_for_stage) for each stage.
            for name_check in final_proposals_this_step.keys():
                pid_val = proposals_after_aggressive_sd.get(
                    name_check, initial_proposals_meta[name_check].current_replicas
                )
                eff_min_val = current_effective_mins[name_check]
                final_proposals_this_step[name_check] = (
                    max(pid_val, eff_min_val) if eff_min_val > 0 else pid_val
                )  # if eff_min is 0, allow PID to go to 0
            return final_proposals_this_step

        projected_total_replicas_with_deltas = (
            sum_of_effective_mins_for_baseline + total_requested_increase_replicas_above_eff_mins
        )

        # 1. Max Replicas Config
        if projected_total_replicas_with_deltas > self.max_replicas:
            # Headroom is how many *additional* replicas (beyond sum_of_eff_mins) we can add
            permissible_increase_headroom = max(0, self.max_replicas - sum_of_effective_mins_for_baseline)
            factor = permissible_increase_headroom / total_requested_increase_replicas_above_eff_mins
            reduction_factor = min(reduction_factor, factor)
            limiting_reasons.append(
                f"MaxReplicas (Limit={self.max_replicas}, HeadroomAboveEffMins={permissible_increase_headroom}, "
                f"Factor={factor:.3f})"
            )

        # 2. Core Based Replica Limit
        if (
            self.core_based_replica_limit is not None
            and projected_total_replicas_with_deltas > self.core_based_replica_limit
        ):
            permissible_increase_headroom = max(0, self.core_based_replica_limit - sum_of_effective_mins_for_baseline)
            factor = permissible_increase_headroom / total_requested_increase_replicas_above_eff_mins
            reduction_factor = min(reduction_factor, factor)
            limiting_reasons.append(
                f"CoreLimit (Limit={self.core_based_replica_limit}, "
                f"HeadroomAboveEffMins={permissible_increase_headroom}, Factor={factor:.3f})"
            )

        # 3. Memory Limit
        # Memory check is based on current_global_memory_usage_mb + memory_for_the_increase_deltas
        projected_total_global_memory_mb = current_global_memory_usage_mb + total_projected_mem_increase_for_deltas_mb
        if projected_total_global_memory_mb > self.effective_memory_limit_mb:
            # How much memory can we actually add without breaching the effective limit?
            permissible_mem_increase_mb = max(0.0, self.effective_memory_limit_mb - current_global_memory_usage_mb)
            factor_mem = (
                permissible_mem_increase_mb / total_projected_mem_increase_for_deltas_mb
                if total_projected_mem_increase_for_deltas_mb > 1e-9
                else 0.0
            )
            reduction_factor = min(reduction_factor, factor_mem)
            limiting_reasons.append(
                f"MemoryLimit (Factor={factor_mem:.3f}, AvailableMemForIncrease={permissible_mem_increase_mb:.1f}MB)"
            )

        # Apply reduction to the deltas
        if reduction_factor <= 0.001:  # Epsilon for float
            logger.debug(
                f"[ConstraintMgr-Proportional] Scale-up beyond effective minimums fully constrained by global limits. "
                f"Reasons: {'; '.join(limiting_reasons) if limiting_reasons else 'None'}. "
                f"Final ReductionFactor={reduction_factor:.3f}."
                " Stages will remain at their effective minimums (or prior phase values if lower and eff_min is 0)."
            )
            # final_proposals_this_step already contains effective minimums.
            # Need to ensure if PID wanted lower than eff_min (and eff_min was 0), that is respected.
            for name_final_check in final_proposals_this_step.keys():
                pid_val_final = proposals_after_aggressive_sd.get(
                    name_final_check, initial_proposals_meta[name_final_check].current_replicas
                )
                eff_min_final = current_effective_mins[name_final_check]
                # If effective min is 0, allow PID's value (which could be 0). Otherwise, floor is effective min.
                final_proposals_this_step[name_final_check] = (
                    pid_val_final if eff_min_final == 0 else max(pid_val_final, eff_min_final)
                )

        elif reduction_factor < 1.0:
            logger.debug(
                f"[ConstraintMgr-Proportional] Reducing requested scale-up (beyond effective_mins) by "
                f"factor {reduction_factor:.3f}. "
                f"Limiting Factors: {'; '.join(limiting_reasons)}."
            )
            for name, increase_delta_above_eff_min, _ in upscale_deltas_above_eff_min:
                allowed_increase = math.floor(increase_delta_above_eff_min * reduction_factor)
                # Add this allowed increase to the stage's effective minimum
                final_value_for_stage = current_effective_mins[name] + allowed_increase
                final_proposals_this_step[name] = final_value_for_stage
                if allowed_increase != increase_delta_above_eff_min:
                    logger.debug(
                        f"[ConstraintMgr-{name}] Proportional Adj: EffMin={current_effective_mins[name]}, "
                        f"ReqIncreaseAboveEffMin={increase_delta_above_eff_min}, AllowedIncrease={allowed_increase} "
                        f"-> FinalVal={final_value_for_stage}"
                    )
        else:  # reduction_factor is ~1.0, meaning full requested increase (above effective_mins) is allowed
            logger.debug(
                "[ConstraintMgr-Proportional] Full requested scale-up (beyond effective_mins) "
                "is permissible by global limits."
            )
            for name, increase_delta_above_eff_min, _ in upscale_deltas_above_eff_min:
                # The full PID-intended value (which came in as proposals_after_aggressive_sd) is applied.
                # Since final_proposals_this_step was initialized with effective_mins,
                # and increase_delta_above_eff_min = pid_proposed_val - eff_min_for_stage,
                # then eff_min_for_stage + increase_delta_above_eff_min = pid_proposed_val.
                pid_intended_val = proposals_after_aggressive_sd.get(
                    name, initial_proposals_meta[name].current_replicas
                )
                final_proposals_this_step[name] = (
                    pid_intended_val  # This effectively applies the PID's full wish for this stage
                )

        return final_proposals_this_step

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
    ) -> Dict:
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
                        logger.debug(
                            f"[ConstraintMgr-{name}] Forcing minimum {final_target} replica due to global wake-up."
                        )
                        final_adjustments[name] = final_target

        return final_adjustments

    def _log_final_constraint_summary(
        self,
        final_adjustments: Dict[str, int],
        initial_proposals: Dict[str, "StagePIDProposal"],  # Forward reference
        global_in_flight: int,
        current_global_memory_usage_mb: int,
        num_edges: int,
        sum_of_effective_mins: int,
        can_globally_scale_beyond_effective_mins: bool,
    ) -> None:
        """Logs a structured and readable summary of the final state and limit checks."""

        final_stage_replicas_total = sum(final_adjustments.values())
        projected_final_memory_mb = sum(
            final_adjustments.get(name, 0) * initial_proposals[name].conservative_cost_estimate
            for name in final_adjustments
        )
        num_queue_actors = num_edges
        total_ray_components_for_info = final_stage_replicas_total + num_queue_actors

        logger.debug("[ConstraintMgr] --- Final Decision & Constraint Summary ---")

        # --- I. Overall Pipeline State ---
        logger.debug(f"[ConstraintMgr]   Pipeline Activity: {global_in_flight} tasks in-flight.")
        logger.debug(f"[ConstraintMgr]   Effective Min Replicas (Sum): {sum_of_effective_mins}")
        logger.debug(
            f"[ConstraintMgr]     └─ Global Scaling Beyond Mins Permitted? {can_globally_scale_beyond_effective_mins}"
        )

        # --- II. Final Component Counts ---
        logger.debug(f"[ConstraintMgr]   Final Stage Replicas: {final_stage_replicas_total} (Target for caps)")
        logger.debug(f"[ConstraintMgr]   Queue/Edge Actors   : {num_queue_actors} (Informational)")
        logger.debug(f"[ConstraintMgr]   Total Ray Components: {total_ray_components_for_info} (Informational)")

        # --- III. Resource Limits & Projected Usage (for Stages) ---
        # Configured Limits
        max_r_cfg_str = str(self.max_replicas)
        core_based_limit_str = (
            str(self.core_based_replica_limit) if self.core_based_replica_limit is not None else "N/A"
        )
        eff_mem_limit_str = f"{self.effective_memory_limit_mb:.1f}MB"

        logger.debug("[ConstraintMgr]   Global Limits (Stages):")
        logger.debug(f"[ConstraintMgr]     ├─ MaxTotalReplicas  : {max_r_cfg_str}")
        logger.debug(
            f"[ConstraintMgr]     ├─ CoreBasedRepLimit : {core_based_limit_str} "
            f"(System EffCores: {self.available_cores if self.available_cores is not None else 'N/A'})"
        )
        logger.debug(f"[ConstraintMgr]     └─ EffectiveMemLimit : {eff_mem_limit_str} ")

        # Usage vs Limits
        logger.debug("[ConstraintMgr]   Projected Usage (Stages):")
        logger.debug(f"[ConstraintMgr]     ├─ Replicas          : {final_stage_replicas_total}")
        logger.debug(
            f"[ConstraintMgr]     └─ Memory            : {projected_final_memory_mb:.1f}MB "
            f"(Current: {current_global_memory_usage_mb:.1f}MB)"
        )

        # --- IV. Limit Adherence Analysis (for Stages) ---
        unexpected_breaches_details = []

        # 1. Max Stage Replicas
        status_max_r = "OK"
        if final_stage_replicas_total > self.max_replicas:
            if not (sum_of_effective_mins >= self.max_replicas and final_stage_replicas_total <= sum_of_effective_mins):
                status_max_r = f"BREACHED (Final={final_stage_replicas_total} > Limit={self.max_replicas})"
                unexpected_breaches_details.append(f"MaxReplicas: {status_max_r}")
            else:
                status_max_r = f"NOTE: Limit met/exceeded by SumOfMins ({sum_of_effective_mins})"

        # 2. Core-Based Stage Replica Limit
        status_core_r = "N/A"
        if self.core_based_replica_limit is not None:
            status_core_r = "OK"
            if final_stage_replicas_total > self.core_based_replica_limit:
                if not (
                    sum_of_effective_mins >= self.core_based_replica_limit
                    and final_stage_replicas_total <= sum_of_effective_mins
                ):
                    status_core_r = (
                        f"BREACHED (Final={final_stage_replicas_total} > Limit={self.core_based_replica_limit})"
                    )
                    unexpected_breaches_details.append(f"CoreBasedLimit: {status_core_r}")
                else:
                    status_core_r = f"NOTE: Limit met/exceeded by SumOfMins ({sum_of_effective_mins})"

        # 3. Memory Limit
        tolerance = 0.01  # MB
        status_mem = "OK"
        if projected_final_memory_mb > (self.effective_memory_limit_mb + tolerance):
            status_mem = (
                f"BREACHED (Projected={projected_final_memory_mb:.1f}MB > Limit={self.effective_memory_limit_mb:.1f}MB)"
            )
            unexpected_breaches_details.append(f"MemoryLimit: {status_mem}")

        logger.debug("[ConstraintMgr]   Limit Adherence (Stages):")
        logger.debug(f"[ConstraintMgr]     ├─ MaxTotalReplicas  : {status_max_r}")
        logger.debug(f"[ConstraintMgr]     ├─ CoreBasedRepLimit : {status_core_r}")
        logger.debug(f"[ConstraintMgr]     └─ EffectiveMemLimit : {status_mem}")

        if unexpected_breaches_details:
            logger.debug(f"[ConstraintMgr]   └─ UNEXPECTED BREACHES: {'; '.join(unexpected_breaches_details)}")
        else:
            logger.debug("[ConstraintMgr]   └─ All hard caps (beyond tolerated minimums/wake-up) appear respected.")

        # --- V. Final Decisions Per Stage ---
        logger.debug("[ConstraintMgr]   Final Decisions (Per Stage):")
        if not final_adjustments:
            logger.debug("[ConstraintMgr]     └─ No stages to adjust.")
        else:
            # Determine max stage name length for alignment
            max_name_len = 0
            if final_adjustments:  # Check if not empty
                max_name_len = max(len(name) for name in final_adjustments.keys())

            for stage_name, count in sorted(final_adjustments.items()):
                orig_prop = initial_proposals.get(stage_name)
                pid_proposed_str = f"(PID: {orig_prop.proposed_replicas if orig_prop else 'N/A'})"
                current_str = f"(Current: {orig_prop.current_replicas if orig_prop else 'N/A'})"
                min_replicas = self._get_effective_min_replicas(stage_name, orig_prop.metrics, global_in_flight)
                eff_min_str = f"(EffMin: {min_replicas if orig_prop else 'N/A'})"

                # Basic alignment, can be improved with more sophisticated padding
                logger.debug(
                    f"[ConstraintMgr]     └─ {stage_name:<{max_name_len}} : "
                    f"{count:<3} {pid_proposed_str} {current_str} {eff_min_str}"
                )

        logger.debug("[ConstraintMgr] --- Constraint Summary END ---")

    # --- Public Method ---

    def apply_constraints(
        self,
        initial_proposals: Dict[str, "StagePIDProposal"],
        global_in_flight: int,  # Renamed from global_in_flight
        current_global_memory_usage_mb: int,
        num_edges: int,
    ) -> Dict[str, int]:
        """
        Applies all configured constraints to initial replica proposals.
        (Docstring from previous version is fine)
        """
        logger.debug(
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
        intermediate_adjustments: Dict[str, int] = {
            name: prop.proposed_replicas for name, prop in initial_proposals.items()
        }
        logger.debug(f"[ConstraintMgr] Intermediate Adjustments (Phase 1 - From PID): {intermediate_adjustments}")

        # --- Phase 2: Aggressive Memory Scale-Down (Optional) ---
        try:
            intermediate_adjustments = self._apply_aggressive_memory_scale_down(
                intermediate_adjustments, initial_proposals, current_global_memory_usage_mb, global_in_flight
            )
            logger.debug(
                "[ConstraintMgr] Intermediate Adjustments (Phase 2 - After Aggressive Mem Scale-Down): "
                f"{intermediate_adjustments}"
            )
        except Exception as e_agg:
            logger.error(f"[ConstraintMgr] Error during aggressive memory scale-down: {e_agg}", exc_info=True)
            intermediate_adjustments = {name: prop.current_replicas for name, prop in initial_proposals.items()}

        # --- Calculate Current Effective Minimums and Their Sum ---
        current_effective_mins: Dict[str, int] = {}
        sum_of_effective_mins = 0
        for name, prop in initial_proposals.items():
            eff_min = self._get_effective_min_replicas(name, prop.metrics, global_in_flight)
            current_effective_mins[name] = eff_min
            sum_of_effective_mins += eff_min

        logger.debug(
            f"[ConstraintMgr] Calculated Effective Minimums: TotalSum={sum_of_effective_mins}. "
            # f"IndividualMins: {current_effective_mins}" # Can be verbose
        )

        # --- Determine if Baseline (Sum of Mins) Breaches Global Caps ---
        # This logic determines if we are *allowed* to scale any stage *beyond its own effective minimum*
        # if doing so would contribute to breaching a global cap that's *already threatened by the sum of minimums*.
        can_globally_scale_beyond_effective_mins_due_to_cores = True
        if self.core_based_replica_limit is not None and sum_of_effective_mins >= self.core_based_replica_limit:
            can_globally_scale_beyond_effective_mins_due_to_cores = False

        can_globally_scale_beyond_effective_mins_due_to_max_r = True
        if sum_of_effective_mins >= self.max_replicas:
            can_globally_scale_beyond_effective_mins_due_to_max_r = False

        # Combined gatekeeper for proportional scaling logic
        # If either cores or max_replicas cap is hit by sum of mins, we can't scale up further.
        # (Memory is handled slightly differently in proportional scaler - it looks at available headroom for increase)
        can_globally_scale_up_stages = (
            can_globally_scale_beyond_effective_mins_due_to_cores
            and can_globally_scale_beyond_effective_mins_due_to_max_r
        )

        # --- Phase 3: Apply Global Constraints & Proportional Allocation ---
        try:
            tentative_adjustments_from_prop = self._apply_global_constraints_proportional(
                intermediate_adjustments,
                initial_proposals,
                current_global_memory_usage_mb,
                current_effective_mins,
                can_globally_scale_up_stages,  # Use the combined flag
            )
            logger.debug(
                f"[ConstraintMgr] Tentative Adjustments (Phase 3 - After Proportional Allocation): "
                f"{tentative_adjustments_from_prop}"
            )
        except Exception as e_prop:
            logger.error(f"[ConstraintMgr] Error during global proportional allocation: {e_prop}", exc_info=True)
            tentative_adjustments_from_prop = {}
            for name, count in intermediate_adjustments.items():  # Fallback logic
                tentative_adjustments_from_prop[name] = max(count, current_effective_mins.get(name, 0))

        # --- Phase 4: Enforce Per-Stage Min/Max Replica Bounds ---
        final_adjustments: Dict[str, int] = {}
        for stage_name, proposal_meta in initial_proposals.items():
            replicas_after_proportional = tentative_adjustments_from_prop.get(
                stage_name, proposal_meta.current_replicas
            )
            try:
                bounded_replicas = self._enforce_replica_bounds(
                    stage_name, replicas_after_proportional, proposal_meta.metrics, global_in_flight
                )
                final_adjustments[stage_name] = bounded_replicas
            except Exception as e_bounds:
                logger.error(
                    f"[ConstraintMgr-{stage_name}] Error during per-stage bound enforcement: {e_bounds}", exc_info=True
                )
                final_adjustments[stage_name] = max(
                    proposal_meta.current_replicas, current_effective_mins.get(stage_name, 0)
                )
        logger.debug(f"[ConstraintMgr] Final Adjustments (Phase 4 - After Per-Stage Bounds): {final_adjustments}")

        # --- Phase 5: Apply Global Consistency (e.g., Wake-up Safety) ---
        try:
            final_adjustments = self._apply_global_consistency(final_adjustments, initial_proposals)
            logger.debug(f"[ConstraintMgr] Final Adjustments (Phase 5 - After Global Consistency): {final_adjustments}")
        except Exception as e_gc:
            logger.error(f"[ConstraintMgr] Error during global consistency application: {e_gc}", exc_info=True)

        # --- Log Final Summary ---
        self._log_final_constraint_summary(
            final_adjustments,
            initial_proposals,
            global_in_flight,
            current_global_memory_usage_mb,
            num_edges,
            sum_of_effective_mins,  # Pass this calculated value
            can_globally_scale_up_stages,  # Pass this for context in logging
        )

        logger.debug("[ConstraintMgr] --- Applying Constraints END ---")
        return final_adjustments
