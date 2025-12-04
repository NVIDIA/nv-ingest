# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Runtime replica resolution for static scaling mode.

This module provides functionality to resolve replica counts for stages using
non-static strategies when dynamic scaling is disabled, ensuring total memory
consumption stays within the static_memory_threshold.
"""

import logging
from typing import List
from copy import deepcopy

from nv_ingest.pipeline.pipeline_schema import (
    PipelineConfigSchema,
    StageConfig,
    ReplicaCalculationStrategy,
    ReplicaStrategyConfig,
)
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)


def resolve_static_replicas(pipeline_config: PipelineConfigSchema) -> PipelineConfigSchema:
    """
    Resolve static replica counts for all stages when dynamic scaling is disabled.

    This function calculates the static replica counts for stages using non-static
    strategies, ensuring the total memory consumption stays within the configured
    static_memory_threshold. If the total exceeds the threshold, all non-static
    stages are scaled down proportionally (minimum 1 replica each).

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        The pipeline configuration with potentially unresolved replica strategies.

    Returns
    -------
    PipelineConfigSchema
        A new pipeline configuration with all static replica counts resolved.
    """
    # Only resolve if dynamic scaling is disabled
    if not pipeline_config.pipeline.disable_dynamic_scaling:
        logger.debug("Dynamic scaling enabled, skipping static replica resolution")
        return pipeline_config

    logger.info("Resolving static replica counts for disabled dynamic scaling mode")

    # Create a deep copy to avoid modifying the original config
    resolved_config = deepcopy(pipeline_config)

    # Get system resource information
    system_probe = SystemResourceProbe()
    total_memory_mb = system_probe.total_memory_mb
    available_memory_mb = int(total_memory_mb * resolved_config.pipeline.static_memory_threshold)

    logger.info(
        f"System memory: {total_memory_mb}MB, available for static replicas: {available_memory_mb}MB "
        f"(threshold: {resolved_config.pipeline.static_memory_threshold:.1%})"
    )

    # Find stages with non-static strategies and calculate their baseline replica counts
    non_static_stages = []
    total_memory_demand_mb = 0

    for stage in resolved_config.stages:
        if stage.replicas and stage.replicas.static_replicas:
            if isinstance(stage.replicas.static_replicas, ReplicaStrategyConfig):
                strategy_config = stage.replicas.static_replicas
                baseline_replicas = _calculate_baseline_static_replicas(
                    stage, strategy_config, system_probe, resolved_config.pipeline.static_memory_threshold
                )

                memory_per_replica_mb = strategy_config.memory_per_replica_mb or 0
                stage_memory_demand = baseline_replicas * memory_per_replica_mb

                non_static_stages.append(
                    {
                        "stage": stage,
                        "strategy_config": strategy_config,
                        "baseline_replicas": baseline_replicas,
                        "memory_per_replica_mb": memory_per_replica_mb,
                        "baseline_memory_demand_mb": stage_memory_demand,
                    }
                )

                total_memory_demand_mb += stage_memory_demand

                logger.debug(
                    f"Stage '{stage.name}': {baseline_replicas} replicas × "
                    f"{memory_per_replica_mb}MB = {stage_memory_demand}MB"
                )

    if not non_static_stages:
        logger.info("No stages with non-static strategies found")
        return resolved_config

    logger.info(f"Total baseline memory demand: {total_memory_demand_mb}MB from {len(non_static_stages)} stages")

    # Check if we need to scale down
    if total_memory_demand_mb <= available_memory_mb:
        logger.info("Memory demand within threshold, applying baseline replica counts")
        scaling_factor = 1.0
    else:
        # Calculate scaling factor to fit within memory threshold
        scaling_factor = available_memory_mb / total_memory_demand_mb
        logger.warning(
            f"Memory demand exceeds threshold by {((total_memory_demand_mb / available_memory_mb) - 1) * 100:.1f}%, "
            f"scaling down by factor of {scaling_factor:.3f}"
        )

    # Apply the resolved replica counts
    total_actual_memory_mb = 0
    for stage_info in non_static_stages:
        stage = stage_info["stage"]
        baseline_replicas = stage_info["baseline_replicas"]
        memory_per_replica_mb = stage_info["memory_per_replica_mb"]

        # Calculate scaled replica count (minimum 1)
        scaled_replicas = max(1, int(baseline_replicas * scaling_factor))
        actual_memory_mb = scaled_replicas * memory_per_replica_mb
        total_actual_memory_mb += actual_memory_mb

        # Replace the strategy config with a static replica count
        stage.replicas.static_replicas = scaled_replicas

        logger.info(
            f"Stage '{stage.name}': {baseline_replicas} → {scaled_replicas} replicas " f"({actual_memory_mb}MB)"
        )

    logger.info(
        f"Total actual memory allocation: {total_actual_memory_mb}MB "
        f"({(total_actual_memory_mb / total_memory_mb) * 100:.1f}% of system memory)"
    )

    return resolved_config


def _calculate_baseline_static_replicas(
    stage: StageConfig,
    strategy_config: ReplicaStrategyConfig,
    system_probe: SystemResourceProbe,
    static_memory_threshold: float = 0.75,
) -> int:
    """
    Calculate the baseline static replica count for a stage based on its strategy.

    Parameters
    ----------
    stage : StageConfig
        The stage configuration.
    strategy_config : ReplicaStrategyConfig
        The replica strategy configuration.
    system_probe : SystemResourceProbe
        System resource information.
    static_memory_threshold : float, optional
        The global static memory threshold (default: 0.75).

    Returns
    -------
    int
        The calculated baseline replica count.
    """
    strategy = strategy_config.strategy

    if strategy == ReplicaCalculationStrategy.STATIC:
        return strategy_config.value or 1

    elif strategy == ReplicaCalculationStrategy.CPU_PERCENTAGE:
        cpu_percent = strategy_config.cpu_percent or 0.5
        limit = strategy_config.limit or system_probe.cpu_count
        calculated = max(1, int(system_probe.cpu_count * cpu_percent))
        return min(calculated, limit)

    elif strategy == ReplicaCalculationStrategy.MEMORY_THRESHOLDING:
        # For memory thresholding, use a conservative approach for static mode
        memory_per_replica_mb = strategy_config.memory_per_replica_mb or 1000
        available_memory_mb = int(system_probe.total_memory_mb * 0.7)  # Conservative 70%
        calculated = max(1, available_memory_mb // memory_per_replica_mb)
        limit = strategy_config.limit or calculated
        return min(calculated, limit)

    elif strategy == ReplicaCalculationStrategy.MEMORY_STATIC_GLOBAL_PERCENT:
        # Use the global static memory threshold for calculation
        memory_per_replica_mb = strategy_config.memory_per_replica_mb or 1000
        available_memory_mb = int(system_probe.total_memory_mb * static_memory_threshold)
        calculated = max(1, available_memory_mb // memory_per_replica_mb)
        limit = strategy_config.limit or calculated
        return min(calculated, limit)

    else:
        logger.warning(f"Unknown replica strategy '{strategy}' for stage '{stage.name}', defaulting to 1 replica")
        return 1


def get_memory_intensive_stages(pipeline_config: PipelineConfigSchema) -> List[str]:
    """
    Identify stages that are memory-intensive and may need special handling.

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        The pipeline configuration.

    Returns
    -------
    List[str]
        List of stage names that are memory-intensive.
    """
    memory_intensive_stages = []

    for stage in pipeline_config.stages:
        if stage.replicas and stage.replicas.static_replicas:
            if isinstance(stage.replicas.static_replicas, ReplicaStrategyConfig):
                strategy_config = stage.replicas.static_replicas
                memory_per_replica_mb = strategy_config.memory_per_replica_mb or 0

                # Consider stages using >5GB per replica as memory-intensive
                if memory_per_replica_mb > 5000:
                    memory_intensive_stages.append(stage.name)

    return memory_intensive_stages
