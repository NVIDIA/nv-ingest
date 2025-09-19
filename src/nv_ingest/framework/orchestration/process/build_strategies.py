# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic pipeline build strategies.

This module defines an abstract PipelineBuildStrategy and concrete implementations
for Ray and the lightweight Python execution frameworks. It encapsulates the
framework-specific environment preparation, pipeline construction, and lifecycle
operations (start/stop), allowing launch sites to be framework-agnostic.
"""

from __future__ import annotations

import logging
import os
import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Set, Type
import json

import ray
from ray import LoggingConfig

from nv_ingest.pipeline.config.replica_resolver import resolve_static_replicas
from nv_ingest.pipeline.pipeline_schema import (
    PipelineConfigSchema,
    StageConfig,
    StageType,
    ReplicaStrategyConfig,
)
from nv_ingest_api.util.string_processing.configuration import (
    pretty_print_pipeline_config,
)

# Ray framework imports for building the pipeline
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline, ScalingConfig
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.orchestration.ray.util.pipeline.tools import wrap_callable_as_stage

from nv_ingest_api.util.imports.callable_signatures import ingest_stage_callable_signature
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_actor_class_from_path, resolve_callable_from_path
from nv_ingest_api.util.introspection.class_inspect import (
    find_pydantic_config_schema,
    find_pydantic_config_schema_unified,
)
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

# Python framework (lightweight) imports
from nv_ingest.framework.orchestration.python.python_pipeline import PythonPipeline
from nv_ingest.framework.orchestration.python.stages.meta.python_stage_base import PythonStage

logger = logging.getLogger(__name__)


class PipelineBuildStrategy(ABC):
    """
    Framework-agnostic strategy for building and running pipelines.

    Implementations encapsulate all framework-specific setup (e.g., Ray
    initialization), pipeline construction and lifecycle operations, so that
    callers do not need to be framework-aware.
    """

    def __init__(self, pipeline_config: PipelineConfigSchema) -> None:
        self._config = pipeline_config

    @abstractmethod
    def prepare_environment(self) -> None:
        """Perform any framework-specific environment initialization."""

    @abstractmethod
    def build(self) -> Any:
        """Build the pipeline and return a framework-specific pipeline handle."""

    @abstractmethod
    def start(self, pipeline_handle: Any) -> None:
        """Start the pipeline for processing."""

    @abstractmethod
    def stop(self, pipeline_handle: Any) -> None:
        """Stop the pipeline and perform best-effort cleanup."""


class RayPipelineBuildStrategy(PipelineBuildStrategy):
    """
    Strategy for Ray-based pipeline execution.

    Responsibilities:
    - Initialize Ray runtime
    - Resolve replicas and pretty print final config
    - Build pipeline via IngestPipelineBuilder
    - Start/stop the pipeline and manage Ray shutdown
    """

    def __init__(self, pipeline_config: PipelineConfigSchema) -> None:
        super().__init__(pipeline_config)
        self._system_resource_probe: SystemResourceProbe = SystemResourceProbe()

    def prepare_environment(self) -> None:
        if not ray.is_initialized():
            logging_config = _build_logging_config_from_env()

            # Clear existing handlers from root logger before Ray adds its handler
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            logger.info("Cleared existing root logger handlers to prevent Ray logging duplicates")

            ray.init(
                namespace="nv_ingest_ray",
                ignore_reinit_error=True,
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
                logging_config=logging_config,  # Ray will add its own StreamHandler
                _system_config={
                    "local_fs_capacity_threshold": 0.9,
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {
                                "directory_path": [
                                    "/tmp/ray_spill_testing_0",
                                    "/tmp/ray_spill_testing_1",
                                    "/tmp/ray_spill_testing_2",
                                    "/tmp/ray_spill_testing_3",
                                ],
                                "buffer_size": 100_000_000,
                            },
                        },
                    ),
                },
            )

    def build(self) -> Any:
        # Resolve static replicas and pretty print the final configuration
        config = resolve_static_replicas(self._config)
        try:
            pretty_output = pretty_print_pipeline_config(config, config_path=None)
            logger.info("\n" + pretty_output)
        except Exception:
            pass

        # Construct RayPipeline scaling configuration
        scaling_config = ScalingConfig(
            dynamic_memory_scaling=not config.pipeline.disable_dynamic_scaling,
            dynamic_memory_threshold=config.pipeline.dynamic_memory_threshold,
            pid_kp=config.pipeline.pid_controller.kp,
            pid_ki=config.pipeline.pid_controller.ki,
            pid_ema_alpha=config.pipeline.pid_controller.ema_alpha,
            pid_target_queue_depth=config.pipeline.pid_controller.target_queue_depth,
            pid_penalty_factor=config.pipeline.pid_controller.penalty_factor,
            pid_error_boost_factor=config.pipeline.pid_controller.error_boost_factor,
            rcm_memory_safety_buffer_fraction=config.pipeline.pid_controller.rcm_memory_safety_buffer_fraction,
        )

        pipeline: RayPipeline = RayPipeline(scaling_config=scaling_config)

        # Validate dependencies
        self._validate_dependencies(config)

        # Build stages
        built_stages: Set[str] = set()
        total_cpus = os.cpu_count() or 1
        for stage_cfg in config.stages:
            if not stage_cfg.enabled:
                logger.info(f"Stage '{stage_cfg.name}' is disabled. Skipping.")
                continue
            self._build_stage(config, pipeline, stage_cfg, total_cpus)
            built_stages.add(stage_cfg.name)

        # Add edges
        for edge_cfg in config.edges:
            if not (edge_cfg.from_stage in built_stages and edge_cfg.to_stage in built_stages):
                logger.warning(
                    f"Skipping edge from '{edge_cfg.from_stage}' to '{edge_cfg.to_stage}' "
                    f"because one or both stages are disabled or failed to build."
                )
                continue
            pipeline.make_edge(
                from_stage=edge_cfg.from_stage,
                to_stage=edge_cfg.to_stage,
                queue_size=edge_cfg.queue_size,
            )

        # Finalize Ray pipeline
        pipeline.build()
        return pipeline

    def start(self, pipeline_handle: Any) -> None:
        # pipeline_handle is RayPipeline
        pipeline_handle.start()

    def stop(self, pipeline_handle: Any) -> None:
        try:
            if pipeline_handle is not None:
                pipeline_handle.stop()
        finally:
            if ray.is_initialized():
                ray.shutdown()

    # ------------------------
    # Internal helper methods
    # ------------------------

    def _validate_dependencies(self, config: PipelineConfigSchema) -> None:
        """
        Validates stage dependencies, checking for undefined stages and circular dependencies.
        """
        all_stage_names = {s.name for s in config.stages}
        dependency_graph: Dict[str, List[str]] = {s.name: s.runs_after for s in config.stages}

        # Check for dependencies on non-existent stages
        for stage_name, deps in dependency_graph.items():
            for dep_name in deps:
                if dep_name not in all_stage_names:
                    raise ValueError(
                        f"Stage '{stage_name}' has an invalid dependency: '{dep_name}' is not a defined stage."
                    )

        visiting: Set[str] = set()
        visited: Set[str] = set()

        for stage_name in all_stage_names:
            if stage_name not in visited:
                self._detect_cycle_util(stage_name, dependency_graph, visiting, visited)

    def _detect_cycle_util(
        self, stage_name: str, graph: Dict[str, List[str]], visiting: Set[str], visited: Set[str]
    ) -> None:
        visiting.add(stage_name)
        for dependency in graph.get(stage_name, []) or []:
            if dependency in visiting:
                raise ValueError(f"Circular dependency detected involving stage '{stage_name}' and '{dependency}'.")
            if dependency not in visited:
                self._detect_cycle_util(dependency, graph, visiting, visited)
        visiting.remove(stage_name)
        visited.add(stage_name)

    def _build_stage(
        self, config: PipelineConfigSchema, pipeline: RayPipeline, stage_config: StageConfig, total_cpus: int
    ) -> None:
        """Builds and adds a single stage to the Ray pipeline."""
        logger.debug(f"Building stage '{stage_config.name}'...")
        stage_type_enum = StageType(stage_config.type)
        expected_base_class: Optional[Type] = {
            StageType.SOURCE: RayActorSourceStage,
            StageType.SINK: RayActorSinkStage,
            StageType.STAGE: RayActorStage,
        }.get(stage_type_enum)

        if not expected_base_class:
            raise ValueError(f"Invalid stage type '{stage_config.type}' for stage '{stage_config.name}'")

        # Handle callable vs actor stage configurations
        if stage_config.callable:
            callable_fn = resolve_callable_from_path(
                stage_config.callable, signature_schema=ingest_stage_callable_signature
            )
            config_schema = find_pydantic_config_schema_unified(callable_fn, param_name="stage_config")
            if not config_schema:
                raise ValueError(
                    f"Callable stage '{stage_config.name}' must have a Pydantic schema in its stage_config parameter"
                )
            actor_class = wrap_callable_as_stage(callable_fn, config_schema, required_tasks=stage_config.task_filters)
            config_instance = config_schema(**stage_config.config) if config_schema else None
        else:
            actor_class = resolve_actor_class_from_path(stage_config.stage_impl, expected_base_class)
            config_schema = find_pydantic_config_schema(actor_class, expected_base_class)
            config_instance = config_schema(**stage_config.config) if config_schema else None

        add_method = getattr(pipeline, f"add_{stage_config.type.value}", None)
        if not add_method:
            raise AttributeError(f"Pipeline has no method 'add_{stage_config.type.value}'")

        replicas = stage_config.replicas
        min_replicas, max_replicas = 1, 1

        # Check if dynamic scaling is disabled by checking pipeline config
        dynamic_scaling_disabled = getattr(config.pipeline, "disable_dynamic_scaling", False)

        if replicas and total_cpus:
            # Handle new replica configuration format
            if hasattr(replicas, "min_replicas") and replicas.min_replicas is not None:
                min_replicas = replicas.min_replicas
            elif getattr(replicas, "cpu_count_min", None) is not None:  # Legacy support
                min_replicas = replicas.cpu_count_min
            elif getattr(replicas, "cpu_percent_min", None) is not None:  # Legacy support
                min_replicas = math.floor(replicas.cpu_percent_min * total_cpus)

            # For max_replicas, prioritize based on scaling mode
            if dynamic_scaling_disabled:
                # Static scaling mode - use static_replicas if available
                if hasattr(replicas, "static_replicas") and replicas.static_replicas is not None:
                    if isinstance(replicas.static_replicas, int):
                        max_replicas = replicas.static_replicas
                        min_replicas = replicas.static_replicas  # In static mode, min == max
                        logger.debug(f"Stage '{stage_config.name}': Using resolved static replicas = {max_replicas}")
                    else:
                        logger.warning(
                            f"Stage '{stage_config.name}': static_replicas not resolved to int, using "
                            "legacy calculation"
                        )
                        max_replicas = self._calculate_legacy_max_replicas(replicas, total_cpus)
                else:
                    max_replicas = self._calculate_legacy_max_replicas(replicas, total_cpus)
            else:
                # Dynamic scaling mode - use max_replicas
                if hasattr(replicas, "max_replicas") and replicas.max_replicas is not None:
                    if isinstance(replicas.max_replicas, int):
                        max_replicas = replicas.max_replicas
                    else:
                        max_replicas = self._calculate_strategy_based_replicas(
                            stage_config.name, replicas.max_replicas, total_cpus
                        )
                        logger.debug(
                            f"Stage '{stage_config.name}': max_replicas calculated from strategy = {max_replicas}"
                        )
                else:
                    max_replicas = self._calculate_legacy_max_replicas(replicas, total_cpus)

            # Ensure max_replicas is not less than min_replicas
            max_replicas = max(min_replicas, max_replicas)

        actor_kwarg = f"{stage_config.type.value}_actor"
        add_method(
            name=stage_config.name,
            **{actor_kwarg: actor_class},
            config=config_instance,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        logger.debug(f"Added stage '{stage_config.name}' ({min_replicas}-{max_replicas} replicas) to the pipeline.")

    def _calculate_legacy_max_replicas(self, replicas, total_cpus: int) -> int:
        if getattr(replicas, "cpu_count_max", None) is not None:
            return replicas.cpu_count_max
        elif getattr(replicas, "cpu_percent_max", None) is not None:
            return math.ceil(replicas.cpu_percent_max * total_cpus)
        else:
            return 1

    def _calculate_strategy_based_replicas(
        self, stage_name: str, strategy_config: ReplicaStrategyConfig, total_cpus: int
    ) -> int:
        """Calculate replica count based on ReplicaStrategyConfig for dynamic scaling."""
        from nv_ingest.pipeline.pipeline_schema import ReplicaCalculationStrategy

        strategy = strategy_config.strategy

        if strategy == ReplicaCalculationStrategy.STATIC:
            return strategy_config.value or 1
        elif strategy == ReplicaCalculationStrategy.CPU_PERCENTAGE:
            cpu_percent = strategy_config.cpu_percent or 0.5
            limit = strategy_config.limit or total_cpus
            calculated = max(1, int(total_cpus * cpu_percent))
            result = min(calculated, limit)
            logger.debug(
                f"Stage '{stage_name}': CPU_PERCENTAGE strategy: {cpu_percent:.1%} of {total_cpus} "
                f"CPUs = {calculated}, limited to {result}"
            )
            return result
        elif strategy == ReplicaCalculationStrategy.MEMORY_THRESHOLDING:
            memory_per_replica_mb = strategy_config.memory_per_replica_mb or 1000
            available_memory_mb = int(self._system_resource_probe.total_memory_mb * 0.8)
            calculated = max(1, available_memory_mb // memory_per_replica_mb)
            limit = strategy_config.limit or calculated
            result = min(calculated, limit)
            logger.debug(
                f"Stage '{stage_name}': MEMORY_THRESHOLDING strategy:"
                f" {available_memory_mb}MB / {memory_per_replica_mb}MB = {calculated}, limited to {result}"
            )
            return result
        elif strategy == ReplicaCalculationStrategy.MEMORY_STATIC_GLOBAL_PERCENT:
            memory_per_replica_mb = strategy_config.memory_per_replica_mb or 1000
            dynamic_threshold = getattr(self._config.pipeline, "dynamic_memory_threshold", 0.8)
            available_memory_mb = int(self._system_resource_probe.total_memory_mb * dynamic_threshold)
            calculated = max(1, available_memory_mb // memory_per_replica_mb)
            limit = strategy_config.limit or calculated
            result = min(calculated, limit)
            logger.debug(
                f"Stage '{stage_name}': MEMORY_STATIC_GLOBAL_PERCENT strategy (dynamic):"
                f" {available_memory_mb}MB / {memory_per_replica_mb}MB = {calculated}, limited to {result}"
            )
            return result
        else:
            logger.warning(f"Unknown replica strategy '{strategy}' for stage '{stage_name}', defaulting to 1 replica")
            return 1


class PythonPipelineBuildStrategy(PipelineBuildStrategy):
    """
    Strategy for Python (lightweight) pipeline execution.

    Current implementation wires a message-broker source to a message-broker sink
    and allows for future insertion of in-process processing functions.
    """

    def prepare_environment(self) -> None:
        # No special environment initialization required for pure Python mode
        return None

    def build(self) -> Any:
        """
        Build a strictly linear Python pipeline from the provided config.

        Rules:
        - Exactly one SOURCE (in-degree 0, out-degree 1)
        - Zero or more STAGE nodes (each in-degree 1, out-degree 1)
        - Exactly one SINK (in-degree 1, out-degree 0)
        - All enabled stages must be connected in a single chain
        - Callables are not yet supported in Python mode
        - Replicas are not used (implicitly 1 per stage)
        """
        # Pretty print final config for parity with Ray
        try:
            pretty_output = pretty_print_pipeline_config(self._config, config_path=None)
            logger.info("\n" + pretty_output)
        except Exception:
            pass

        # Validate and derive a linear ordering of enabled stages
        ordered_stage_cfgs = self._validate_and_order_linear_stages(self._config)

        # Instantiate pipeline in linear mode
        pipeline: PythonPipeline = PythonPipeline(enable_streaming=False)

        # Build and add each stage in order
        for stage_cfg in ordered_stage_cfgs:
            if stage_cfg.callable is not None:
                raise ValueError(
                    f"PythonPipelineBuildStrategy does not yet support callable stages: '{stage_cfg.name}'"
                )

            if stage_cfg.stage_impl is None:
                raise ValueError(f"Stage '{stage_cfg.name}' must specify 'stage_impl' when using the Python framework")

            # Resolve class and its Pydantic config schema
            actor_class = resolve_actor_class_from_path(stage_cfg.stage_impl, PythonStage)
            config_schema = find_pydantic_config_schema(actor_class, PythonStage)
            config_instance = config_schema(**(stage_cfg.config or {})) if config_schema else None

            # Add to pipeline according to stage type (no replicas supported)
            stage_type_enum = StageType(stage_cfg.type)
            if stage_type_enum == StageType.SOURCE:
                pipeline.add_source(name=stage_cfg.name, source_actor=actor_class, config=config_instance)
            elif stage_type_enum == StageType.SINK:
                pipeline.add_sink(name=stage_cfg.name, sink_actor=actor_class, config=config_instance)
            else:
                pipeline.add_stage(name=stage_cfg.name, stage_actor=actor_class, config=config_instance)

        return pipeline

    def start(self, pipeline_handle: Any) -> None:
        # pipeline_handle is PythonPipeline
        pipeline_handle.start()

    def stop(self, pipeline_handle: Any) -> None:
        try:
            if pipeline_handle is not None:
                pipeline_handle.stop()
        except Exception:
            pass

    # ------------------------
    # Internal helper methods
    # ------------------------

    def _validate_and_order_linear_stages(self, config: PipelineConfigSchema) -> List[StageConfig]:
        """
        Ensure the enabled stage/edge subgraph is a single linear chain and return stages in order.

        Raises ValueError if the pipeline is not strictly linear.
        """
        # Consider only enabled stages
        enabled_stages: Dict[str, StageConfig] = {s.name: s for s in (config.stages or []) if s.enabled}
        if not enabled_stages:
            raise ValueError("Python pipeline requires at least one enabled stage")

        # Build degree maps using only edges that connect enabled stages
        indegree: Dict[str, int] = {name: 0 for name in enabled_stages}
        outdegree: Dict[str, int] = {name: 0 for name in enabled_stages}
        adjacency: Dict[str, str] = {}

        for e in config.edges or []:
            u = getattr(e, "from_stage", None)
            v = getattr(e, "to_stage", None)
            if u in enabled_stages and v in enabled_stages:
                outdegree[u] += 1
                indegree[v] += 1
                if u in adjacency:
                    # Multiple outgoing edges from a node implies non-linear
                    raise ValueError(f"Pipeline is not linear: stage '{u}' has multiple outgoing edges")
                adjacency[u] = v

        # Classify by declared type
        sources = [s for s in enabled_stages.values() if StageType(s.type) == StageType.SOURCE]
        sinks = [s for s in enabled_stages.values() if StageType(s.type) == StageType.SINK]

        if len(sources) != 1 or len(sinks) != 1:
            raise ValueError("Pipeline must have exactly one SOURCE and one SINK in Python mode")

        # Degree constraints for linear chain
        for name, s in enabled_stages.items():
            st = StageType(s.type)
            if st == StageType.SOURCE:
                if indegree[name] != 0 or outdegree[name] != 1:
                    raise ValueError(f"SOURCE stage '{name}' must have in-degree 0 and out-degree 1")
            elif st == StageType.SINK:
                if indegree[name] != 1 or outdegree[name] != 0:
                    raise ValueError(f"SINK stage '{name}' must have in-degree 1 and out-degree 0")
            else:
                if indegree[name] != 1 or outdegree[name] != 1:
                    raise ValueError(f"STAGE '{name}' must have in-degree 1 and out-degree 1")

        # Edge count must be N-1 for a single chain
        edge_count = sum(
            1 for e in (config.edges or []) if e.from_stage in enabled_stages and e.to_stage in enabled_stages
        )
        node_count = len(enabled_stages)
        if edge_count != node_count - 1:
            raise ValueError(
                f"Pipeline is not linear: expected {node_count - 1} edges connecting"
                f" {node_count} stages, found {edge_count}"
            )

        # Produce ordered list by following edges from the unique source
        start = sources[0].name
        ordered_names: List[str] = [start]
        while ordered_names[-1] in adjacency:
            ordered_names.append(adjacency[ordered_names[-1]])

        if len(ordered_names) != node_count:
            raise ValueError("Pipeline is not a single connected chain of enabled stages")

        # Map back to StageConfig objects in order
        return [enabled_stages[n] for n in ordered_names]


def _build_logging_config_from_env() -> LoggingConfig:
    """
    Build Ray LoggingConfig from environment variables.
    Mirrors the configuration used in execution.launch_pipeline.
    """

    # Apply package-level preset defaults first
    preset_level = os.environ.get("INGEST_RAY_LOG_LEVEL", "DEVELOPMENT").upper()

    # Define preset configurations
    presets = {
        "PRODUCTION": {
            "RAY_LOGGING_LEVEL": "ERROR",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "9",  # 10GB total
            "RAY_DISABLE_IMPORT_WARNING": "1",
            "RAY_USAGE_STATS_ENABLED": "0",
        },
        "DEVELOPMENT": {
            "RAY_LOGGING_LEVEL": "INFO",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "19",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
        "DEBUG": {
            "RAY_LOGGING_LEVEL": "DEBUG",
            "RAY_LOGGING_ENCODING": "JSON",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "name,funcName,lineno",
            "RAY_DEDUP_LOGS": "0",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "536870912",  # 512MB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "39",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
    }

    # Validate preset level
    if preset_level not in presets:
        logger.warning(
            f"Invalid INGEST_RAY_LOG_LEVEL '{preset_level}', using DEVELOPMENT. "
            f"Valid presets: {list(presets.keys())}"
        )
        preset_level = "DEVELOPMENT"

    # Apply preset defaults (only if env var not already set)
    preset_config = presets[preset_level]
    for key, default_value in preset_config.items():
        if key not in os.environ:
            os.environ[key] = default_value

    logger.info(f"Applied Ray logging preset: {preset_level}")

    # Get log level from environment, default to INFO
    log_level = os.environ.get("RAY_LOGGING_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        logger.warning(f"Invalid RAY_LOGGING_LEVEL '{log_level}', using INFO. Valid levels: {valid_levels}")
        log_level = "INFO"

    # Get encoding format from environment, default to TEXT
    encoding = os.environ.get("RAY_LOGGING_ENCODING", "TEXT").upper()

    # Validate encoding
    valid_encodings = ["TEXT", "JSON"]
    if encoding not in valid_encodings:
        logger.warning(f"Invalid RAY_LOGGING_ENCODING '{encoding}', using TEXT. Valid encodings: {valid_encodings}")
        encoding = "TEXT"

    # Get additional standard logger attributes
    additional_attrs_str = os.environ.get("RAY_LOGGING_ADDITIONAL_ATTRS", "")
    additional_log_standard_attrs = []
    if additional_attrs_str:
        additional_log_standard_attrs = [attr.strip() for attr in additional_attrs_str.split(",") if attr.strip()]

    # Set log deduplication environment variable if specified
    dedup_logs = os.environ.get("RAY_DEDUP_LOGS", "1")
    if dedup_logs is not None:
        os.environ["RAY_DEDUP_LOGS"] = str(dedup_logs)

    # Set log to driver environment variable if specified
    log_to_driver = os.environ.get("RAY_LOG_TO_DRIVER", "0")
    if log_to_driver is not None:
        os.environ["RAY_LOG_TO_DRIVER"] = str(log_to_driver)

    # Configure log rotation settings
    rotate_bytes = os.environ.get("RAY_LOGGING_ROTATE_BYTES", "1073741824")  # Default: 1GB per file
    if rotate_bytes is not None:
        try:
            rotate_bytes_int = int(rotate_bytes)
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = str(rotate_bytes_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BYTES '{rotate_bytes}', using default (1GB)")
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = "1073741824"

    rotate_backup_count = os.environ.get("RAY_LOGGING_ROTATE_BACKUP_COUNT", "19")  # Default: 19 backups (20GB Max)
    if rotate_backup_count is not None:
        try:
            backup_count_int = int(rotate_backup_count)
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = str(backup_count_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BACKUP_COUNT '{rotate_backup_count}', using default (19)")
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = "19"

    # Configure Ray internal logging verbosity
    disable_import_warning = os.environ.get("RAY_DISABLE_IMPORT_WARNING", "0")
    if disable_import_warning is not None:
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = str(disable_import_warning)

    # Configure usage stats collection
    usage_stats_enabled = os.environ.get("RAY_USAGE_STATS_ENABLED", "1")
    if usage_stats_enabled is not None:
        os.environ["RAY_USAGE_STATS_ENABLED"] = str(usage_stats_enabled)

    # Create LoggingConfig with validated parameters
    logging_config = LoggingConfig(
        encoding=encoding,
        log_level=log_level,
        additional_log_standard_attrs=additional_log_standard_attrs,
    )

    logger.info(
        f"Ray logging configured: preset={preset_level}, level={log_level}, encoding={encoding}, "
        f"additional_attrs={additional_log_standard_attrs}, "
        f"dedup_logs={os.environ.get('RAY_DEDUP_LOGS', '1')}, "
        f"log_to_driver={os.environ.get('RAY_LOG_TO_DRIVER', '0')}, "
        f"rotate_bytes={os.environ.get('RAY_LOGGING_ROTATE_BYTES', '1073741824')}, "
        f"rotate_backup_count={os.environ.get('RAY_LOGGING_ROTATE_BACKUP_COUNT', '19')}"
    )

    return logging_config


def _find_stage_by_name_local(config: PipelineConfigSchema, name: str) -> Optional[Any]:
    for st in getattr(config, "stages", []) or []:
        if getattr(st, "name", None) == name:
            return st
    return None
