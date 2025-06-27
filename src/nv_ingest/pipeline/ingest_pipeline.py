# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import inspect
import math
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.pipeline.pipeline_schema import PipelineConfig
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_callable_from_path
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    A high-level orchestrator for building and managing an ingestion pipeline
    from a declarative YAML configuration.

    This class is responsible for:
    1. Parsing a YAML file that defines the pipeline's stages, connections, and configurations.
    2. Resolving all stage actors (classes or functions) from their import paths.
    3. Constructing an underlying RayPipeline instance with the specified topology.
    4. Providing a simple interface to start and stop the pipeline.
    """

    def __init__(self, config_path: str, pipeline: Optional[RayPipeline] = None):
        """
        Initializes the IngestPipeline with a path to a YAML configuration file.

        Parameters
        ----------
        config_path : str
            The path to the pipeline's YAML configuration file.
        pipeline : Optional[RayPipeline], optional
            An existing RayPipeline instance to build upon. If None, a new one is created.
        """
        self.config_path = config_path
        self._config: PipelineConfig = self._load_config(config_path)
        self._pipeline: RayPipeline = pipeline or RayPipeline()
        self._stage_configs: Dict[str, BaseModel] = {}
        self._system_resource_probe = SystemResourceProbe()

    def _load_config(self, config_path: str) -> PipelineConfig:
        """
        Loads and validates the YAML configuration file against the Pydantic schema.

        Parameters
        ----------
        config_path : str
            The path to the YAML file.

        Returns
        -------
        PipelineConfig
            The validated configuration object.
        """
        logger.info(f"Loading pipeline configuration from: {config_path}")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return PipelineConfig(**config_dict)

    def _resolve_config_schema(self, actor_class: Any) -> Any:
        """
        Introspects the actor's __init__ to find its config schema.
        """
        try:
            sig = inspect.signature(actor_class.__init__)
            config_param = sig.parameters.get("config")
            if config_param and issubclass(config_param.annotation, BaseModel):
                return config_param.annotation
        except (ValueError, TypeError):
            # Handles built-ins or other non-introspectable callables
            pass
        return None

    def build(self):
        """
        Builds the RayPipeline instance based on the loaded configuration.
        """
        logger.info("Building the ingestion pipeline...")
        total_cpus = self._system_resource_probe.get_effective_cores()

        # Sort stages by phase for logical processing
        sorted_stages = sorted(self._config.stages, key=lambda s: s.phase.value)

        # 1. Add all stages defined in the config
        for stage_config in sorted_stages:
            if not stage_config.enabled:
                logger.info(f"Stage '{stage_config.name}' is disabled and will be skipped.")
                continue

            # Resolve the actor class from the provided path
            actor_class = resolve_callable_from_path(stage_config.actor)

            # Resolve the stage's config schema and create an instance of it
            config_schema = self._resolve_config_schema(actor_class)
            config_instance = config_schema(**stage_config.config) if config_schema else None
            if config_instance:
                self._stage_configs[stage_config.name] = config_instance

            # Determine the method to add the stage to the pipeline (add_stage or add_sink)
            add_method = getattr(self._pipeline, f"add_{stage_config.type}", None)
            if not add_method:
                raise ValueError(f"Invalid stage type '{stage_config.type}' for stage '{stage_config.name}'")

            # Resolve replica counts
            replicas = stage_config.replicas

            # Min replicas
            if replicas.cpu_count_min is not None:
                min_replicas = replicas.cpu_count_min
            elif replicas.cpu_percent_min is not None:
                min_replicas = math.floor(replicas.cpu_percent_min * total_cpus)
            else:
                min_replicas = 1  # Default value

            # Max replicas
            if replicas.cpu_count_max is not None:
                max_replicas = replicas.cpu_count_max
            elif replicas.cpu_percent_max is not None:
                max_replicas = math.floor(replicas.cpu_percent_max * total_cpus)
            else:
                max_replicas = total_cpus  # Default value

            # Ensure max_replicas is at least 1 if not 0, and at least min_replicas
            if max_replicas > 0:
                max_replicas = max(1, max_replicas)
            max_replicas = max(min_replicas, max_replicas)

            add_method(
                name=stage_config.name,
                stage_actor=actor_class,
                config=config_instance,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            )
            logger.info(f"Added stage '{stage_config.name}' of type '{stage_config.type}' to the pipeline.")

        # 2. Inject dependencies between phases before validation
        self._inject_phase_dependencies()

        # 3. Add all edges defined in the config
        for edge_config in self._config.edges:
            self._pipeline.make_edge(edge_config.from_stage, edge_config.to_stage, queue_size=edge_config.queue_size)
            logger.info(f"Added edge from '{edge_config.from_stage}' to '{edge_config.to_stage}'.")

        # 4. Validate all dependencies
        self._validate_dependencies()

        # 5. Finalize the pipeline build
        self._pipeline.build()
        logger.info("Ingestion pipeline built successfully.")

    def _inject_phase_dependencies(self):
        """
        Automatically adds 'runs_after' dependencies between consecutive phases.

        Ensures that every stage in a phase must run after all stages in the
        immediately preceding phase.
        """
        logger.info("Injecting cross-phase dependencies...")

        stages_by_phase: Dict[int, List[str]] = {}
        for stage in self._config.stages:
            if not stage.enabled:
                continue
            phase = stage.phase.value
            if phase not in stages_by_phase:
                stages_by_phase[phase] = []
            stages_by_phase[phase].append(stage.name)

        sorted_phases = sorted(stages_by_phase.keys())

        for i in range(1, len(sorted_phases)):
            current_phase_num = sorted_phases[i]
            prev_phase_num = sorted_phases[i - 1]

            prev_phase_stages = stages_by_phase[prev_phase_num]

            for stage_config in self._config.stages:
                if stage_config.phase.value == current_phase_num:
                    # Add all stages from the previous phase as dependencies
                    existing_deps = set(stage_config.runs_after)
                    new_deps = existing_deps.union(prev_phase_stages)

                    if len(new_deps) > len(existing_deps):
                        logger.debug(
                            f"Injecting dependencies for stage '{stage_config.name}': runs after {prev_phase_stages}"
                        )
                        stage_config.runs_after = sorted(list(new_deps))

    def _validate_dependencies(self):
        """
        Validates that all 'runs_after' constraints are satisfied by the pipeline graph.
        """
        logger.info("Validating pipeline dependencies...")

        # Build a simple adjacency list representation of the graph
        adj = {stage.name: [] for stage in self._config.stages if stage.enabled}
        for edge in self._config.edges:
            if edge.from_stage in adj and edge.to_stage in adj:
                adj[edge.from_stage].append(edge.to_stage)

        # Define a helper for graph traversal (DFS)
        def is_reachable(start: str, end: str) -> bool:
            visited = set()
            stack = [start]
            while stack:
                current = stack.pop()
                if current == end:
                    return True
                if current not in visited:
                    visited.add(current)
                    for neighbor in adj.get(current, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            return False

        # Check each stage's dependencies
        for stage in self._config.stages:
            if not stage.enabled:
                continue

            for dependency in stage.runs_after:
                # Ensure the dependency stage exists and is enabled
                dependency_stage_config = next((s for s in self._config.stages if s.name == dependency), None)
                if not dependency_stage_config or not dependency_stage_config.enabled:
                    logger.warning(
                        f"Stage '{stage.name}' has a dependency '{dependency}' which is missing or disabled. "
                        f"Skipping validation for this dependency."
                    )
                    continue

                if not is_reachable(dependency, stage.name):
                    raise ValueError(
                        f"Dependency validation failed: Stage '{stage.name}' is configured to run after "
                        f"'{dependency}', but there is no valid path between them in the pipeline graph."
                    )
                logger.debug(f"Successfully validated that '{stage.name}' runs after '{dependency}'.")

        logger.info("All pipeline dependencies are satisfied.")

    def start(self):
        """
        Starts the underlying RayPipeline.
        """
        if not self._pipeline:
            raise RuntimeError("Pipeline has not been built yet. Call build() before start().")
        logger.info("Starting the ingestion pipeline...")
        self._pipeline.start()

    def stop(self):
        """
        Stops the underlying RayPipeline.
        """
        if self._pipeline:
            logger.info("Stopping the ingestion pipeline...")
            self._pipeline.stop()

    def get_pipeline(self) -> RayPipeline:
        """
        Returns the underlying RayPipeline instance.
        """
        return self._pipeline
