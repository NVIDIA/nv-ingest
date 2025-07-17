# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Dict, Optional, Type

from pydantic import ValidationError

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.pipeline.pipeline_schema import PipelineConfig, StageConfig, StageType
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_actor_class_from_path
from nv_ingest_api.util.introspection.class_inspect import find_pydantic_config_schema
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    A high-level builder for creating and configuring an ingestion pipeline.

    This class translates a `PipelineConfig` object into a runnable `RayPipeline`,
    handling class resolution, configuration validation, replica calculation,
    and stage/edge construction.

    Attributes
    ----------
    _config : PipelineConfig
        The declarative configuration for the pipeline.
    _pipeline : RayPipeline
        The underlying RayPipeline instance being constructed.
    _system_resource_probe : SystemResourceProbe
        A utility to probe for available system resources like CPU cores.

    """

    def __init__(self, config: PipelineConfig, system_resource_probe: Optional[SystemResourceProbe] = None):
        """
        Initializes the IngestPipeline.

        Parameters
        ----------
        config : PipelineConfig
            The pipeline configuration object.
        system_resource_probe : Optional[SystemResourceProbe], optional
            A probe for system resources. If not provided, a default instance
            will be created. Defaults to None.
        """
        logger.debug(f"Initializing IngestPipeline for '{config.name}'.")
        self._config: PipelineConfig = config
        self._pipeline: RayPipeline = RayPipeline()
        self._system_resource_probe: SystemResourceProbe = system_resource_probe or SystemResourceProbe()
        self._is_built: bool = False

    def build(self) -> Dict[str, object]:
        """
        Builds the ingestion pipeline from the configuration.

        This method iterates through all enabled stages and edges defined in the
        configuration, resolves actor classes, validates configurations, calculates
        resource allocations, and constructs the final Ray pipeline.

        Returns
        -------
        Dict[str, object]
            A dictionary of the built Ray actor handles, keyed by stage name.

        Raises
        ------
        ValueError
            If a stage has an invalid type, a dependency is missing, or an actor
            class cannot be resolved.
        ValidationError
            If a stage's configuration fails Pydantic validation.
        AttributeError
            If an actor class is missing or cannot be found.
        """
        logger.info(f"Building ingestion pipeline '{self._config.name}'...")
        total_cpus: float = self._system_resource_probe.get_effective_cores()

        # 1. Add all stages defined in the config
        for stage_config in self._config.stages:
            if not stage_config.enabled:
                logger.info(f"Stage '{stage_config.name}' is disabled and will be skipped.")
                continue

            try:
                self._build_stage(stage_config, total_cpus)
            except (ValueError, AttributeError, ValidationError) as e:
                logger.error(f"Failed to build stage '{stage_config.name}': {e}", exc_info=True)
                raise

        # 2. Add all edges defined in the config
        if self._config.edges:
            for edge_config in self._config.edges:
                self._pipeline.make_edge(edge_config.from_stage, edge_config.to_stage, edge_config.queue_size)
                logger.info(f"Added edge from '{edge_config.from_stage}' to '{edge_config.to_stage}'.")

        # 3. Validate all dependencies
        self._validate_dependencies()

        # 4. Finalize the pipeline build and return the actors
        built_actors = self._pipeline.build()
        logger.info("Ingestion pipeline built successfully.")
        self._is_built = True
        return built_actors

    def _build_stage(self, stage_config: StageConfig, total_cpus: int) -> None:
        """Helper method to build a single pipeline stage."""
        logger.debug(f"Building stage '{stage_config.name}'...")
        stage_type_enum = StageType(stage_config.type)
        expected_base_class: Optional[Type] = {
            StageType.SOURCE: RayActorSourceStage,
            StageType.SINK: RayActorSinkStage,
            StageType.STAGE: RayActorStage,
        }.get(stage_type_enum)

        if not expected_base_class:
            raise ValueError(f"Invalid stage type '{stage_config.type}' for stage '{stage_config.name}'")

        actor_class = resolve_actor_class_from_path(stage_config.actor, expected_base_class)
        config_schema = find_pydantic_config_schema(actor_class, expected_base_class)
        config_instance = config_schema(**stage_config.config) if config_schema else None

        add_method = getattr(self._pipeline, f"add_{stage_config.type.value}", None)
        if not add_method:
            raise AttributeError(f"Pipeline has no method 'add_{stage_config.type.value}'")

        replicas = stage_config.replicas
        min_replicas, max_replicas = 1, 1
        if replicas and total_cpus:
            min_replicas = (
                replicas.cpu_count_min
                if replicas.cpu_count_min is not None
                else math.floor(replicas.cpu_percent_min * total_cpus)
            )
            max_replicas = (
                replicas.cpu_count_max
                if replicas.cpu_count_max is not None
                else math.floor(replicas.cpu_percent_max * total_cpus)
            )
            if max_replicas > 0:
                max_replicas = max(1, max_replicas)
            max_replicas = max(min_replicas, max_replicas)

        actor_kwarg = f"{stage_config.type.value}_actor"
        add_method(
            name=stage_config.name,
            **{actor_kwarg: actor_class},
            config=config_instance,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        logger.info(f"Added stage '{stage_config.name}' ({min_replicas}-{max_replicas} replicas) to the pipeline.")

    def _validate_dependencies(self) -> None:
        """
        Validates that all stage dependencies ('runs_after') are defined.

        Raises
        ------
        ValueError
            If a stage lists a dependency in `runs_after` that is not defined
            in the pipeline configuration.
        """
        all_stage_names = {s.name for s in self._config.stages}
        for stage_config in self._config.stages:
            if stage_config.runs_after:
                for dep_name in stage_config.runs_after:
                    if dep_name not in all_stage_names:
                        raise ValueError(
                            f"Stage '{stage_config.name}' has an invalid dependency: '{dep_name}'"
                            f" is not a defined stage."
                        )

    def start(self) -> None:
        """
        Starts the underlying RayPipeline, making it ready to process data.

        Raises
        ------
        RuntimeError
            If the pipeline has not been built by calling `build()` first.
        """
        if not self._is_built:
            raise RuntimeError("Pipeline has not been built yet. Call build() before start().")
        logger.info("Starting the ingestion pipeline...")
        self._pipeline.start()

    def stop(self) -> None:
        """
        Stops the underlying RayPipeline gracefully.
        """
        if self._pipeline:
            logger.info("Stopping the ingestion pipeline...")
            self._pipeline.stop()

    def get_pipeline(self) -> RayPipeline:
        """
        Returns the underlying RayPipeline instance.

        Returns
        -------
        RayPipeline
            The raw RayPipeline object.
        """
        return self._pipeline
