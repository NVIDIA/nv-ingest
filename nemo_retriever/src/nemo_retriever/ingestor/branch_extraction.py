# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Branch execution for manifest-planned retriever ingest extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from nemo_retriever.graph import InprocessExecutor, RayDataExecutor
from nemo_retriever.graph.executor import call_pandas_function_on_arrow, preflight_executors
from nemo_retriever.common.url_fetch import restore_url_source_value
from nemo_retriever.graph.ingestor_runtime import (
    batch_tuning_to_node_overrides,
    build_graph,
    build_post_extract_graph,
    default_concurrency_node_names,
    require_ocr_actor_for_bounded_ocr,
    _image_embedding_requires_page_image,
)
from nemo_retriever.ingestor.manifest import (
    ExtractionBranchPlan,
    ResolvedExtractionInputs,
    format_branch_summary,
    resolve_branch_extraction_inputs,
)


logger = logging.getLogger(__name__)


def ensure_pandas_columns(batch_df: Any, *, columns: tuple[str, ...]) -> Any:
    """Pad a pandas batch to a stable schema before unioning branch outputs."""

    for column in columns:
        if column not in batch_df.columns:
            batch_df[column] = None
    return batch_df.loc[:, list(columns)]


def restore_source_urls(batch_df: Any, *, source_map: dict[str, str]) -> Any:
    """Restore caller-facing URLs after suffix-bearing transport identifiers."""

    if not source_map or batch_df.empty:
        return batch_df
    for column in ("path", "source_path", "source_id", "metadata"):
        if column in batch_df.columns:
            batch_df[column] = batch_df[column].map(lambda value: restore_url_source_value(value, source_map))
    return batch_df


def _driver_local_files_to_ray_dataset(ray_module: Any, paths: list[str]) -> Any:
    """Copy driver-local files into Ray's spillable object store.

    Each payload is read and submitted separately so the driver's Python heap
    does not retain the complete URL corpus. Ray owns the resulting blocks,
    making them available to workers that do not share the driver's temporary
    filesystem.
    """

    import pandas as pd

    frame_refs = []
    for path in paths:
        frame = pd.DataFrame([{"bytes": Path(path).read_bytes(), "path": path}])
        frame_refs.append(ray_module.put(frame))
    return ray_module.data.from_pandas_refs(frame_refs)


def _filesystem_files_to_ray_dataset(ray_module: Any, paths: list[str]) -> Any:
    """Create a Ray dataset for files visible to Ray workers."""

    try:
        return ray_module.data.read_binary_files(paths, include_paths=True)
    except FileNotFoundError as exc:
        from nemo_retriever.common.input_files import raise_input_path_not_found

        raise_input_path_not_found(paths, exc)


def _union_ray_datasets(datasets: list[Any]) -> Any:
    """Union source datasets before applying a shared extraction graph."""

    combined = datasets[0]
    for dataset in datasets[1:]:
        combined = combined.union(dataset)
    return combined


@dataclass
class ExtractionBranchExecutor:
    """Run manifest extraction branches and common post-extraction stages."""

    run_mode: str
    branches: tuple[ExtractionBranchPlan, ...]
    documents: list[str]
    buffers: list[tuple[str, BytesIO]]
    source_map: dict[str, str]
    driver_local_paths: set[str]
    inline_rows: list[dict[str, str]]
    split_config: dict[str, Any]
    extract_params: Any | None
    text_params: Any | None
    html_params: Any | None
    audio_chunk_params: Any | None
    asr_params: Any | None
    video_frame_params: Any | None
    video_text_dedup_params: Any | None
    av_fuse_params: Any | None
    embed_params: Any | None
    caption_params: Any | None
    dedup_params: Any | None
    store_params: Any | None
    vdb_upload_params: Any | None
    webhook_params: Any | None
    post_extract_order: tuple[str, ...]
    ray_address: str | None
    batch_size: int
    num_cpus: float
    num_gpus: float
    node_overrides: dict[str, dict[str, Any]]
    show_progress: bool
    allow_no_gpu: bool
    ensure_batch_runtime: Callable[[], tuple[Any, Any]]
    return_results: bool = True
    validate_batch: Callable[[Any], None] | None = None

    def execute(self) -> Any:
        logger.info(
            "Retriever ingest manifest planned %d extraction branches: %s",
            len(self.branches),
            format_branch_summary(self.branches),
        )
        if self.run_mode == "batch":
            return self._execute_batch()
        return self._execute_inprocess()

    def _execute_batch(self) -> Any:
        resolved_branches = [(branch, self._resolve_branch(branch)) for branch in self.branches]
        branch_graphs = [self._build_extraction_only_graph(extraction) for _, extraction in resolved_branches]
        require_ocr_actor_for_bounded_ocr(self.extract_params, branch_graphs)
        ray_module, cluster_resources = self.ensure_batch_runtime()
        effective_allow_no_gpu = self.allow_no_gpu or cluster_resources.total_gpu_count() == 0
        branch_datasets: list[Any] = []
        branch_executors: list[RayDataExecutor] = []
        branch_inputs: list[tuple[RayDataExecutor, Any]] = []
        for (branch, effective_extraction), graph in zip(resolved_branches, branch_graphs):
            logger.info(
                "Retriever ingest extraction branch family=%s files=%d graph_mode=%s",
                branch.family,
                len(branch.input_paths),
                effective_extraction.extraction_mode,
            )
            derived_overrides = batch_tuning_to_node_overrides(
                effective_extraction.extract_params,
                None,
                store_params=None,
                cluster_resources=cluster_resources,
                allow_no_gpu=effective_allow_no_gpu,
                caption_params=None,
                video_frame_params=effective_extraction.video_frame_params,
                extraction_mode=effective_extraction.extraction_mode,
            )
            file_paths, in_memory_rows = self._partition_branch_inputs(branch)
            ray_managed_paths = [path for path in file_paths if path in self.driver_local_paths]
            file_paths = [path for path in file_paths if path not in self.driver_local_paths]
            if ray_managed_paths:
                source_datasets: list[Any] = []
                if file_paths:
                    source_datasets.append(_filesystem_files_to_ray_dataset(ray_module, file_paths))
                source_datasets.append(_driver_local_files_to_ray_dataset(ray_module, ray_managed_paths))
                if in_memory_rows:
                    source_datasets.append(ray_module.data.from_items(_as_binary_source_rows(in_memory_rows)))
                inputs: list[Any] = [_union_ray_datasets(source_datasets)]
            else:
                inputs = []
                if file_paths:
                    inputs.append(file_paths)
                if in_memory_rows:
                    inputs.append(ray_module.data.from_items(in_memory_rows))
            for input_data in inputs:
                executor = self._ray_executor(
                    graph,
                    derived_overrides,
                    default_concurrency_node_names(effective_extraction.extract_params, None, None, None),
                    source_cpu_reservation=(
                        1 if file_paths and (ray_managed_paths or isinstance(input_data, list)) else 0
                    ),
                )
                branch_executors.append(executor)
                branch_inputs.append((executor, input_data))

        logger.info("Retriever ingest post-extraction stages: %s", format_post_stage_summary(self.post_extract_order))
        post_graph = build_post_extract_graph(
            dedup_params=self.dedup_params,
            embed_params=self.embed_params,
            caption_params=self.caption_params,
            store_params=self.store_params,
            vdb_upload_params=self.vdb_upload_params,
            webhook_params=self.webhook_params,
            stage_order=self.post_extract_order,
            reshape_content_before_embed=self._should_reshape_content_before_embed(),
        )
        post_overrides = batch_tuning_to_node_overrides(
            None,
            self.embed_params,
            store_params=self.store_params,
            cluster_resources=cluster_resources,
            allow_no_gpu=effective_allow_no_gpu,
            caption_params=self.caption_params,
            video_frame_params=None,
        )
        post_executor = self._ray_executor(
            post_graph,
            post_overrides,
            default_concurrency_node_names(None, self.embed_params, self.store_params, self.caption_params),
            source_cpu_reservation=0,
        )
        if hasattr(cluster_resources, "available_cpu_count"):
            normalization_cpus = 1.0 if len(branch_inputs) > 1 else 0.0
            preflight_executors(
                [*branch_executors, post_executor],
                cluster_resources,
                reserved_cpus=normalization_cpus,
            )

        for executor, input_data in branch_inputs:
            dataset = executor.build_dataset(input_data)
            if self.source_map:
                dataset = dataset.map_batches(
                    call_pandas_function_on_arrow,
                    batch_format="pyarrow",
                    fn_kwargs={
                        "fn": restore_source_urls,
                        "fn_kwargs": {"source_map": self.source_map},
                    },
                )
            branch_datasets.append(dataset)
        normalized = normalize_ray_branch_datasets(branch_datasets)
        combined = normalized[0]
        for branch_ds in normalized[1:]:
            combined = combined.union(branch_ds)
        if self.return_results:
            return post_executor.ingest(combined)
        return post_executor._ingest(combined, return_results=False, validate_batch=self.validate_batch)

    def _execute_inprocess(self) -> Any:
        frames = []
        for branch in self.branches:
            effective_extraction = self._resolve_branch(branch)
            logger.info(
                "Retriever ingest extraction branch family=%s files=%d graph_mode=%s",
                branch.family,
                len(branch.input_paths),
                effective_extraction.extraction_mode,
            )
            graph = self._build_extraction_only_graph(effective_extraction)
            executor = InprocessExecutor(graph, show_progress=self.show_progress)
            frames.append(
                restore_source_urls(executor.ingest(self._inprocess_branch_input(branch)), source_map=self.source_map)
            )

        combined = concat_dataframes(frames)
        logger.info("Retriever ingest post-extraction stages: %s", format_post_stage_summary(self.post_extract_order))
        post_graph = build_post_extract_graph(
            dedup_params=self.dedup_params,
            embed_params=self.embed_params,
            caption_params=self.caption_params,
            store_params=self.store_params,
            vdb_upload_params=self.vdb_upload_params,
            webhook_params=self.webhook_params,
            stage_order=self.post_extract_order,
            reshape_content_before_embed=self._should_reshape_content_before_embed(),
        )
        return InprocessExecutor(post_graph, show_progress=self.show_progress).ingest(combined)

    def _should_reshape_content_before_embed(self) -> bool:
        return any(branch.family in {"pdf", "image"} for branch in self.branches)

    def _resolve_branch(self, branch: ExtractionBranchPlan) -> ResolvedExtractionInputs:
        resolved = resolve_branch_extraction_inputs(
            branch,
            extract_params=self.extract_params,
            text_params=self.text_params,
            html_params=self.html_params,
            audio_chunk_params=self.audio_chunk_params,
            asr_params=self.asr_params,
            video_frame_params=self.video_frame_params,
            video_text_dedup_params=self.video_text_dedup_params,
            av_fuse_params=self.av_fuse_params,
        )
        if (
            branch.family == "pdf"
            and resolved.extract_params is not None
            and _image_embedding_requires_page_image(self.embed_params)
            and not resolved.extract_params.extract_page_as_image
        ):
            resolved = replace(
                resolved,
                extract_params=resolved.extract_params.model_copy(update={"extract_page_as_image": True}),
            )
        return resolved

    def _build_extraction_only_graph(self, effective_extraction: ResolvedExtractionInputs) -> Any:
        return build_graph(
            extraction_mode=effective_extraction.extraction_mode,
            extract_params=effective_extraction.extract_params,
            text_params=effective_extraction.text_params,
            html_params=effective_extraction.html_params,
            audio_chunk_params=effective_extraction.audio_chunk_params,
            asr_params=effective_extraction.asr_params,
            video_frame_params=effective_extraction.video_frame_params,
            video_text_dedup_params=effective_extraction.video_text_dedup_params,
            av_fuse_params=effective_extraction.av_fuse_params,
            split_config=self.split_config,
            stage_order=(),
        )

    def _ray_executor(
        self,
        graph: Any,
        derived_overrides: dict[str, dict[str, Any]],
        auto_concurrency_nodes: set[str],
        source_cpu_reservation: float,
    ) -> RayDataExecutor:
        return RayDataExecutor(
            graph,
            ray_address=self.ray_address,
            batch_size=self.batch_size,
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            node_overrides=merge_node_overrides(derived_overrides, self.node_overrides),
            auto_concurrency_nodes=auto_concurrency_nodes
            - {name for name, override in self.node_overrides.items() if "concurrency" in override},
            source_cpu_reservation=source_cpu_reservation,
        )

    def _inprocess_branch_input(self, branch: ExtractionBranchPlan) -> Any:
        inline_by_path = self._inline_rows_by_path()
        if not self.buffers and not any(path in inline_by_path for path in branch.input_paths):
            return list(branch.input_paths)

        import pandas as pd

        buffer_by_name = {name: buf for name, buf in self.buffers}
        file_paths: list[str] = []
        buffer_rows: list[dict[str, Any]] = []
        inline_rows: list[dict[str, str]] = []
        for path in branch.input_paths:
            if path in inline_by_path:
                inline_rows.append(inline_by_path[path])
            elif path in buffer_by_name:
                buffer_rows.append({"bytes": buffer_by_name[path].getvalue(), "path": path})
            else:
                file_paths.append(path)

        frames = []
        if file_paths:
            frames.append(InprocessExecutor._load_files(file_paths))
        if buffer_rows:
            frames.append(pd.DataFrame(buffer_rows))
        if inline_rows:
            frames.append(pd.DataFrame(inline_rows))
        return concat_dataframes(frames)

    def _inline_rows_by_path(self) -> dict[str, dict[str, str]]:
        return {row["path"]: row for row in self.inline_rows}

    def _partition_branch_inputs(self, branch: ExtractionBranchPlan) -> tuple[list[str], list[dict[str, Any]]]:
        inline_by_path = self._inline_rows_by_path()
        buffer_by_name = {name: buf for name, buf in self.buffers}
        file_paths: list[str] = []
        in_memory_rows: list[dict[str, Any]] = []
        for path in branch.input_paths:
            row = inline_by_path.get(path)
            if row is not None:
                in_memory_rows.append(row)
            elif path in buffer_by_name:
                in_memory_rows.append({"bytes": buffer_by_name[path].getvalue(), "path": path})
            else:
                file_paths.append(path)
        return file_paths, in_memory_rows


def _as_binary_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Align inline text with binary source datasets only for URL branches."""

    return [{"bytes": row["text"].encode("utf-8"), "path": row["path"]} if "text" in row else row for row in rows]


def merge_node_overrides(
    derived_overrides: dict[str, dict[str, Any]],
    explicit_overrides: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge generated and caller-supplied Ray node override dictionaries.

    Parameters
    ----------
    derived_overrides
        Overrides calculated from runtime resource heuristics and tuning
        parameters.
    explicit_overrides
        Overrides provided directly by the caller. Values here take precedence
        over matching keys from ``derived_overrides``.

    Returns
    -------
    dict[str, dict[str, Any]]
        A merged override mapping keyed by graph node name.
    """

    merged_overrides: dict[str, dict[str, Any]] = {}
    for node_name in set(derived_overrides) | set(explicit_overrides):
        merged_overrides[node_name] = {
            **derived_overrides.get(node_name, {}),
            **explicit_overrides.get(node_name, {}),
        }
    return merged_overrides


def concat_dataframes(frames: list[Any]) -> Any:
    """Concatenate branch DataFrames while preserving the union of columns.

    Parameters
    ----------
    frames
        Pandas DataFrames produced by extraction branches.

    Returns
    -------
    Any
        A pandas DataFrame with every column seen across the inputs. When
        ``frames`` is empty, returns an empty frame with ``bytes`` and ``path``
        columns.
    """

    import pandas as pd

    if not frames:
        return pd.DataFrame(columns=["bytes", "path"])
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    normalized = [frame.reindex(columns=columns) for frame in frames]
    return pd.concat(normalized, ignore_index=True, sort=False)


def normalize_ray_branch_datasets(branch_datasets: list[Any]) -> list[Any]:
    """Pad Ray branch datasets to a common schema before unioning them.

    Parameters
    ----------
    branch_datasets
        Ray ``Dataset`` objects produced by manifest extraction branches.

    Returns
    -------
    list[Any]
        Datasets whose pandas batches are projected to a stable column order.
        If any dataset lacks an already-known schema, the original datasets are
        returned unchanged so Ray does not eagerly execute extraction just to
        discover schema information.
    """

    columns: list[str] = []
    seen: set[str] = set()
    for dataset in branch_datasets:
        dataset_columns = ray_dataset_columns(dataset)
        if not dataset_columns:
            # Avoid eager schema discovery: Ray computes missing schemas by
            # executing a limit=1 plan, which pre-runs extraction branches.
            return branch_datasets
        for column in dataset_columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    if not columns:
        return branch_datasets
    stable_columns = tuple(columns)
    return [
        dataset.map_batches(
            call_pandas_function_on_arrow,
            batch_format="pyarrow",
            fn_kwargs={
                "fn": ensure_pandas_columns,
                "fn_kwargs": {"columns": stable_columns},
            },
        )
        for dataset in branch_datasets
    ]


def ray_dataset_columns(dataset: Any) -> tuple[str, ...]:
    """Return known column names from a Ray dataset without forcing execution.

    Parameters
    ----------
    dataset
        Ray ``Dataset`` or test double exposing ``schema``.

    Returns
    -------
    tuple[str, ...]
        Column names if Ray already has schema metadata, otherwise an empty
        tuple.

    Raises
    ------
    Exception
        Propagates non-compatibility errors raised by ``dataset.schema``.
    """

    try:
        schema = dataset.schema(fetch_if_missing=False)
    except TypeError:
        schema = dataset.schema()
    if schema is None:
        return ()
    names = getattr(schema, "names", None)
    if callable(names):
        names = names()
    if names is None:
        base_schema = getattr(schema, "base_schema", None)
        names = getattr(base_schema, "names", None) if base_schema is not None else None
        if callable(names):
            names = names()
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def format_post_stage_summary(post_extract_order: tuple[str, ...]) -> str:
    """Format post-extraction stage names for log messages.

    Parameters
    ----------
    post_extract_order
        Ordered stage names that will run after branch union.

    Returns
    -------
    str
        Comma-separated stage names, or ``"none"`` when no post stages are
        configured.
    """

    return ", ".join(post_extract_order) if post_extract_order else "none"
