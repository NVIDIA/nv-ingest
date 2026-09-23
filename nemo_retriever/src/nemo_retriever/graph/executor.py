# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline executors that run a :class:`Graph` against input data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set
import pandas as pd

from nemo_retriever.common.nvtx import batch_phase

if TYPE_CHECKING:
    import ray.data

from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.graph.operator_resolution import resolve_graph
from nemo_retriever.common.ray_resource_hueristics import ClusterResources
from nemo_retriever.common.input_files import (
    _is_explicit_glob_path,
    expand_input_file_patterns,
    raise_input_path_not_found,
)
from nemo_retriever.common.ray_runtime import ensure_local_ray_runtime
from nemo_retriever.common import ray_resource_hueristics as _rrh
from nemo_retriever.common.ray_resource_hueristics import (
    gather_cluster_resources,
    NEMOTRON_PARSE_BATCH_SIZE,
    VLLM_GPUS_PER_ACTOR,
    OCR_GPUS_PER_ACTOR,
)

import logging

logger = logging.getLogger(__name__)

# Heuristic GPU fraction for GPUOperator nodes that load a local model.
# Reuses the same baseline constant as the batch ingest mode.
_DEFAULT_GPU_OPERATOR_NUM_GPUS = OCR_GPUS_PER_ACTOR
_STREAM_INGEST_PREFETCH_BATCHES = 1
_DatasetSegment = Literal["all", "before_stream_ingest", "after_stream_ingest"]

# Ray can briefly report stale available resources after a Dataset releases its
# actors. Keep this wait short: it only covers teardown accounting propagation,
# not capacity held by another active workload.
_RESOURCE_RELEASE_WAIT_SECONDS = 1.0
_RESOURCE_RELEASE_POLL_SECONDS = 0.01


def _contains_null_arrow_child(data_type: Any) -> bool:
    """Return whether a nested Arrow type contains an inferred null child."""
    import pyarrow as pa

    if pa.types.is_null(data_type):
        return True
    if pa.types.is_struct(data_type):
        return any(_contains_null_arrow_child(field.type) for field in data_type)
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        return _contains_null_arrow_child(data_type.value_type)
    if pa.types.is_map(data_type):
        return _contains_null_arrow_child(data_type.key_type) or _contains_null_arrow_child(data_type.item_type)
    return False


def _is_row_unsafe_arrow_column(field: Any) -> bool:
    """Return whether pandas must not back a column with its Arrow array.

    Two column shapes leave pandas unable to read rows once Arrow-backed dtypes
    are preserved:

    * Nested types carrying an inferred ``null`` child, such as a ``page_image``
      struct whose ``image_b64`` was stripped. pandas indexes the null child at
      the parent's row offset, but pyarrow sizes null children independently of
      their parent, so row access raises ``ArrowIndexError`` and an Arrow
      roundtrip reports a child shorter than its parent.
    * Ray's pickled-object extension columns, whose payloads pandas would
      otherwise interpret as malformed extension arrays.
    """
    if getattr(field.type, "extension_name", None) == "ray.data.arrow_pickled_object":
        return True
    return _contains_null_arrow_child(field.type)


def _materialize_row_unsafe_columns(table: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite Arrow columns pandas cannot index as plain object columns."""
    import pyarrow as pa

    if not isinstance(table, (pa.Table, pa.RecordBatch)):
        return frame

    for index, field in enumerate(table.schema):
        if not _is_row_unsafe_arrow_column(field):
            continue
        frame[field.name] = pd.Series(table.column(index).to_pylist(), index=frame.index, dtype=object)
    return frame


def _normalize_object_tensor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert object-backed Ray tensor columns to ordinary pandas objects."""
    from ray.data.extensions import TensorDtype

    columns = [
        name for name, dtype in frame.dtypes.items() if isinstance(dtype, TensorDtype) and dtype.element_dtype.hasobject
    ]
    if not columns:
        return frame

    normalized = frame.copy(deep=False)
    for name in columns:
        normalized[name] = frame[name].astype(object)
    return normalized


def arrow_table_to_pandas(table: Any) -> pd.DataFrame:
    """Convert a Ray Arrow batch to a row-safe pandas DataFrame.

    Ray 2.56+ preserves Arrow-backed pandas dtypes, so columns pandas cannot
    index through their Arrow arrays (nested null children, Ray pickled-object
    extensions) are materialized as ordinary object columns. Native pandas
    blocks with object-backed Ray tensor columns are normalized the same way.
    Every other column keeps its Arrow-backed dtype.
    """
    if isinstance(table, pd.DataFrame):
        return _normalize_object_tensor_columns(table)

    from ray.data.block import BlockAccessor

    frame = BlockAccessor.for_block(table).to_pandas()
    return _normalize_object_tensor_columns(_materialize_row_unsafe_columns(table, frame))


@batch_phase("ray.materialize")
def ray_dataset_to_pandas(dataset: ray.data.Dataset) -> pd.DataFrame:
    """Materialize a Ray Dataset without returning malformed Arrow arrays.

    Ray 2.56+ enables Arrow-backed pandas conversion by default. Calling
    ``Dataset.to_pandas()`` directly can therefore expose nested Arrow columns
    that pandas cannot index by row. Forcing a pandas block to Arrow can also
    fail for object-backed tensor columns. Read each block in its native format
    and convert it through
    :func:`arrow_table_to_pandas` before concatenating so the public SDK result
    is safe to consume with standard pandas APIs.

    Parameters
    ----------
    dataset
        Ray dataset to materialize in its native block formats.

    Returns
    -------
    pandas.DataFrame
        Row-safe DataFrame containing all rows from ``dataset``.
    """
    frames = [arrow_table_to_pandas(block) for block in dataset.iter_batches(batch_format=None, batch_size=None)]
    if frames:
        return pd.concat(frames, ignore_index=True)

    schema = dataset.schema()
    names = getattr(schema, "names", None)
    return pd.DataFrame(columns=list(names) if names is not None else None)


@batch_phase("result.concat")
def _concat_terminal_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate retained terminal batches under one low-cardinality range."""

    return pd.concat(frames, ignore_index=True)


def call_pandas_function_on_arrow(
    table: Any,
    *,
    fn: Any,
    fn_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Invoke a pandas batch function through the safe Arrow boundary."""
    return fn(arrow_table_to_pandas(table), **(fn_kwargs or {}))


class _ArrowPandasOperatorAdapter:
    """Convert valid Arrow batches to pandas before invoking an NRL operator."""

    def __init__(self, operator_class: type, operator_kwargs: dict[str, Any]) -> None:
        self._operator = operator_class(**operator_kwargs)

    def __call__(self, table: Any) -> Any:
        return self._operator(arrow_table_to_pandas(table))


def _make_arrow_pandas_operator_adapter(operator_class: type) -> type[_ArrowPandasOperatorAdapter]:
    """Keep the wrapped operator recognizable in Ray plans and worker logs."""
    adapter_name = f"{operator_class.__name__}ArrowPandasAdapter"
    return type(adapter_name, (_ArrowPandasOperatorAdapter,), {})


def _preserves_pandas_output(operator_class: type, operator_kwargs: dict[str, Any]) -> bool:
    """Return whether an operator's heterogeneous rows should stay in pandas."""
    return bool(
        getattr(operator_class, "PRESERVE_PANDAS_OUTPUT", False) or operator_kwargs.get("preserve_pandas_output", False)
    )


def _requires_stable_pandas_blocks(nodes: list[Node]) -> bool:
    """Return whether repartitions must not promote object columns to tensors."""
    return any(_preserves_pandas_output(node.operator_class, node.operator_kwargs) for node in nodes)


def _concurrency_bounds(concurrency: Any) -> tuple[int, int, int]:
    """Return the minimum, maximum, and initial actor-pool sizes."""
    if isinstance(concurrency, tuple):
        if len(concurrency) not in (2, 3):
            raise ValueError("Ray actor-pool concurrency tuples must contain (min, max) or (min, max, initial)")
        minimum, maximum = int(concurrency[0]), int(concurrency[1])
        initial = int(concurrency[2]) if len(concurrency) == 3 else minimum
        return minimum, maximum, initial
    return 1, int(concurrency), 1


def _concurrency_target(concurrency: Any) -> int:
    """Return the largest actor-pool size that resource planning can permit."""
    return _concurrency_bounds(concurrency)[1]


def _concurrency_initial(concurrency: Any) -> int:
    """Return the startup floor for an automatically sized pool."""
    return _concurrency_bounds(concurrency)[2]


def _concurrency_required(concurrency: Any) -> int:
    """Admit an explicit elastic pool at startup, or a fixed pool at full size."""
    return _concurrency_initial(concurrency) if isinstance(concurrency, tuple) else int(concurrency)


def _planned_concurrency(concurrency: Any, planned: int) -> Any:
    """Preserve Ray's actor-pool tuple while capping its maximum size."""
    minimum, _maximum, initial = _concurrency_bounds(concurrency)
    if isinstance(concurrency, tuple) and len(concurrency) == 3:
        return (minimum, max(minimum, initial, planned), initial)
    if isinstance(concurrency, tuple) and len(concurrency) == 2:
        return (min(minimum, planned), planned)
    return planned


def preflight_executors(
    executors: list[Any],
    cluster_resources: ClusterResources,
    *,
    reserved_cpus: float = 0.0,
) -> None:
    """Plan all lazy executor pools and known task work against one Ray resource snapshot."""
    executor_nodes = []
    for executor in executors:
        nodes = executor._linearize(resolve_graph(executor.graph, cluster_resources))
        sink_index = executor._stream_ingest_index(nodes)
        if sink_index is not None:
            nodes = [node for index, node in enumerate(nodes) if index != sink_index]
        executor_nodes.append((executor, nodes))
    effective_resources = _preflight_executor_nodes(
        executor_nodes,
        cluster_resources.available_cpu_count(),
        cluster_resources.available_gpu_count(),
        reserved_cpus=reserved_cpus,
        cluster_resources=cluster_resources,
    )
    for executor in executors:
        executor._resources_preflight_complete = True
        executor._preflight_source_cpu_reservation = executor._source_cpu_reservation
        executor._preflight_cluster_resources = effective_resources or cluster_resources


def _refresh_cluster_resources() -> ClusterResources:
    """Read a fresh Ray cluster resource snapshot."""
    import ray

    return gather_cluster_resources(ray)


def _preflight_executor_nodes(
    executor_nodes: list[tuple[RayDataExecutor, list[Node]]],
    available_cpus: int,
    available_gpus: int,
    *,
    reserved_cpus: float = 0.0,
    cluster_resources: ClusterResources | None = None,
) -> ClusterResources | None:
    """Admit resolved pools together, adjusting only automatically sized concurrency."""
    entries = []
    if reserved_cpus < 0:
        raise ValueError("reserved_cpus must be non-negative")
    for executor, nodes in executor_nodes:
        for node in nodes:
            override = executor._node_overrides.get(node.name, {})
            concurrency = override.get("concurrency", 1)
            entries.append(
                (
                    executor,
                    node.name,
                    concurrency,
                    _concurrency_target(concurrency),
                    _concurrency_initial(concurrency),
                    float(override.get("num_cpus", executor._default_num_cpus)),
                    executor._scheduled_num_gpus(node, override, available_gpus),
                    node.name in executor._auto_concurrency_nodes,
                )
            )
    fixed = [item for item in entries if not item[7]]
    auto = [item for item in entries if item[7]]
    fixed_cpu = sum(_concurrency_required(item[2]) * item[5] for item in fixed)
    source_cpu_reservation = sum(executor._source_cpu_reservation for executor, _nodes in executor_nodes)
    task_cpu_reservation = source_cpu_reservation + reserved_cpus
    fixed_gpu = sum(_concurrency_required(item[2]) * item[6] for item in fixed)
    min_cpu = sum(item[4] * item[5] for item in auto)
    min_gpu = sum(item[4] * item[6] for item in auto)
    requested_cpu = task_cpu_reservation + fixed_cpu + min_cpu
    requested_gpu = fixed_gpu + min_gpu
    effective_resources = cluster_resources
    if (
        (requested_cpu > available_cpus or requested_gpu > available_gpus)
        and cluster_resources is not None
        and requested_cpu <= cluster_resources.total_cpu_count()
        and requested_gpu <= cluster_resources.total_gpu_count()
    ):
        deadline = time.monotonic() + _RESOURCE_RELEASE_WAIT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(_RESOURCE_RELEASE_POLL_SECONDS)
            effective_resources = _refresh_cluster_resources()
            available_cpus = effective_resources.available_cpu_count()
            available_gpus = effective_resources.available_gpu_count()
            if requested_cpu <= available_cpus and requested_gpu <= available_gpus:
                break
        if requested_cpu <= available_cpus and requested_gpu <= available_gpus:
            # Recompute node GPU reservations and auto-concurrency against the
            # refreshed snapshot. A local GPU operator initially observed with
            # zero available GPUs otherwise remains budgeted as a CPU-only node.
            _preflight_executor_nodes(
                executor_nodes,
                available_cpus,
                available_gpus,
                reserved_cpus=reserved_cpus,
            )
            return effective_resources
    actor_cpu_budget = available_cpus - task_cpu_reservation
    if requested_cpu > available_cpus or requested_gpu > available_gpus:
        raise ValueError(
            "Infeasible Ray CPU/GPU plan: requested at least "
            f"{requested_cpu:g} CPUs (including {source_cpu_reservation:g} for source reads and "
            f"{reserved_cpus:g} for other non-actor tasks) "
            f"and {requested_gpu:g} GPUs, but Ray reports "
            f"{available_cpus} CPUs and {available_gpus} GPUs available. "
            "Reduce explicit *_workers or node_overrides concurrency, or wait for cluster capacity."
        )
    used_cpu, used_gpu = fixed_cpu + min_cpu, fixed_gpu + min_gpu
    planned = {(id(item[0]), item[1]): item[4] for item in auto}
    while True:
        candidates = sorted(
            (item for item in auto if planned[(id(item[0]), item[1])] < item[3]),
            key=lambda item: planned[(id(item[0]), item[1])] / item[3],
        )
        selected = next(
            (
                item
                for item in candidates
                if used_cpu + item[5] <= actor_cpu_budget and used_gpu + item[6] <= available_gpus
            ),
            None,
        )
        if selected is None:
            break
        planned[(id(selected[0]), selected[1])] += 1
        used_cpu += selected[5]
        used_gpu += selected[6]
    for executor, name, concurrency, _target, _initial, _cpu, _gpu, _auto in auto:
        executor._node_overrides.setdefault(name, {})["concurrency"] = _planned_concurrency(
            concurrency, planned[(id(executor), name)]
        )
    executors = [executor for executor, _nodes in executor_nodes]
    logger.info(
        "Ray batch resource preflight admitted actor_cpus=%g/%g source_read_cpus=%g "
        "other_task_cpus=%g actor_gpus=%g/%g pools=%s",
        used_cpu,
        actor_cpu_budget,
        source_cpu_reservation,
        reserved_cpus,
        used_gpu,
        available_gpus,
        [
            f"executor[{executors.index(executor)}].{name}={planned[(id(executor), name)]}"
            for executor, name, _concurrency, _target, _initial, _cpu, _gpu, _auto in auto
        ],
    )
    return effective_resources


class AbstractExecutor(ABC):
    """Base class for pipeline executors.

    An executor takes a :class:`Graph` at init time and provides an
    :meth:`ingest` method that feeds data through the graph.
    """

    def __init__(self, graph: Graph) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"graph must be a Graph, got {type(graph).__name__}")
        self.graph = graph

    @abstractmethod
    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Execute the graph against *data* and return results."""
        ...


class InprocessExecutor(AbstractExecutor):
    """Executor that runs a :class:`Graph` in-process on pandas DataFrames.

    No Ray dependency — each node's operator is constructed once from
    ``operator_class(**operator_kwargs)`` and called sequentially on the
    accumulated DataFrame.

    Only linear (single-root, no fan-out) graphs are currently supported.
    """

    def __init__(self, graph: Graph, *, show_progress: bool = True) -> None:
        super().__init__(graph)
        self._show_progress = show_progress

    @staticmethod
    def _linearize(graph: Graph) -> List[Node]:
        """Walk a single-root, single-child-per-node graph and return an ordered list."""
        if not graph.roots:
            return []
        if len(graph.roots) > 1:
            raise ValueError("InprocessExecutor currently supports single-root graphs only.")
        ordered: List[Node] = []
        node = graph.roots[0]
        while node is not None:
            ordered.append(node)
            if len(node.children) > 1:
                raise ValueError(
                    f"InprocessExecutor does not support fan-out. "
                    f"Node {node.name!r} has {len(node.children)} children."
                )
            node = node.children[0] if node.children else None
        return ordered

    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Run the graph in-process on pandas DataFrames.

        Parameters
        ----------
        data
            A ``pandas.DataFrame``, a file path (str), or a list of file
            paths.  When paths are provided, each file is read as raw bytes
            and combined into a single DataFrame with ``bytes`` and ``path``
            columns before being passed through the graph.

        Returns
        -------
        pandas.DataFrame
            The result after all operators have been applied.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (str, list)):
            df = self._load_files(expand_input_file_patterns(data))
        else:
            raise TypeError(
                f"data must be a pandas.DataFrame, file path, or list of paths, " f"got {type(data).__name__}"
            )

        resolved_graph = resolve_graph(self.graph, _rrh.gather_local_resources())
        nodes = self._linearize(resolved_graph)
        operators = []
        for node in nodes:
            op = node.operator_class(**node.operator_kwargs)
            operators.append((node.name, op))

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        if self._show_progress and tqdm is not None:
            pbar = tqdm(operators, desc="Pipeline stages", unit="stage")
            for name, op in pbar:
                pbar.set_postfix_str(name)
                df = op.run(df)
        else:
            for _name, op in operators:
                df = op.run(df)

        return df

    @staticmethod
    def _load_files(paths: List[str]) -> "pd.DataFrame":
        """Read files as raw bytes into a DataFrame with ``bytes`` and ``path`` columns."""
        import pandas as pd
        from pathlib import Path

        rows = []
        for p in paths:
            fp = Path(p)
            if fp.is_file():
                rows.append({"bytes": fp.read_bytes(), "path": str(fp.resolve())})
            elif not _is_explicit_glob_path(p):
                raise_input_path_not_found(p)
        if not rows:
            return pd.DataFrame(columns=["bytes", "path"])
        return pd.DataFrame(rows)


class RayDataExecutor(AbstractExecutor):
    """Executor that builds a Ray Data pipeline from a :class:`Graph`.

    For each :class:`Node` in the graph the executor appends a
    ``map_batches`` stage that uses the node's ``operator_class`` with
    ``fn_constructor_kwargs`` for deferred construction on Ray workers.
    This ensures heavy GPU models are loaded on workers, not serialised
    from the driver.

    The operator's ``__call__`` (defined on :class:`AbstractOperator`)
    delegates to ``run()``, so each ``map_batches`` stage executes the
    full preprocess → process → postprocess pipeline.

    Only linear (single-root, no fan-out) graphs are currently supported.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        ray_address: Optional[str] = None,
        batch_size: int = 1,
        batch_format: str = "pandas",
        num_cpus: float = 1,
        num_gpus: float = 0,
        node_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_concurrency_nodes: Optional[Set[str]] = None,
        source_cpu_reservation: float = 0,
    ) -> None:
        super().__init__(graph)
        source_cpu_reservation = float(source_cpu_reservation)
        if not math.isfinite(source_cpu_reservation) or source_cpu_reservation < 0:
            raise ValueError("source_cpu_reservation must be a finite, non-negative CPU value.")
        self._preflight_cluster_resources: ClusterResources | None = None
        self._ray_address = ray_address
        self._source_cpu_reservation = source_cpu_reservation
        # ``preflight_executors`` records the source reservation it budgeted.
        # A filesystem input supplied later must not silently increase that
        # shared plan: re-planning this executor alone would ignore its peers.
        self._preflight_source_cpu_reservation: float | None = None
        self._default_batch_size = batch_size
        self._default_batch_format = batch_format
        self._default_num_cpus = num_cpus
        self._default_num_gpus = num_gpus
        self._node_overrides = node_overrides or {}
        self._auto_concurrency_nodes = auto_concurrency_nodes or set()
        self._resources_preflight_complete = False

    def _has_remote_endpoint(self, node: Node) -> bool:
        """Return whether a node delegates inference to a remote endpoint."""
        if any("invoke_url" in key and bool(value) for key, value in node.operator_kwargs.items()):
            return True
        return any(
            hasattr(value, "model_dump")
            and any("invoke_url" in key and bool(item) for key, item in value.model_dump(exclude_none=True).items())
            for value in node.operator_kwargs.values()
        )

    def _scheduled_num_gpus(self, node: Node, overrides: Dict[str, Any], available_gpus: int) -> float:
        """Resolve the GPU reservation passed to Ray for a graph node."""
        if "num_gpus" in overrides:
            return float(overrides["num_gpus"])
        if not issubclass(node.operator_class, GPUOperator) or self._has_remote_endpoint(node):
            return float(self._default_num_gpus)
        if available_gpus > 0:
            from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor, NemotronParseGPUActor
            from nemo_retriever.operators.extract.caption.caption import CaptionGPUActor

            if issubclass(node.operator_class, (NemotronParseActor, NemotronParseGPUActor, CaptionGPUActor)):
                return max(float(self._default_num_gpus), VLLM_GPUS_PER_ACTOR)
            return max(float(self._default_num_gpus), _DEFAULT_GPU_OPERATOR_NUM_GPUS)
        logger.warning(
            "Node %r is a GPUOperator with no remote endpoint but the Ray cluster reports 0 available GPUs. "
            "The actor will be scheduled with num_gpus=0 and will likely fail to load its model. "
            "Pass --ocr-invoke-url / --page-elements-invoke-url / --embed-invoke-url to use remote endpoints, "
            "or ensure GPUs are visible to Ray.",
            node.name,
        )
        return float(self._default_num_gpus)

    def _preflight_resources(
        self,
        nodes: List[Node],
        available_cpus: int,
        available_gpus: int,
        *,
        cluster_resources: ClusterResources | None = None,
    ) -> ClusterResources | None:
        """Reduce unspecified pools and reject plans that exclude known task work."""
        return _preflight_executor_nodes(
            [(self, nodes)],
            available_cpus,
            available_gpus,
            cluster_resources=cluster_resources,
        )

    @staticmethod
    def _linearize(graph: Graph) -> List[Node]:
        """Walk a single-root, single-child-per-node graph and return an ordered list."""
        if not graph.roots:
            return []
        if len(graph.roots) > 1:
            raise ValueError("RayDataExecutor currently supports single-root graphs only.")
        ordered: List[Node] = []
        node = graph.roots[0]
        while node is not None:
            ordered.append(node)
            if len(node.children) > 1:
                raise ValueError(
                    f"RayDataExecutor does not support fan-out. "
                    f"Node {node.name!r} has {len(node.children)} children."
                )
            node = node.children[0] if node.children else None
        return ordered

    @staticmethod
    def _stream_ingest_index(nodes: List[Node]) -> int | None:
        """Return one VDB streaming position, falling back for ambiguous graphs."""
        from nemo_retriever.operators.vdb import IngestVdbOperator

        positions = [
            index
            for index, node in enumerate(nodes)
            if isinstance(node.operator, IngestVdbOperator) and node.operator._supports_stream_ingest()
        ]
        return positions[0] if len(positions) == 1 else None

    def ingest(self, data: Any, **kwargs: Any) -> Any:
        """Build, execute, and materialize a Ray Data pipeline from the graph."""

        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"RayDataExecutor.ingest() does not accept setting(s): {unsupported}")

        nodes = self._linearize(self.graph)
        sink_index = self._stream_ingest_index(nodes)
        if sink_index is None:
            return ray_dataset_to_pandas(self.build_dataset(data))

        sink_operator = nodes[sink_index].operator
        has_downstream_nodes = sink_index + 1 < len(nodes)

        dataset = self._build_dataset(
            data,
            segment="before_stream_ingest",
        )

        terminal_frames: list[pd.DataFrame] = []
        batch_iterator = iter(
            dataset.iter_batches(
                batch_format=None,
                batch_size=None,
                prefetch_batches=_STREAM_INGEST_PREFETCH_BATCHES,
            )
        )

        def retained_batches() -> Iterator[pd.DataFrame]:
            for block in batch_iterator:
                frame = arrow_table_to_pandas(block)
                terminal_frames.append(frame)
                yield frame

        try:
            sink_operator._stream_ingest(retained_batches())
        finally:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                close()

        if has_downstream_nodes:
            import ray.data as rd

            if terminal_frames:
                continuation_input = rd.from_pandas(terminal_frames)
            else:
                schema = dataset.schema()
                names = getattr(schema, "names", None)
                continuation_input = rd.from_pandas(pd.DataFrame(columns=list(names) if names is not None else None))
            downstream = self._build_dataset(
                continuation_input,
                segment="after_stream_ingest",
                input_preserves_pandas_output=True,
            )
            return ray_dataset_to_pandas(downstream)

        if terminal_frames:
            return _concat_terminal_frames(terminal_frames)
        schema = dataset.schema()
        names = getattr(schema, "names", None)
        return pd.DataFrame(columns=list(names) if names is not None else None)

    def build_dataset(self, data: Any, **kwargs: Any) -> Any:
        """Build a lazy Ray Data pipeline from the graph.

        Parameters
        ----------
        data
            Input to ``ray.data.read_binary_files`` (a path or list of glob patterns)
            **or** an already-constructed ``ray.data.Dataset``.

        Returns
        -------
        ray.data.Dataset
            The lazy Ray dataset with all graph stages appended.
        """
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(f"RayDataExecutor.build_dataset() does not accept setting(s): {unsupported}")
        return self._build_dataset(data)

    def _build_dataset(
        self,
        data: Any,
        *,
        segment: _DatasetSegment = "all",
        input_preserves_pandas_output: bool = False,
    ) -> Any:
        """Build the complete graph or one private streaming-ingest segment."""

        ray = ensure_local_ray_runtime(self._ray_address)
        import ray.data as rd

        if not isinstance(data, (rd.Dataset, str, list)):
            raise TypeError(
                f"data must be a path/glob string, list of globs, or ray.data.Dataset, " f"got {type(data).__name__}"
            )

        input_paths: Optional[List[str]] = None
        if isinstance(data, (str, list)):
            input_paths = expand_input_file_patterns(data)

        ctx = rd.DataContext.get_current()
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False
        is_filesystem_source = not isinstance(data, rd.Dataset)
        if is_filesystem_source:
            required_source_cpu_reservation = 1
            if self._resources_preflight_complete:
                planned_source_cpu_reservation = self._preflight_source_cpu_reservation
                if (
                    planned_source_cpu_reservation is None
                    or planned_source_cpu_reservation < required_source_cpu_reservation
                ):
                    raise ValueError(
                        "Filesystem inputs require 1 CPU for Ray Data source reads, but shared Ray resource "
                        "preflight completed without that reservation. Construct RayDataExecutor with "
                        "source_cpu_reservation=1 before calling preflight_executors."
                    )
            else:
                self._source_cpu_reservation = required_source_cpu_reservation

        cluster = self._preflight_cluster_resources or gather_cluster_resources(ray)
        available_gpus = cluster.available_gpu_count()
        resolved_graph = resolve_graph(self.graph, cluster)
        all_nodes = self._linearize(resolved_graph)
        sink_index = self._stream_ingest_index(all_nodes)
        if sink_index is not None and segment == "before_stream_ingest":
            nodes = all_nodes[:sink_index]
        elif sink_index is not None and segment == "after_stream_ingest":
            nodes = all_nodes[sink_index + 1 :]
        else:
            nodes = all_nodes
        requires_stable_pandas_blocks = input_preserves_pandas_output or _requires_stable_pandas_blocks(nodes)

        if isinstance(data, rd.Dataset):
            ds = rd.Dataset.copy(data, _deep_copy=True) if requires_stable_pandas_blocks else data
            if requires_stable_pandas_blocks:
                # Ray copies this context into repartition workers. Disabling
                # Arrow output and tensor promotion only on an operator actor is
                # too late because Ray converts its result after the call.
                ds.context.batch_to_block_arrow_format = False
                ds.context.enable_tensor_extension_casting = False
        else:
            try:
                if requires_stable_pandas_blocks:
                    # read_binary_files snapshots the current context onto the
                    # new Dataset; restore the process default immediately so
                    # unrelated pipelines retain Ray's standard behavior.
                    original_arrow_format = ctx.batch_to_block_arrow_format
                    original_tensor_extension_casting = ctx.enable_tensor_extension_casting
                    ctx.batch_to_block_arrow_format = False
                    ctx.enable_tensor_extension_casting = False
                try:
                    ds = rd.read_binary_files(input_paths, include_paths=True)
                finally:
                    if requires_stable_pandas_blocks:
                        ctx.batch_to_block_arrow_format = original_arrow_format
                        ctx.enable_tensor_extension_casting = original_tensor_extension_casting
            except FileNotFoundError as exc:
                raise_input_path_not_found(input_paths or [], exc)
        if nodes and not self._resources_preflight_complete:
            effective_resources = self._preflight_resources(
                nodes,
                cluster.available_cpu_count(),
                available_gpus,
                cluster_resources=cluster,
            )
            if effective_resources is not None:
                cluster = effective_resources
                available_gpus = cluster.available_gpu_count()
        preserve_pandas_output = input_preserves_pandas_output
        for node in nodes:
            overrides = dict(self._node_overrides.get(node.name, {}))
            target_num_rows_per_block = overrides.pop("target_num_rows_per_block", None)
            batch_size = overrides.pop("batch_size", self._default_batch_size)
            batch_format = overrides.pop("batch_format", self._default_batch_format)
            num_cpus = overrides.pop("num_cpus", self._default_num_cpus)
            # Ray 2.49+ requires concurrency to be specified for callable classes.
            # Default to 1 when not explicitly set via node_overrides.
            if "concurrency" not in overrides:
                overrides["concurrency"] = 1

            # vLLM-backed actors handle their own batching efficiently
            # (continuous batching), so feed them more rows per map_batches call.
            from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor, NemotronParseGPUActor
            from nemo_retriever.operators.extract.caption.caption import CaptionGPUActor

            if batch_size == self._default_batch_size and issubclass(
                node.operator_class, (NemotronParseActor, NemotronParseGPUActor, CaptionGPUActor)
            ):
                batch_size = NEMOTRON_PARSE_BATCH_SIZE

            # Self-join operators (AudioVisualFuser, VideoFrameTextDedup) need
            # the entire dataset in one batch — see the repartition site below
            # for the actual single-block enforcement.
            requires_global_batch = bool(getattr(node.operator_class, "REQUIRES_GLOBAL_BATCH", False))
            if requires_global_batch:
                batch_size = None
                target_num_rows_per_block = None

            num_gpus = self._scheduled_num_gpus(node, overrides, available_gpus)
            overrides.pop("num_gpus", None)

            if requires_global_batch:
                # ``num_blocks=1`` is exact; ``target_num_rows_per_block`` is a
                # streaming best-effort cap that can leave joins missing rows.
                # When the operator declares ``GLOBAL_BATCH_GROUP_KEYS`` and
                # concurrency > 1, hash-partition by those keys so rows sharing
                # the keys stay co-located while blocks distribute across actors.
                group_keys = list(getattr(node.operator_class, "GLOBAL_BATCH_GROUP_KEYS", None) or ())
                n_blocks = max(1, int(overrides.get("concurrency") or 1)) if group_keys else 1
                if n_blocks > 1:
                    ds = ds.repartition(num_blocks=n_blocks, keys=group_keys, shuffle=True)
                else:
                    ds = ds.repartition(num_blocks=1)
            elif target_num_rows_per_block is not None and int(target_num_rows_per_block) > 0:
                ds = ds.repartition(target_num_rows_per_block=int(target_num_rows_per_block))

            map_operator_class = node.operator_class
            map_batch_format = batch_format
            constructor_kwargs = node.operator_kwargs
            if batch_format == "pandas":
                # Ray's Arrow-backed pandas conversion can preserve unsafe
                # offsets for sliced structs with inferred null children.
                # Compact the valid Arrow batch before that conversion.
                map_operator_class = _make_arrow_pandas_operator_adapter(node.operator_class)
                stable_pandas_input = preserve_pandas_output
                # Once an operator reshapes the dataset into heterogeneous
                # object rows, keep those rows in pandas through every
                # downstream map stage. Re-enabling Arrow at embedding would
                # otherwise split optional bbox values into object and tensor
                # schemas that Ray cannot concatenate.
                preserve_pandas_output = preserve_pandas_output or _preserves_pandas_output(
                    node.operator_class, node.operator_kwargs
                )
                # The opting-in stage still needs the compacting Arrow adapter
                # on its input. Its downstream consumers must not ask Ray to
                # turn the intentionally preserved pandas block back into Arrow.
                map_batch_format = "pandas" if stable_pandas_input else "pyarrow"
                constructor_kwargs = {
                    "operator_class": node.operator_class,
                    "operator_kwargs": node.operator_kwargs,
                }

            ds = ds.map_batches(
                map_operator_class,
                batch_size=batch_size,
                batch_format=map_batch_format,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                fn_constructor_kwargs=constructor_kwargs,
                **overrides,
            )

        return ds
