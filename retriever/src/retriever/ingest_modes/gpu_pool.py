# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-GPU worker pool for distributing GPU pipeline stages across devices.

Workers are persistent processes that load models once at startup, then
process any number of DataFrame shards via queues.  Setting
``CUDA_VISIBLE_DEVICES`` per worker (before torch import) maps ``cuda:0``
inside each worker to a different physical GPU, so existing model code
works unmodified.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model config dataclasses — picklable recipes to recreate models in workers
# ---------------------------------------------------------------------------


@dataclass
class PageElementsModelConfig:
    """Config to recreate a page-elements model via ``define_model("page_element_v3")``."""

    config_name: str = "page_element_v3"

    def create(self) -> Any:
        from retriever.model.local import NemotronPageElementsV3

        return NemotronPageElementsV3()


@dataclass
class OCRModelConfig:
    """Config to recreate a NemotronOCRV1 model."""

    model_dir: str = ""

    def create(self) -> Any:
        from retriever.model.local import NemotronOCRV1

        return NemotronOCRV1(model_dir=self.model_dir)


@dataclass
class EmbeddingModelConfig:
    """Config to recreate an embedding model (VL or non-VL)."""

    device: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    normalize: bool = True
    max_length: int = 8192
    model_id: Optional[str] = None

    def create(self) -> Any:
        from retriever.model import is_vl_embed_model

        if is_vl_embed_model(self.model_id):
            from retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
                LlamaNemotronEmbedVL1BV2Embedder,
            )

            return LlamaNemotronEmbedVL1BV2Embedder(
                device=self.device,
                hf_cache_dir=self.hf_cache_dir,
                model_id=self.model_id,
            )

        from retriever.model.local.llama_nemotron_embed_1b_v2_embedder import (
            LlamaNemotronEmbed1BV2Embedder,
        )

        return LlamaNemotronEmbed1BV2Embedder(
            device=self.device,
            hf_cache_dir=self.hf_cache_dir,
            normalize=self.normalize,
            max_length=self.max_length,
            model_id=self.model_id,
        )


# ---------------------------------------------------------------------------
# GPUTaskDescriptor — picklable replacement for (func, kwargs_with_model)
# ---------------------------------------------------------------------------


@dataclass
class GPUTaskDescriptor:
    """Picklable description of a single GPU pipeline stage.

    Stores the function as ``module_name`` + ``func_name`` (resolved via
    import in the worker) and the kwargs *without* the ``model`` object.
    A :class:`ModelConfig` (if any) recreates the model in the worker.
    """

    module_name: str
    func_name: str
    kwargs: dict = field(default_factory=dict)
    model_config: Any = None  # One of the *ModelConfig dataclasses, or None

    def resolve(self) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Import the function and rebuild kwargs with a live model."""
        mod = importlib.import_module(self.module_name)
        func = getattr(mod, self.func_name)
        kwargs = dict(self.kwargs)
        if self.model_config is not None:
            kwargs["model"] = self.model_config.create()
        return func, kwargs


# ---------------------------------------------------------------------------
# Conversion: live (func, kwargs) → GPUTaskDescriptor
# ---------------------------------------------------------------------------


def _extract_model_config(func: Callable, kwargs: dict[str, Any]) -> Any:
    """Extract a picklable model config from live kwargs, or None."""
    from retriever.page_elements import detect_page_elements_v3
    from retriever.ocr.ocr import ocr_page_elements
    from .inprocess import embed_text_main_text_embed, explode_content_to_rows

    if func is detect_page_elements_v3:
        if kwargs.get("invoke_url"):
            return None  # Remote endpoint, no local model
        return PageElementsModelConfig()

    if func is ocr_page_elements:
        if kwargs.get("invoke_url"):
            return None  # Remote endpoint, no local model
        model = kwargs.get("model")
        model_dir = ""
        if model is not None and hasattr(model, "_model_dir"):
            model_dir = str(model._model_dir)
        return OCRModelConfig(model_dir=model_dir)

    if func is embed_text_main_text_embed:
        model = kwargs.get("model")
        if model is None:
            return None  # Remote endpoint, no local model
        return EmbeddingModelConfig(
            device=getattr(model, "device", None),
            hf_cache_dir=getattr(model, "hf_cache_dir", None),
            normalize=getattr(model, "normalize", True),
            max_length=getattr(model, "max_length", 8192),
            model_id=getattr(model, "model_id", None),
        )

    if func is explode_content_to_rows:
        return None  # CPU-only, no model

    return None


def gpu_tasks_to_descriptors(
    gpu_tasks: list[tuple[Callable[..., Any], dict[str, Any]]],
) -> list[GPUTaskDescriptor]:
    """Convert live ``[(func, kwargs_with_model), ...]`` into picklable descriptors."""
    descriptors: list[GPUTaskDescriptor] = []
    for func, kwargs in gpu_tasks:
        model_config = _extract_model_config(func, kwargs)
        # Strip model from kwargs (will be recreated from config in worker)
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        descriptors.append(
            GPUTaskDescriptor(
                module_name=func.__module__,
                func_name=func.__qualname__,
                kwargs=clean_kwargs,
                model_config=model_config,
            )
        )
    return descriptors


# ---------------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------------


def _gpu_worker_entry(
    gpu_index: int,
    physical_device_id: str,
    task_descriptors: list[GPUTaskDescriptor],
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    ready_event: multiprocessing.Event,
) -> None:
    """Worker process entry point.  Runs in a ``spawn``-ed process.

    1. Set CUDA_VISIBLE_DEVICES *before* torch import
    2. Resolve functions and create models from configs
    3. Signal ready
    4. Loop: receive (shard_id, DataFrame) → run pipeline → put result
    5. Exit on None sentinel
    """
    # Step 1: Isolate GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device_id

    try:
        # Step 2: Resolve all tasks (imports torch, loads models)
        resolved: list[tuple[Callable[..., Any], dict[str, Any]]] = []
        for desc in task_descriptors:
            func, kwargs = desc.resolve()
            resolved.append((func, kwargs))

        # Step 3: Signal ready
        ready_event.set()

        # Step 4: Processing loop
        while True:
            item = input_queue.get()
            if item is None:
                break  # Sentinel: shutdown

            shard_id, shard_df = item
            try:
                current = shard_df
                for func, kwargs in resolved:
                    current = func(current, **kwargs)
                output_queue.put((shard_id, current, None))
            except Exception as e:
                tb = traceback.format_exc()
                output_queue.put((shard_id, shard_df, f"{type(e).__name__}: {e}\n{tb}"))

    except Exception as e:
        # Startup failure — signal ready so the main process doesn't hang,
        # then put an error on the output queue.
        if not ready_event.is_set():
            ready_event.set()
        output_queue.put((-1, None, f"Worker {gpu_index} startup failed: {e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------------
# GPUWorkerPool
# ---------------------------------------------------------------------------


class GPUWorkerPool:
    """Pool of persistent GPU worker processes for multi-GPU inference.

    Usage::

        with GPUWorkerPool(["0", "1"], descriptors) as pool:
            result_df = pool.process(combined_df)
    """

    def __init__(
        self,
        gpu_devices: Sequence[str],
        task_descriptors: list[GPUTaskDescriptor],
        startup_timeout: float = 600.0,
    ) -> None:
        self._gpu_devices = list(gpu_devices)
        self._task_descriptors = task_descriptors
        self._startup_timeout = startup_timeout
        self._ctx = multiprocessing.get_context("spawn")
        self._workers: list[multiprocessing.Process] = []
        self._input_queues: list[multiprocessing.Queue] = []
        self._output_queue: Optional[multiprocessing.Queue] = None
        self._started = False
        self._shards_submitted = 0

    def start(self) -> None:
        """Spawn workers and wait for all models to load."""
        if self._started:
            return

        self._output_queue = self._ctx.Queue()
        ready_events: list[multiprocessing.Event] = []

        for idx, device_id in enumerate(self._gpu_devices):
            iq = self._ctx.Queue()
            evt = self._ctx.Event()
            p = self._ctx.Process(
                target=_gpu_worker_entry,
                args=(idx, device_id, self._task_descriptors, iq, self._output_queue, evt),
                daemon=True,
            )
            p.start()
            self._workers.append(p)
            self._input_queues.append(iq)
            ready_events.append(evt)

        # Wait for all workers to signal ready
        for idx, evt in enumerate(ready_events):
            if not evt.wait(timeout=self._startup_timeout):
                alive = self._workers[idx].is_alive()
                raise RuntimeError(
                    f"GPU worker {idx} (device {self._gpu_devices[idx]}) "
                    f"failed to become ready within {self._startup_timeout}s "
                    f"(alive={alive})"
                )
            # Check if worker died during startup
            if not self._workers[idx].is_alive():
                raise RuntimeError(f"GPU worker {idx} (device {self._gpu_devices[idx]}) " f"died during startup")

        self._started = True

    def submit(self, shard_id: int, df: pd.DataFrame) -> None:
        """Queue a shard for GPU processing (non-blocking).

        The shard is assigned to a worker via round-robin on *shard_id*.
        """
        if not self._started:
            raise RuntimeError("GPUWorkerPool.start() must be called before submit()")
        worker_idx = shard_id % len(self._workers)
        self._input_queues[worker_idx].put((shard_id, df))
        self._shards_submitted += 1

    def collect_all(
        self,
        show_progress: bool = False,
        on_shard_done: Optional[Callable[[int], None]] = None,
    ) -> pd.DataFrame:
        """Block until all submitted shards complete, return concatenated result.

        Parameters
        ----------
        show_progress : bool
            Show an internal tqdm bar (only used when *on_shard_done* is None).
        on_shard_done : callable, optional
            Called with the *shard_id* after each shard completes.  When
            provided, the internal tqdm bar is suppressed so the caller
            can drive its own progress.
        """
        if not self._started:
            raise RuntimeError("GPUWorkerPool.start() must be called before collect_all()")

        n = self._shards_submitted
        if n == 0:
            return pd.DataFrame()

        results: dict[int, pd.DataFrame] = {}
        errors: list[str] = []

        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:
            _tqdm = None

        progress = None
        if show_progress and on_shard_done is None and _tqdm is not None:
            progress = _tqdm(total=n, desc="GPU shards", unit="shard")

        collect_timeout = 3600.0  # 1 hour per shard max
        for _ in range(n):
            shard_id, result_df, error = self._output_queue.get(timeout=collect_timeout)
            if error is not None:
                errors.append(f"Shard {shard_id}: {error}")
                print(f"Warning: GPU shard {shard_id} failed: {error}")
                # On error, include the original shard data so we don't lose rows
                if result_df is not None and isinstance(result_df, pd.DataFrame):
                    results[shard_id] = result_df
            else:
                results[shard_id] = result_df
            if progress is not None:
                progress.update(1)
            if on_shard_done is not None:
                on_shard_done(shard_id)

        if progress is not None:
            progress.close()

        if errors:
            print(f"Warning: {len(errors)} GPU shard(s) had errors")

        # Reset counter for potential reuse
        self._shards_submitted = 0

        if not results:
            return pd.DataFrame()

        # Reassemble in original shard order
        ordered = [results[sid] for sid in sorted(results.keys())]
        return pd.concat(ordered, ignore_index=True)

    def process(
        self,
        df: pd.DataFrame,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Shard *df* across workers, collect results, return concatenated DataFrame.

        This is a convenience wrapper around :meth:`submit` + :meth:`collect_all`
        for callers that have a single combined DataFrame.
        """
        if not self._started:
            raise RuntimeError("GPUWorkerPool.start() must be called before process()")

        n_workers = len(self._workers)
        n_rows = len(df)

        if n_rows == 0:
            return df

        # Shard by rows and submit
        indices = np.array_split(range(n_rows), n_workers)
        for shard_id, idx_chunk in enumerate(indices):
            if len(idx_chunk) == 0:
                continue
            shard_df = df.iloc[idx_chunk].reset_index(drop=True)
            self.submit(shard_id, shard_df)

        result = self.collect_all(show_progress=show_progress)
        return result if not result.empty else df

    def shutdown(self) -> None:
        """Send shutdown sentinels, join workers, terminate stragglers."""
        for iq in self._input_queues:
            try:
                iq.put(None)
            except Exception:
                pass

        for p in self._workers:
            p.join(timeout=30.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)

        self._workers.clear()
        self._input_queues.clear()
        self._started = False

    def __enter__(self) -> "GPUWorkerPool":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()
