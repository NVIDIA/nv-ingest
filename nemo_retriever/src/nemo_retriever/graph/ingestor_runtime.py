# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for graph-backed ingestor implementations."""

from __future__ import annotations

import logging
from functools import partial
from typing import cast
from typing import Any

from nemo_retriever.operators.extract.caption.caption import CaptionActor
from nemo_retriever.operators.extract.audio.asr_actor import ASRActor
from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor
from nemo_retriever.operators.dedup import dedup_images
from nemo_retriever.graph import Graph, StoreOperator, UDFOperator, WebhookNotifyOperator
from nemo_retriever.common.modality.content_transforms import (
    _CONTENT_COLUMNS,
    collapse_content_to_page_rows,
    explode_content_to_rows,
)
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractOperator
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.extract.video.audio_visual_fuser import AudioVisualFuser
from nemo_retriever.operators.extract.video.ocr_actor import VideoFrameOCRActor
from nemo_retriever.operators.extract.video.text_dedup import VideoFrameTextDedup
from nemo_retriever.operators.extract.video.split import VideoSplitActor
from nemo_retriever.operators.extract.ocr.ocr import resolve_ocr_archetype
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.operators.extract.table.table_detection import TableStructureActor
from nemo_retriever.operators.extract.pdf.extract import PDFExtractionActor, build_pdf_extraction_kwargs
from nemo_retriever.operators.extract.pdf.split import PDFSplitActor
from nemo_retriever.common.params import TextChunkParams, VdbUploadParams, resolve_split_params
from nemo_retriever.operators.vdb import IngestVdbOperator
from nemo_retriever.operators.extract.txt.ray_data import TextChunkActor, TxtSplitActor
from nemo_retriever.common.modality.convert.to_pdf import DocToPdfConversionActor
from nemo_retriever.ingestor.plans import IngestExecutionPlan, dedup_params_enabled
from nemo_retriever.common.ray_resource_hueristics import (
    ClusterResources,
    resolve_requested_plan,
)

logger = logging.getLogger(__name__)

DEFAULT_STORE_WORKERS = 4
DEFAULT_STORE_CPUS_PER_ACTOR = 0.1


def _batch_tuning(params: Any) -> Any:
    return getattr(params, "batch_tuning", None)


def _local_embed_requested(params: Any) -> bool:
    """Return whether embedding configuration explicitly requires a local actor."""
    if params is None:
        return False
    endpoint = getattr(params, "embed_invoke_url", None) or getattr(params, "embedding_endpoint", None)
    if str(endpoint or "").strip():
        return False
    fields_set = getattr(params, "model_fields_set", set())
    if "local_ingest_embed_backend" in fields_set:
        return True
    gpu_embed = getattr(_batch_tuning(params), "gpu_embed", None)
    return gpu_embed is not None and float(gpu_embed) > 0


def _image_embedding_requires_page_image(params: Any | None) -> bool:
    """Return whether embedding consumes a rendered image for each page row."""
    return getattr(params, "embed_granularity", None) == "page" and getattr(params, "embed_modality", None) in {
        "image",
        "text_image",
    }


def default_concurrency_node_names(
    extract_params: Any | None,
    embed_params: Any | None,
    store_params: Any | None,
    caption_params: Any | None,
) -> set[str]:
    """Return pools whose concurrency came from an unspecified default."""
    names: set[str] = set()
    names.update({MultiTypeExtractOperator.__name__, TxtSplitActor.__name__})
    extract_tuning = _batch_tuning(extract_params)
    if extract_params is not None:
        worker_fields = {
            resolve_ocr_archetype(extract_params).__name__: "ocr_workers",
            PageElementDetectionActor.__name__: "page_elements_workers",
            TableStructureActor.__name__: "table_structure_workers",
            PDFExtractionActor.__name__: "pdf_extract_workers",
            NemotronParseActor.__name__: "nemotron_parse_workers",
        }
        names.update(
            name for name, field in worker_fields.items() if not _positive(getattr(extract_tuning, field, None))
        )
    embed_tuning = _batch_tuning(embed_params)
    if (
        embed_params is not None
        and not _positive(getattr(embed_tuning, "embed_workers", None))
        and _elastic_embed_concurrency(embed_tuning) is None
    ):
        names.add(_BatchEmbedActor.__name__)
    store_tuning = _batch_tuning(store_params)
    if store_params is not None and not _positive(getattr(store_tuning, "store_workers", None)):
        names.add(StoreOperator.__name__)
    if caption_params is not None:
        names.add(CaptionActor.__name__)
    return names


def _positive(value: Any) -> Any:
    return value if value not in (None, 0, 0.0, "", False) else None


def _elastic_embed_concurrency(tuning: Any) -> tuple[int, int, int] | None:
    if tuning is None:
        return None
    minimum = getattr(tuning, "embed_workers_min", None)
    initial = getattr(tuning, "embed_workers_initial", None)
    maximum = getattr(tuning, "embed_workers_max", None)
    if minimum is None or initial is None or maximum is None:
        return None
    # Ray actor pools encode elasticity as (minimum, maximum, initial).
    return (int(minimum), int(maximum), int(initial))


def _nim_remote_http_kwargs(extract_params: Any) -> dict[str, int]:
    """Forward ExtractParams.remote_retry into stage kwargs for higher HTTP parallelism."""
    rr = getattr(extract_params, "remote_retry", None)
    if rr is None:
        return {}
    return {
        "remote_max_pool_workers": int(rr.remote_max_pool_workers),
        "remote_max_retries": int(rr.remote_max_retries),
        "remote_max_429_retries": int(rr.remote_max_429_retries),
    }


def batch_tuning_to_node_overrides(
    extract_params: Any | None,
    embed_params: Any | None,
    cluster_resources: ClusterResources | None = None,
    allow_no_gpu: bool | None = None,
    caption_params: Any | None = None,
    caption_gpus_per_actor: float | None = None,
    video_frame_params: Any | None = None,
    store_params: Any | None = None,
    extraction_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Translate BatchTuningParams from stage params into RayDataExecutor node_overrides.

    Explicit (non-zero) values from BatchTuningParams always win.  When a field
    is absent or zero, the heuristic default from ``resolve_requested_plan`` is
    used instead — provided ``cluster_resources`` is supplied (i.e. Ray is
    already initialised).  Without ``cluster_resources`` only explicit values
    are emitted, matching the previous behaviour.

    Final actor-pool sizing is reconciled after graph resolution by the executor
    resource preflight.
    """
    auto_allow_no_gpu = bool(cluster_resources is not None and cluster_resources.total_gpu_count() == 0)
    effective_allow_no_gpu = allow_no_gpu if allow_no_gpu is not None else auto_allow_no_gpu
    plan = (
        resolve_requested_plan(
            cluster_resources=cluster_resources,
            allow_no_gpu=effective_allow_no_gpu,
            caption_enabled=caption_params is not None,
            override_caption_gpus_per_actor=caption_gpus_per_actor,
        )
        if cluster_resources is not None
        else None
    )

    overrides: dict[str, dict[str, Any]] = {}

    def _resolve(explicit: Any, fallback: Any = None) -> Any:
        v = _positive(explicit)
        if v is None and fallback is not None:
            v = fallback
        return v

    def _set(node_name: str, key: str, explicit: Any, fallback: Any = None) -> None:
        v = _resolve(explicit, fallback)
        if v is not None:
            overrides.setdefault(node_name, {})[key] = v

    def _set_gpu(node_name: str, explicit: Any, fallback: Any = None) -> None:
        """Like _set for num_gpus, but treats 0.0 as a valid explicit value."""
        v = explicit if explicit is not None else fallback
        if v is not None:
            overrides.setdefault(node_name, {})["num_gpus"] = v

    def _force_cpu_only(node_name: str) -> None:
        overrides.setdefault(node_name, {})["num_gpus"] = 0.0

    embed_tuning = _batch_tuning(embed_params)
    embed_concurrency: int | tuple[int, int, int] = 0
    embed_cpus: float = 1.0
    local_caption_concurrency: int | None = None
    local_caption_gpus_per_actor: float | None = None
    if caption_params is not None and cluster_resources is not None:
        caption_invoke_url = _positive(getattr(caption_params, "endpoint_url", None))
        if not effective_allow_no_gpu and not caption_invoke_url:
            available_gpus = max(1, int(cluster_resources.available_gpu_count()))
            local_caption_gpus_per_actor = (
                _resolve(caption_gpus_per_actor, plan.caption_gpus_per_actor if plan else None) or 1.0
            )
            # Local captioning is the visual-workload bottleneck. On DGX-class
            # hosts, use the GPU pool for caption actors and leave one GPU's
            # budget for downstream embedding.
            local_caption_concurrency = 1 if available_gpus <= 1 else max(1, available_gpus - 1)

    if embed_params is not None:
        embed_invoke_url = _positive(getattr(embed_params, "embed_invoke_url", None))
        explicit_bs = getattr(embed_tuning, "embed_batch_size", None) if embed_tuning is not None else None
        embed_bs = _positive(explicit_bs) or (plan.embed_batch_size if plan else None)
        _set(_BatchEmbedActor.__name__, "batch_size", embed_bs)
        if embed_bs:
            overrides.setdefault(_BatchEmbedActor.__name__, {})["target_num_rows_per_block"] = embed_bs
        explicit_embed_workers = getattr(embed_tuning, "embed_workers", None) if embed_tuning is not None else None
        elastic_embed_concurrency = _elastic_embed_concurrency(embed_tuning)
        embed_workers_fallback = plan.embed_initial_actors if plan else None
        if (
            local_caption_concurrency is not None
            and local_caption_gpus_per_actor is not None
            and _positive(explicit_embed_workers) is None
            and elastic_embed_concurrency is None
            and cluster_resources is not None
            and plan is not None
        ):
            caption_gpu_budget = local_caption_concurrency * local_caption_gpus_per_actor
            remaining_gpu_budget = max(0.0, float(cluster_resources.available_gpu_count()) - caption_gpu_budget)
            if remaining_gpu_budget > 0 and plan.embed_gpus_per_actor > 0:
                embed_workers_fallback = max(1, int(remaining_gpu_budget // plan.embed_gpus_per_actor))
            else:
                embed_workers_fallback = 1
        embed_concurrency = (
            elastic_embed_concurrency
            or _resolve(
                explicit_embed_workers,
                embed_workers_fallback,
            )
            or 0
        )
        _set(_BatchEmbedActor.__name__, "concurrency", embed_concurrency or None)
        embed_cpus = (
            _resolve(
                getattr(embed_tuning, "embed_cpus_per_actor", None) if embed_tuning is not None else None,
            )
            or 1.0
        )
        _set(_BatchEmbedActor.__name__, "num_cpus", embed_cpus if embed_cpus != 1.0 else None)
        if effective_allow_no_gpu:
            _force_cpu_only(_BatchEmbedActor.__name__)
        elif not embed_invoke_url:
            _set_gpu(
                _BatchEmbedActor.__name__,
                getattr(embed_tuning, "gpu_embed", None) if embed_tuning is not None else None,
                plan.embed_gpus_per_actor if plan else None,
            )

    if caption_params is not None:
        caption_invoke_url = _positive(getattr(caption_params, "endpoint_url", None))
        if effective_allow_no_gpu:
            _force_cpu_only(CaptionActor.__name__)
        elif not caption_invoke_url:
            if local_caption_concurrency is not None:
                overrides.setdefault(CaptionActor.__name__, {})["concurrency"] = local_caption_concurrency
            _set_gpu(
                CaptionActor.__name__,
                caption_gpus_per_actor,
                plan.caption_gpus_per_actor if plan else None,
            )

    extract_tuning = _batch_tuning(extract_params)
    ocr_concurrency: int = 0
    ocr_cpus: float = 1.0
    page_elements_concurrency: int = 0
    page_elements_cpus: float = 1.0
    if extract_params is not None:
        ocr_invoke_url = _positive(getattr(extract_params, "ocr_invoke_url", None))
        page_elements_invoke_url = _positive(getattr(extract_params, "page_elements_invoke_url", None))
        ocr_actor_name = resolve_ocr_archetype(extract_params).__name__

        ocr_bs = _positive(
            getattr(extract_tuning, "ocr_inference_batch_size", None) if extract_tuning is not None else None
        ) or (plan.ocr_batch_size if plan else None)
        _set(ocr_actor_name, "batch_size", ocr_bs)
        ocr_concurrency = (
            _resolve(
                getattr(extract_tuning, "ocr_workers", None) if extract_tuning is not None else None,
                plan.ocr_initial_actors if plan else None,
            )
            or 0
        )
        _set(ocr_actor_name, "concurrency", ocr_concurrency or None)
        ocr_cpus = (
            _resolve(
                getattr(extract_tuning, "ocr_cpus_per_actor", None) if extract_tuning is not None else None,
            )
            or 1.0
        )
        _set(ocr_actor_name, "num_cpus", ocr_cpus if ocr_cpus != 1.0 else None)
        if effective_allow_no_gpu:
            _force_cpu_only(ocr_actor_name)
        elif not ocr_invoke_url:
            _set_gpu(
                ocr_actor_name,
                getattr(extract_tuning, "gpu_ocr", None) if extract_tuning is not None else None,
                plan.ocr_gpus_per_actor if plan else None,
            )

        pe_bs = _positive(
            getattr(extract_tuning, "page_elements_batch_size", None) if extract_tuning is not None else None
        ) or (plan.page_elements_batch_size if plan else None)
        _set(PageElementDetectionActor.__name__, "batch_size", pe_bs)
        if pe_bs:
            overrides.setdefault(PageElementDetectionActor.__name__, {})["target_num_rows_per_block"] = pe_bs
        page_elements_concurrency = (
            _resolve(
                getattr(extract_tuning, "page_elements_workers", None) if extract_tuning is not None else None,
                plan.page_elements_initial_actors if plan else None,
            )
            or 0
        )
        _set(PageElementDetectionActor.__name__, "concurrency", page_elements_concurrency or None)
        page_elements_cpus = (
            _resolve(
                getattr(extract_tuning, "page_elements_cpus_per_actor", None) if extract_tuning is not None else None,
            )
            or 1.0
        )
        _set(PageElementDetectionActor.__name__, "num_cpus", page_elements_cpus if page_elements_cpus != 1.0 else None)
        if effective_allow_no_gpu:
            _force_cpu_only(PageElementDetectionActor.__name__)
        elif not page_elements_invoke_url:
            _set_gpu(
                PageElementDetectionActor.__name__,
                getattr(extract_tuning, "gpu_page_elements", None) if extract_tuning is not None else None,
                plan.page_elements_gpus_per_actor if plan else None,
            )

        # --- Table Structure ---
        table_structure_invoke_url = _positive(getattr(extract_params, "table_structure_invoke_url", None))
        ts_bs = _positive(
            getattr(extract_tuning, "table_structure_batch_size", None) if extract_tuning is not None else None
        ) or (plan.table_structure_batch_size if plan else None)
        _set(TableStructureActor.__name__, "batch_size", ts_bs)
        if ts_bs:
            overrides.setdefault(TableStructureActor.__name__, {})["target_num_rows_per_block"] = ts_bs
        ts_concurrency = _resolve(
            getattr(extract_tuning, "table_structure_workers", None) if extract_tuning is not None else None,
            plan.table_structure_initial_actors if plan else None,
        ) or (2 if table_structure_invoke_url else 0)
        _set(TableStructureActor.__name__, "concurrency", ts_concurrency or None)
        ts_cpus = (
            _resolve(
                getattr(extract_tuning, "table_structure_cpus_per_actor", None) if extract_tuning is not None else None,
            )
            or 1.0
        )
        _set(TableStructureActor.__name__, "num_cpus", ts_cpus)
        if effective_allow_no_gpu:
            _force_cpu_only(TableStructureActor.__name__)
        elif not table_structure_invoke_url:
            _set_gpu(
                TableStructureActor.__name__,
                getattr(extract_tuning, "gpu_table_structure", None) if extract_tuning is not None else None,
                plan.table_structure_gpus_per_actor if plan else None,
            )

        np_bs = _positive(
            getattr(extract_tuning, "nemotron_parse_batch_size", None) if extract_tuning is not None else None
        ) or (plan.nemotron_parse_batch_size if plan else None)
        _set(NemotronParseActor.__name__, "batch_size", np_bs)
        _set(
            NemotronParseActor.__name__,
            "concurrency",
            getattr(extract_tuning, "nemotron_parse_workers", None) if extract_tuning is not None else None,
            plan.nemotron_parse_initial_actors if plan else None,
        )
        if effective_allow_no_gpu:
            _force_cpu_only(NemotronParseActor.__name__)
        else:
            _set_gpu(
                NemotronParseActor.__name__,
                getattr(extract_tuning, "gpu_nemotron_parse", None) if extract_tuning is not None else None,
                plan.nemotron_parse_gpus_per_actor if plan else None,
            )

        pdf_bs = _positive(
            getattr(extract_tuning, "pdf_extract_batch_size", None) if extract_tuning is not None else None
        ) or (plan.pdf_extract_batch_size if plan else None)
        pdf_extract_cpus = (
            _resolve(
                getattr(extract_tuning, "pdf_extract_num_cpus", None) if extract_tuning is not None else None,
                plan.pdf_extract_cpus_per_task if plan else None,
            )
            or 1.0
        )
        pdf_extract_tasks = _resolve(
            getattr(extract_tuning, "pdf_extract_workers", None) if extract_tuning is not None else None,
            plan.pdf_extract_tasks if plan else None,
        )

        _set(PDFExtractionActor.__name__, "batch_size", pdf_bs)
        _set(PDFExtractionActor.__name__, "concurrency", pdf_extract_tasks)
        _set(PDFExtractionActor.__name__, "num_cpus", pdf_extract_cpus if pdf_extract_cpus != 1.0 else None)

    if extraction_mode == "text" and cluster_resources is not None:
        available_cpus = cluster_resources.available_cpu_count()
        if available_cpus > 0:
            _set(TxtSplitActor.__name__, "concurrency", min(available_cpus, 8))
            _set(TxtSplitActor.__name__, "num_cpus", 1)
            _force_cpu_only(TxtSplitActor.__name__)

    # VideoSplitActor: one ffmpeg subprocess per input video, ~1-2 CPU cores
    # per actor during decode. Default Ray Data concurrency=1 serialises every
    # video, making this stage the wall-clock bottleneck on multi-video inputs.
    # Scale with available CPUs (one actor per ~4 cores leaves headroom for
    # downstream ASR/OCR/fuse stages); cap at 8 to avoid disk-I/O contention
    # on slower storage. With fewer input videos than the cap, Ray Data only
    # spawns as many actors as there are blocks — so an oversized cap is safe.
    if video_frame_params is not None and getattr(video_frame_params, "enabled", True):
        cpus = cluster_resources.total_cpu_count() if cluster_resources is not None else 0
        if cpus > 0:
            _set(VideoSplitActor.__name__, "concurrency", max(1, min(cpus // 4, 8)))

    if store_params is not None:
        store_tuning = _batch_tuning(store_params)
        store_workers = _positive(getattr(store_tuning, "store_workers", None) if store_tuning is not None else None)
        store_workers = int(store_workers or DEFAULT_STORE_WORKERS)
        store_override = overrides.setdefault(StoreOperator.__name__, {})
        # Ray actor pool tuple is (min, max, initial); keep store lazy at startup.
        store_override["concurrency"] = (1, store_workers, 1) if store_workers > 1 else 1
        store_override["num_cpus"] = DEFAULT_STORE_CPUS_PER_ACTOR

    return overrides


def _resolve_execution_inputs(
    *,
    execution_plan: IngestExecutionPlan | None,
    extraction_mode: str,
    extract_params: Any | None,
    text_params: Any | None,
    html_params: Any | None,
    audio_chunk_params: Any | None,
    asr_params: Any | None,
    dedup_params: Any | None,
    split_config: dict[str, Any] | None,
    caption_params: Any | None,
    store_params: Any | None,
    embed_params: Any | None,
    webhook_params: Any | None = None,
    stage_order: tuple[str, ...],
) -> tuple[
    str,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    tuple[str, ...],
]:
    """Resolve legacy builder args or a shared execution plan into one input tuple."""

    if execution_plan is None:
        return (
            extraction_mode,
            extract_params,
            text_params,
            html_params,
            audio_chunk_params,
            asr_params,
            dedup_params,
            split_config,
            caption_params,
            store_params,
            embed_params,
            webhook_params,
            stage_order,
        )

    stage_map = {stage.name: stage.params for stage in execution_plan.stages}
    return (
        execution_plan.extraction_mode,
        execution_plan.extract_params,
        execution_plan.text_params,
        execution_plan.html_params,
        execution_plan.audio_chunk_params,
        execution_plan.asr_params,
        stage_map.get("dedup"),
        execution_plan.split_config,
        stage_map.get("caption"),
        stage_map.get("store"),
        stage_map.get("embed"),
        stage_map.get("webhook"),
        tuple(stage.name for stage in execution_plan.stages),
    )


def _should_build_audio_graph(
    *,
    extraction_mode: str | None,
    extract_params: Any | None,
    asr_params: Any | None,
) -> bool:
    """True iff the audio-only ``MediaChunkActor → ASRActor`` graph applies.

    The audio-only shortcut graph is dedicated to **audio inputs**: it
    constructs :class:`MediaChunkActor` unconditionally and has no
    dispatch path for PDF / image / text / HTML uploads. Routing a
    non-audio request through this branch is the bug that surfaces as
    ``RuntimeError: MediaChunkActor requires media dependencies; missing:
    ffmpeg, ffprobe`` for PDF ingestion.

    Returning ``True`` therefore requires an explicit audio signal:

    * ``extraction_mode == "audio"`` — the caller (or the upstream
      auto-detector in :meth:`GraphIngestor._resolve_effective_extraction_inputs`)
      classified the inputs as audio.
    * ``extract_params.method == "audio"`` — the legacy params-driven
      opt-in used by tests and a few direct callers.

    The mere presence of ``asr_params`` is **not** a sufficient signal:
    in service mode ``asr_params`` is auto-derived from the cluster's
    ``audio_grpc_endpoint`` and would otherwise force every PDF upload
    through the audio-only graph.
    """
    if (extraction_mode or "").strip().lower() == "audio":
        return True
    method = str(getattr(extract_params, "method", "") or "").strip().lower()
    if method == "audio":
        return True
    _ = asr_params  # kept for backwards-compatible kw signature
    return False


def _maybe_append_chunk_actor(graph: Graph, split_config: dict[str, Any], key: str) -> Graph:
    """Append a TextChunkActor to *graph* when split_config[key] requests chunking.

    Skips on both ``None`` (absent) and ``False`` (explicit opt-out).
    """
    params = split_config.get(key)
    if isinstance(params, TextChunkParams):
        graph = graph >> TextChunkActor(params)
    return graph


def _append_ordered_transform_stages(
    graph: Graph,
    *,
    dedup_params: Any | None,
    caption_params: Any | None,
    store_params: Any | None,
    embed_params: Any | None,
    vdb_upload_params: VdbUploadParams | None = None,
    webhook_params: Any | None = None,
    stage_order: tuple[str, ...],
    supports_dedup: bool,
    reshape_content_before_embed: bool,
) -> Graph:
    """Append post-extraction transform stages in the exact recorded plan order."""

    pending_stages = [
        stage
        for stage in stage_order
        if stage in {"dedup", "caption", "store", "embed"} and (supports_dedup or stage != "dedup")
    ]
    if not pending_stages:
        if supports_dedup and dedup_params_enabled(dedup_params):
            pending_stages.append("dedup")
        if caption_params is not None:
            pending_stages.append("caption")
        if store_params is not None:
            pending_stages.append("store")
        if embed_params is not None:
            pending_stages.append("embed")

    for stage_name in pending_stages:
        if stage_name == "store" and store_params is not None:
            graph = graph >> StoreOperator(params=store_params)
        elif stage_name == "dedup" and supports_dedup and dedup_params_enabled(dedup_params):
            dedup_kwargs = cast(dict[str, Any], dedup_params.model_dump(mode="python"))
            graph = graph >> UDFOperator(partial(dedup_images, **dedup_kwargs), name="DedupImages")
        elif stage_name == "caption" and caption_params is not None:
            graph = graph >> CaptionActor(caption_params)
        elif stage_name == "embed" and embed_params is not None:
            if reshape_content_before_embed:
                content_columns = (_CONTENT_COLUMNS + ("images",)) if caption_params is not None else _CONTENT_COLUMNS
                if embed_params.embed_granularity == "page":
                    graph = graph >> UDFOperator(
                        partial(
                            collapse_content_to_page_rows,
                            modality=embed_params.embed_modality,
                            content_columns=content_columns,
                        ),
                        name="CollapseContentToPageRows",
                        preserve_pandas_output=True,
                    )
                else:
                    graph = graph >> UDFOperator(
                        partial(
                            explode_content_to_rows,
                            modality=embed_params.embed_modality,
                            text_elements_modality=embed_params.text_elements_modality or embed_params.embed_modality,
                            structured_elements_modality=embed_params.structured_elements_modality
                            or embed_params.embed_modality,
                            content_columns=content_columns,
                        ),
                        name="ExplodeContentToRows",
                        preserve_pandas_output=True,
                    )
            graph = graph >> _BatchEmbedActor(
                params=embed_params,
                force_local=_local_embed_requested(embed_params),
            )

    if vdb_upload_params is not None:
        graph = graph >> IngestVdbOperator(
            vdb_op=vdb_upload_params.vdb_op,
            vdb_kwargs=vdb_upload_params.to_ingest_operator_kwargs(),
        )

    if webhook_params is not None and getattr(webhook_params, "endpoint_url", None):
        graph = graph >> WebhookNotifyOperator(params=webhook_params)

    return graph


def build_post_extract_graph(
    *,
    dedup_params: Any | None = None,
    embed_params: Any | None = None,
    caption_params: Any | None = None,
    store_params: Any | None = None,
    vdb_upload_params: VdbUploadParams | None = None,
    webhook_params: Any | None = None,
    stage_order: tuple[str, ...] = (),
    reshape_content_before_embed: bool = True,
) -> Graph:
    """Build only the common stages that run after extraction branch union."""

    return _append_ordered_transform_stages(
        Graph(),
        dedup_params=dedup_params,
        caption_params=caption_params,
        store_params=store_params,
        embed_params=embed_params,
        vdb_upload_params=vdb_upload_params,
        webhook_params=webhook_params,
        stage_order=stage_order,
        supports_dedup=True,
        reshape_content_before_embed=reshape_content_before_embed,
    )


def build_graph(
    *,
    execution_plan: IngestExecutionPlan | None = None,
    extraction_mode: str = "pdf",
    extract_params: Any | None = None,
    text_params: Any | None = None,
    html_params: Any | None = None,
    audio_chunk_params: Any | None = None,
    asr_params: Any | None = None,
    dedup_params: Any | None = None,
    embed_params: Any | None = None,
    split_config: dict[str, Any] | None = None,
    caption_params: Any | None = None,
    store_params: Any | None = None,
    vdb_upload_params: VdbUploadParams | None = None,
    webhook_params: Any | None = None,
    video_frame_params: Any | None = None,
    video_text_dedup_params: Any | None = None,
    av_fuse_params: Any | None = None,
    stage_order: tuple[str, ...] = (),
) -> Graph:
    """Build a batch graph from explicit params or a shared execution plan."""

    (
        extraction_mode,
        extract_params,
        text_params,
        html_params,
        audio_chunk_params,
        asr_params,
        dedup_params,
        split_config,
        caption_params,
        store_params,
        embed_params,
        webhook_params,
        stage_order,
    ) = _resolve_execution_inputs(
        execution_plan=execution_plan,
        extraction_mode=extraction_mode,
        extract_params=extract_params,
        text_params=text_params,
        html_params=html_params,
        audio_chunk_params=audio_chunk_params,
        asr_params=asr_params,
        dedup_params=dedup_params,
        split_config=split_config,
        caption_params=caption_params,
        store_params=store_params,
        embed_params=embed_params,
        webhook_params=webhook_params,
        stage_order=stage_order,
    )

    sink_vdb: VdbUploadParams | None = None
    if execution_plan is not None:
        for sink in execution_plan.sinks:
            if sink.name == "vdb_upload":
                sink_vdb = sink.params
                break
    effective_vdb_upload_params = vdb_upload_params if vdb_upload_params is not None else sink_vdb

    # GraphIngestor pre-resolves split_config; tests and other direct callers
    # may omit it, in which case fill in defaults consistently with the
    # ingestor surface.
    if split_config is None:
        split_config = resolve_split_params(None)

    # Page-level image modalities require a full page raster regardless of
    # whether PDF extraction runs through the dedicated or auto-dispatch graph.
    if (
        extract_params is not None
        and _image_embedding_requires_page_image(embed_params)
        and not extract_params.extract_page_as_image
    ):
        extract_params = extract_params.model_copy(update={"extract_page_as_image": True})

    # Video ingestion uses a dedicated chain so each stage (fan-out, ASR,
    # frame OCR, scene fusion) shows up as its own Ray Data MapBatches op.
    # The audio-only shortcut below would otherwise short-circuit to a
    # single ``MediaChunkActor → ASRActor`` graph and we'd lose frame OCR.
    has_video_branch = video_frame_params is not None
    if has_video_branch:
        # Each stream's actor is appended only when that stream is enabled.
        # This skips the eager Parakeet load when audio is off and avoids
        # empty Ray Data MapBatches stages cluttering the dashboard.
        audio_enabled = audio_chunk_params is not None and getattr(audio_chunk_params, "enabled", True)
        audio_only = audio_chunk_params is not None and getattr(audio_chunk_params, "audio_only", False)
        frames_enabled = getattr(video_frame_params, "enabled", True) and not audio_only
        text_dedup_enabled = (
            frames_enabled and video_text_dedup_params is not None and getattr(video_text_dedup_params, "enabled", True)
        )
        fuse_enabled = (
            audio_enabled and frames_enabled and av_fuse_params is not None and getattr(av_fuse_params, "enabled", True)
        )

        graph = Graph() >> VideoSplitActor(
            audio_chunk_params=audio_chunk_params,
            video_frame_params=video_frame_params,
        )
        if audio_enabled:
            graph = graph >> ASRActor(params=asr_params)
        if frames_enabled:
            graph = graph >> VideoFrameOCRActor(
                ocr_invoke_url=getattr(extract_params, "ocr_invoke_url", None),
                ocr_version=getattr(extract_params, "ocr_version", "v2"),
                ocr_lang=getattr(extract_params, "ocr_lang", None),
                api_key=getattr(extract_params, "ocr_api_key", None) or getattr(extract_params, "api_key", None),
                inference_batch_size=int(getattr(extract_params, "inference_batch_size", None) or 8),
                request_timeout_s=float(
                    getattr(extract_params, "ocr_request_timeout_s", None)
                    or getattr(extract_params, "request_timeout_s", None)
                    or 120.0
                ),
            )
        if text_dedup_enabled:
            graph = graph >> VideoFrameTextDedup(params=video_text_dedup_params)
        if fuse_enabled:
            graph = graph >> AudioVisualFuser(params=av_fuse_params)
        graph = _maybe_append_chunk_actor(graph, split_config, "video")
    elif _should_build_audio_graph(
        extraction_mode=extraction_mode,
        extract_params=extract_params,
        asr_params=asr_params,
    ):
        graph = Graph() >> MediaChunkActor(params=audio_chunk_params) >> ASRActor(params=asr_params)
        graph = _maybe_append_chunk_actor(graph, split_config, "audio")
    elif extraction_mode == "text":
        configured_text_params = split_config.get("text")
        effective_text_params = (
            configured_text_params if isinstance(configured_text_params, TextChunkParams) else text_params
        )
        graph = Graph() >> TxtSplitActor(params=effective_text_params)
    elif extraction_mode in {"html", "audio", "image", "auto"}:
        graph = Graph() >> MultiTypeExtractOperator(
            extraction_mode=extraction_mode,
            extract_params=extract_params,
            text_params=text_params,
            html_params=html_params,
            audio_chunk_params=audio_chunk_params,
            asr_params=asr_params,
            caption_params=caption_params,
            video_frame_params=video_frame_params,
            video_text_dedup_params=video_text_dedup_params,
            av_fuse_params=av_fuse_params,
            split_config=split_config,
        )
    else:
        graph = Graph()
        graph = graph >> DocToPdfConversionActor() >> PDFSplitActor()

        tuning = _batch_tuning(extract_params)
        parse_mode = extract_params.method == "nemotron_parse" or (
            tuning is not None
            and (_positive(getattr(tuning, "nemotron_parse_workers", None)) is not None)
            and (_positive(getattr(tuning, "gpu_nemotron_parse", None)) is not None)
            and (_positive(getattr(tuning, "nemotron_parse_batch_size", None)) is not None)
        )

        extract_kwargs = build_pdf_extraction_kwargs(extract_params)

        if parse_mode:
            # PDF extraction renders pages to images required by Nemotron Parse.
            extract_kwargs["extract_page_as_image"] = True
            graph = graph >> PDFExtractionActor(**extract_kwargs)

            parse_kwargs: dict[str, Any] = {
                "extract_text": extract_params.extract_text,
                "extract_tables": extract_params.extract_tables,
                "extract_charts": extract_params.extract_charts,
                "extract_infographics": extract_params.extract_infographics,
            }
            if extract_params.nemotron_parse_invoke_url:
                parse_kwargs["nemotron_parse_invoke_url"] = extract_params.nemotron_parse_invoke_url
            elif extract_params.invoke_url:
                parse_kwargs["invoke_url"] = extract_params.invoke_url
            if extract_params.api_key:
                parse_kwargs["api_key"] = extract_params.api_key
            if extract_params.nemotron_parse_model:
                parse_kwargs["nemotron_parse_model"] = extract_params.nemotron_parse_model
            parse_kwargs.update(_nim_remote_http_kwargs(extract_params))
            graph = graph >> NemotronParseActor(**parse_kwargs)
        else:
            detect_kwargs: dict[str, Any] = {}
            if extract_params.page_elements_invoke_url:
                detect_kwargs["page_elements_invoke_url"] = extract_params.page_elements_invoke_url
            if extract_params.api_key:
                detect_kwargs["api_key"] = extract_params.api_key
            if extract_params.inference_batch_size:
                detect_kwargs["inference_batch_size"] = int(extract_params.inference_batch_size)

            ocr_kwargs: dict[str, Any] = {}
            if extract_params.method in ("pdfium_hybrid", "ocr") and extract_params.extract_text:
                ocr_kwargs["extract_text"] = True
            if extract_params.extract_tables:
                ocr_kwargs["extract_tables"] = True
            if extract_params.extract_charts:
                ocr_kwargs["extract_charts"] = True
            if extract_params.extract_infographics:
                ocr_kwargs["extract_infographics"] = True
            ocr_kwargs["use_table_structure"] = extract_params.use_table_structure
            ocr_kwargs["ocr_version"] = getattr(extract_params, "ocr_version", "v2")
            if getattr(extract_params, "ocr_lang", None) is not None:
                ocr_kwargs["ocr_lang"] = extract_params.ocr_lang
            if extract_params.ocr_invoke_url:
                ocr_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                ocr_kwargs["api_key"] = extract_params.api_key
            ocr_batch_size = _positive(
                getattr(tuning, "ocr_inference_batch_size", None) if tuning is not None else None
            ) or _positive(getattr(extract_params, "inference_batch_size", None))
            if ocr_batch_size:
                ocr_kwargs["inference_batch_size"] = int(ocr_batch_size)

            table_kwargs: dict[str, Any] = {}
            if extract_params.table_structure_invoke_url:
                table_kwargs["table_structure_invoke_url"] = extract_params.table_structure_invoke_url
            if extract_params.ocr_invoke_url:
                table_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                table_kwargs["api_key"] = extract_params.api_key
            if extract_params.table_output_format:
                table_kwargs["table_output_format"] = extract_params.table_output_format
            table_kwargs["ocr_version"] = getattr(extract_params, "ocr_version", "v2")
            if getattr(extract_params, "ocr_lang", None) is not None:
                table_kwargs["ocr_lang"] = extract_params.ocr_lang

            _rr = _nim_remote_http_kwargs(extract_params)
            detect_kwargs.update(_rr)
            ocr_kwargs.update(_rr)
            table_kwargs.update(_rr)

            needs_ocr = any(
                bool(ocr_kwargs.get(key))
                for key in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
            )
            page_elements_needed = extract_params.use_page_elements and (
                (extract_params.use_table_structure and extract_params.extract_tables) or needs_ocr
            )
            graph = graph >> PDFExtractionActor(**extract_kwargs)
            if page_elements_needed:
                graph = graph >> PageElementDetectionActor(**detect_kwargs)
            if extract_params.use_table_structure and extract_params.extract_tables:
                graph = graph >> TableStructureActor(**table_kwargs)
            if needs_ocr:
                ocr_archetype = resolve_ocr_archetype(extract_params)
                logger.info(
                    "Selected OCR engine: %s (%s)",
                    getattr(extract_params, "ocr_version", "v2"),
                    ocr_archetype.__name__,
                )
                graph = graph >> ocr_archetype(**ocr_kwargs)

        graph = _maybe_append_chunk_actor(graph, split_config, "pdf")

    return _append_ordered_transform_stages(
        graph,
        dedup_params=dedup_params,
        caption_params=caption_params,
        store_params=store_params,
        embed_params=embed_params,
        vdb_upload_params=effective_vdb_upload_params,
        webhook_params=webhook_params,
        stage_order=stage_order,
        supports_dedup=True,
        reshape_content_before_embed=extraction_mode in {"pdf", "image", "auto"},
    )


# build_inprocess_graph previously maintained a separate graph shape.
# In-process execution now intentionally reuses the shared graph builder so
# both modes inherit the same defaults, node ordering, and optional stages.
build_inprocess_graph = build_graph
