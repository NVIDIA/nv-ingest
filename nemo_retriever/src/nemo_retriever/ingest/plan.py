# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-facing graph ingest planning built on top of the lower-level ingestor execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from nemo_retriever.ingestor.manifest import (
    ExtractionBranchPlan,
    build_input_manifest,
    plan_extraction_branches,
)
from nemo_retriever.ingestor.plans import resolve_effective_dedup_params
from nemo_retriever.common.modality.ocr.config import OCRLang, OCRVersion
from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    BatchTuningParams,
    CaptionParams,
    DedupParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    StoreParams,
    TextChunkParams,
    VdbUploadParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
    build_embed_option_kwargs,
)
from nemo_retriever.common.input_files import (
    AUTO_INPUT_EXTENSIONS,
    INPUT_TYPE_EXTENSIONS,
    expand_input_file_patterns,
    resolve_input_files,
)
from nemo_retriever.ingest.index_mode import (
    RequestedIngestIndexMode,
    inspect_existing_lancedb_mode,
    resolve_ingest_index_mode,
    validate_requested_index_mode,
)
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL, resolve_embed_model
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision

IngestRunModeValue = Literal["inprocess", "batch"]
IngestInputTypeValue = Literal["auto", "pdf", "doc", "txt", "html", "image", "audio", "video"]
IngestProfileValue = Literal["auto", "fast-text"]
IngestIndexModeValue = RequestedIngestIndexMode
AudioSplitTypeValue = Literal["size", "time", "frame"]
LocalIngestEmbedBackendValue = Literal["vllm", "hf"]
OcrLangValue = OCRLang
OcrVersionValue = OCRVersion
TableOutputFormatValue = Literal["pseudo_markdown", "markdown"]
_SUPPORTED_RUN_MODES: tuple[IngestRunModeValue, ...] = ("inprocess", "batch")
_SUPPORTED_PROFILES: tuple[IngestProfileValue, ...] = ("auto", "fast-text")
_SUPPORTED_AUDIO_SPLIT_TYPES: tuple[AudioSplitTypeValue, ...] = ("size", "time", "frame")
_SUPPORTED_INPUT_TYPES: tuple[IngestInputTypeValue, ...] = (
    "auto",
    "pdf",
    "doc",
    "txt",
    "html",
    "image",
    "audio",
    "video",
)
_AUDIO_SPLIT_INTERVAL = 500000
_VIDEO_FRAME_FPS = 0.5
_VIDEO_TEXT_DEDUP_MAX_DROPPED_FRAMES = 2
_DEFAULT_TEXT_CHUNK_MAX_TOKENS = 1024
_DEFAULT_TEXT_CHUNK_OVERLAP_TOKENS = 150


@dataclass(frozen=True)
class IngestSourceOptions:
    documents: Sequence[str]
    profile: IngestProfileValue = "auto"
    input_type: IngestInputTypeValue = "auto"


@dataclass(frozen=True)
class IngestRuntimeOptions:
    run_mode: IngestRunModeValue = "inprocess"
    ray_address: str | None = None
    ray_log_to_driver: bool | None = None


@dataclass(frozen=True)
class IngestExtractBatchOptions:
    pdf_split_batch_size: int | None = None
    pdf_extract_workers: int | None = None
    pdf_extract_batch_size: int | None = None
    pdf_extract_cpus_per_task: float | None = None
    page_elements_workers: int | None = None
    page_elements_batch_size: int | None = None
    page_elements_cpus_per_actor: float | None = None
    page_elements_gpus_per_actor: float | None = None
    ocr_workers: int | None = None
    ocr_batch_size: int | None = None
    ocr_cpus_per_actor: float | None = None
    ocr_gpus_per_actor: float | None = None
    table_structure_workers: int | None = None
    table_structure_batch_size: int | None = None
    table_structure_cpus_per_actor: float | None = None
    table_structure_gpus_per_actor: float | None = None
    nemotron_parse_workers: int | None = None
    nemotron_parse_batch_size: int | None = None
    nemotron_parse_gpus_per_actor: float | None = None


@dataclass(frozen=True)
class IngestExtractOptions:
    method: str | None = None
    dpi: int | None = None
    extract_text: bool | None = None
    extract_images: bool | None = None
    extract_nested_images: bool | None = None
    extract_tables: bool | None = None
    extract_charts: bool | None = None
    extract_infographics: bool | None = None
    extract_page_as_image: bool | None = None
    use_page_elements: bool | None = None
    use_table_structure: bool | None = None
    page_elements_invoke_url: str | None = None
    ocr_invoke_url: str | None = None
    ocr_version: OcrVersionValue | None = None
    ocr_lang: OcrLangValue | None = None
    table_structure_invoke_url: str | None = None
    table_output_format: TableOutputFormatValue | None = None
    extract_api_key: str | None = None
    batch: IngestExtractBatchOptions = field(default_factory=IngestExtractBatchOptions)


@dataclass(frozen=True)
class IngestMediaOptions:
    segment_audio: bool | None = None
    audio_split_type: AudioSplitTypeValue = "size"
    audio_split_interval: int | None = None
    video_extract_audio: bool | None = None
    video_extract_frames: bool | None = None
    video_frame_fps: float | None = None
    video_frame_dedup: bool | None = None
    video_frame_text_dedup: bool | None = None
    video_frame_text_dedup_max_dropped_frames: int | None = None
    video_av_fuse: bool | None = None


@dataclass(frozen=True)
class IngestCaptionOptions:
    enabled: bool = False
    caption_invoke_url: str | None = None
    caption_api_key: str | None = None
    caption_model_name: str | None = None
    caption_device: str | None = None
    caption_context_text_max_chars: int | None = None
    caption_gpu_memory_utilization: float | None = None
    caption_temperature: float | None = None
    caption_top_p: float | None = None
    caption_max_tokens: int | None = None
    caption_infographics: bool | None = None


@dataclass(frozen=True)
class IngestDedupOptions:
    enabled: bool | None = None
    iou_threshold: float | None = None


@dataclass(frozen=True)
class IngestChunkOptions:
    enabled: bool = False
    text_chunk_max_tokens: int | None = None
    text_chunk_overlap_tokens: int | None = None


@dataclass(frozen=True)
class IngestEmbedBatchOptions:
    embed_workers: int | None = None
    embed_batch_size: int | None = None
    embed_cpus_per_actor: float | None = None
    embed_gpus_per_actor: float | None = None
    embed_workers_min: int | None = None
    embed_workers_initial: int | None = None
    embed_workers_max: int | None = None


@dataclass(frozen=True)
class IngestEmbedOptions:
    embed_invoke_url: str | None = None
    embed_model_name: str | None = None
    embed_model_provider_prefix: str | None = None
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None
    embed_api_key: str | None = None
    embed_modality: str | None = None
    text_elements_modality: str | None = None
    structured_elements_modality: str | None = None
    embed_granularity: str | None = None
    batch: IngestEmbedBatchOptions = field(default_factory=IngestEmbedBatchOptions)


@dataclass(frozen=True)
class IngestImageStoreOptions:
    images_uri: str | None = None
    workers: int | None = None


@dataclass(frozen=True)
class IngestStorageOptions:
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"
    overwrite: bool = True
    index_mode: IngestIndexModeValue = "auto"


@dataclass(frozen=True)
class IngestPlanRequest:
    source: IngestSourceOptions
    runtime: IngestRuntimeOptions = field(default_factory=IngestRuntimeOptions)
    extract: IngestExtractOptions = field(default_factory=IngestExtractOptions)
    media: IngestMediaOptions = field(default_factory=IngestMediaOptions)
    caption: IngestCaptionOptions = field(default_factory=IngestCaptionOptions)
    dedup: IngestDedupOptions = field(default_factory=IngestDedupOptions)
    chunk: IngestChunkOptions = field(default_factory=IngestChunkOptions)
    embed: IngestEmbedOptions = field(default_factory=IngestEmbedOptions)
    image_store: IngestImageStoreOptions = field(default_factory=IngestImageStoreOptions)
    storage: IngestStorageOptions = field(default_factory=IngestStorageOptions)


def _validate_run_mode(run_mode: str) -> IngestRunModeValue:
    if run_mode not in _SUPPORTED_RUN_MODES:
        raise ValueError(f"run_mode must be one of {', '.join(_SUPPORTED_RUN_MODES)}, got {run_mode!r}.")
    return cast(IngestRunModeValue, run_mode)


def validate_ingest_input_type(input_type: str) -> IngestInputTypeValue:
    if input_type not in _SUPPORTED_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {', '.join(_SUPPORTED_INPUT_TYPES)}, got {input_type!r}.")
    return cast(IngestInputTypeValue, input_type)


def validate_ingest_profile(profile: str) -> IngestProfileValue:
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(f"profile must be one of {', '.join(_SUPPORTED_PROFILES)}, got {profile!r}.")
    return cast(IngestProfileValue, profile)


def validate_ingest_index_mode(index_mode: str) -> IngestIndexModeValue:
    return validate_requested_index_mode(index_mode)


def _validate_audio_split_type(split_type: str) -> AudioSplitTypeValue:
    if split_type not in _SUPPORTED_AUDIO_SPLIT_TYPES:
        raise ValueError(
            f"audio_split_type must be one of {', '.join(_SUPPORTED_AUDIO_SPLIT_TYPES)}, got {split_type!r}."
        )
    return cast(AudioSplitTypeValue, split_type)


# The ingest command accepts bare dataset directories; expand those to supported
# files before passing file/glob inputs through the shared input normalizer.
def validate_ingest_document_types(documents: Sequence[str], *, input_type: IngestInputTypeValue) -> None:
    allowed_extensions = AUTO_INPUT_EXTENSIONS if input_type == "auto" else INPUT_TYPE_EXTENSIONS[input_type]
    unsupported = [
        document
        for document in documents
        if not any(ch in str(document) for ch in "*?[") and Path(document).suffix.lower() not in allowed_extensions
    ]
    if unsupported:
        examples = ", ".join(unsupported[:3])
        raise ValueError(f"Unsupported input file type(s) for retriever ingest: {examples}")


def expand_ingest_documents(documents: Sequence[str], *, input_type: IngestInputTypeValue = "auto") -> list[str]:
    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, input_type)
            if not directory_files:
                raise FileNotFoundError(f"No supported ingest files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    document_list = expand_input_file_patterns(inputs)
    validate_ingest_document_types(document_list, input_type=input_type)
    return document_list


@dataclass(frozen=True)
class ResolvedIngestPlan:
    documents: list[str]
    profile: IngestProfileValue
    branches: tuple[ExtractionBranchPlan, ...]
    create_kwargs: dict[str, Any]
    extract_params: ExtractParams
    text_params: TextChunkParams | None
    html_params: HtmlChunkParams | None
    audio_chunk_params: AudioChunkParams | None
    asr_params: ASRParams | None
    video_frame_params: VideoFrameParams | None
    video_text_dedup_params: VideoFrameTextDedupParams | None
    av_fuse_params: AudioVisualFuseParams | None
    split_config: dict[str, Any] | None
    dedup_params: DedupParams | None
    caption_params: CaptionParams | None
    embed_params: EmbedParams | None
    store_params: StoreParams | None
    vdb_params: VdbUploadParams | None
    lancedb_uri: str
    table_name: str
    sparse: bool = False

    def extract_call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for key, value in {
            "text_params": self.text_params,
            "html_params": self.html_params,
            "audio_chunk_params": self.audio_chunk_params,
            "asr_params": self.asr_params,
            "video_frame_params": self.video_frame_params,
            "video_text_dedup_params": self.video_text_dedup_params,
            "av_fuse_params": self.av_fuse_params,
        }.items():
            if value is not None:
                kwargs[key] = value
        return kwargs


def _branch_families(branches: Sequence[ExtractionBranchPlan]) -> set[str]:
    return {branch.family for branch in branches}


def _require_branch_families(
    *,
    profile: IngestProfileValue,
    branches: Sequence[ExtractionBranchPlan],
    allowed: set[str],
    description: str,
) -> None:
    observed = _branch_families(branches)
    disallowed = observed - allowed
    if disallowed:
        allowed_text = ", ".join(sorted(allowed))
        observed_text = ", ".join(sorted(observed))
        raise ValueError(
            f"--profile {profile} only supports {description} inputs ({allowed_text}); observed {observed_text}."
        )


def _validate_profile_manifest(profile: IngestProfileValue, branches: Sequence[ExtractionBranchPlan]) -> None:
    if profile == "fast-text":
        _require_branch_families(profile=profile, branches=branches, allowed={"pdf"}, description="PDF/document")


def profile_extract_defaults(profile: IngestProfileValue) -> dict[str, Any]:
    if profile == "fast-text":
        return {
            "method": "pdfium",
            "extract_text": True,
            "extract_images": False,
            "extract_tables": False,
            "extract_charts": False,
            "extract_infographics": False,
            "extract_page_as_image": False,
            "use_page_elements": False,
        }
    return {"method": "pdfium_hybrid"}


def _build_asr_params(*, segment_audio: bool | None, needed: bool) -> ASRParams | None:
    if not needed and segment_audio is None:
        return None
    from nemo_retriever.operators.extract.audio.asr_actor import asr_params_from_env

    params = asr_params_from_env()
    if segment_audio is None:
        return params
    return params.model_copy(update={"segment_audio": bool(segment_audio)})


def _resolve_media_params(
    *,
    branches: Sequence[ExtractionBranchPlan],
    media: IngestMediaOptions,
    audio_split_type: AudioSplitTypeValue,
) -> tuple[
    AudioChunkParams | None,
    ASRParams | None,
    VideoFrameParams | None,
    VideoFrameTextDedupParams | None,
    AudioVisualFuseParams | None,
]:
    families = _branch_families(branches)
    needs_audio = bool(families & {"audio", "video"})
    needs_video = "video" in families
    if not needs_audio and not needs_video:
        return None, _build_asr_params(segment_audio=media.segment_audio, needed=False), None, None, None

    split_interval = (
        int(media.audio_split_interval) if media.audio_split_interval is not None else _AUDIO_SPLIT_INTERVAL
    )
    audio_chunk_params = AudioChunkParams(
        enabled=bool(media.video_extract_audio) if media.video_extract_audio is not None and needs_video else True,
        split_type=audio_split_type,
        split_interval=split_interval,
    )
    asr_params = _build_asr_params(segment_audio=media.segment_audio, needed=needs_audio)

    if not needs_video:
        return audio_chunk_params, asr_params, None, None, None

    video_frame_params = VideoFrameParams(
        enabled=bool(media.video_extract_frames) if media.video_extract_frames is not None else True,
        fps=float(media.video_frame_fps) if media.video_frame_fps is not None else _VIDEO_FRAME_FPS,
        dedup=bool(media.video_frame_dedup) if media.video_frame_dedup is not None else True,
    )
    video_text_dedup_params = VideoFrameTextDedupParams(
        enabled=bool(media.video_frame_text_dedup) if media.video_frame_text_dedup is not None else True,
        max_dropped_frames=(
            int(media.video_frame_text_dedup_max_dropped_frames)
            if media.video_frame_text_dedup_max_dropped_frames is not None
            else _VIDEO_TEXT_DEDUP_MAX_DROPPED_FRAMES
        ),
    )
    av_fuse_params = AudioVisualFuseParams(
        enabled=bool(media.video_av_fuse) if media.video_av_fuse is not None else True
    )
    return audio_chunk_params, asr_params, video_frame_params, video_text_dedup_params, av_fuse_params


def build_caption_params(
    *,
    enabled: bool,
    caption_invoke_url: str | None = None,
    caption_api_key: str | None = None,
    caption_model_name: str | None = None,
    caption_device: str | None = None,
    caption_context_text_max_chars: int | None = None,
    caption_gpu_memory_utilization: float | None = None,
    caption_temperature: float | None = None,
    caption_top_p: float | None = None,
    caption_max_tokens: int | None = None,
    caption_infographics: bool | None = None,
) -> CaptionParams | None:
    overrides = {
        "caption_invoke_url": caption_invoke_url,
        "caption_model_name": caption_model_name,
        "caption_device": caption_device,
        "caption_context_text_max_chars": caption_context_text_max_chars,
        "caption_gpu_memory_utilization": caption_gpu_memory_utilization,
        "caption_temperature": caption_temperature,
        "caption_top_p": caption_top_p,
        "caption_max_tokens": caption_max_tokens,
        "caption_infographics": caption_infographics,
    }
    if not enabled:
        provided = [name for name, value in overrides.items() if value is not None]
        if provided:
            raise ValueError(f"Caption options require --caption: {', '.join(provided)}.")
        return None
    if caption_context_text_max_chars is not None and caption_context_text_max_chars < 0:
        raise ValueError("caption_context_text_max_chars must be >= 0.")

    caption_kwargs = {
        key: value
        for key, value in {
            "endpoint_url": caption_invoke_url,
            "api_key": caption_api_key,
            "model_name": caption_model_name,
            "device": caption_device,
            "context_text_max_chars": caption_context_text_max_chars,
            "gpu_memory_utilization": caption_gpu_memory_utilization,
            "temperature": caption_temperature,
            "top_p": caption_top_p,
            "max_tokens": caption_max_tokens,
            "caption_infographics": caption_infographics,
        }.items()
        if value is not None
    }
    return CaptionParams(**caption_kwargs)


def build_dedup_params(*, enabled: bool | None, iou_threshold: float | None = None) -> DedupParams | None:
    """Build explicit dedup parameters while preserving unspecified intent.

    Parameters
    ----------
    enabled
        ``True`` explicitly enables deduplication, ``False`` returns the
        all-disabled opt-out, and ``None`` leaves deduplication unspecified.
    iou_threshold
        Optional bounding-box intersection-over-union threshold. This override
        is valid only when ``enabled`` is explicitly ``True``.

    Returns
    -------
    DedupParams | None
        Explicit enabled or disabled parameters, or ``None`` when the caller
        did not specify a deduplication policy.

    Raises
    ------
    ValueError
        If ``iou_threshold`` is provided without explicitly enabling
        deduplication or is outside the supported range.
    """

    if enabled is not True:
        if iou_threshold is not None:
            raise ValueError("Dedup options require --dedup: dedup_iou_threshold.")
        if enabled is False:
            return DedupParams(content_hash=False, bbox_iou=False)
        return None
    dedup_kwargs = {}
    if iou_threshold is not None:
        dedup_kwargs["iou_threshold"] = iou_threshold
    return DedupParams(**dedup_kwargs)


def build_store_params(*, images_uri: str | None, workers: int | None = None) -> StoreParams | None:
    if images_uri is None:
        return None

    store_kwargs: dict[str, Any] = {"storage_uri": images_uri}
    if workers:
        store_kwargs["batch_tuning"] = BatchTuningParams(store_workers=workers)
    return StoreParams(**store_kwargs)


def _build_extract_batch_tuning(batch: IngestExtractBatchOptions) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "pdf_split_batch_size": batch.pdf_split_batch_size,
            "pdf_extract_workers": batch.pdf_extract_workers,
            "pdf_extract_batch_size": batch.pdf_extract_batch_size,
            # BatchTuningParams names this per-Ray-task reservation num_cpus.
            "pdf_extract_num_cpus": batch.pdf_extract_cpus_per_task,
            "page_elements_workers": batch.page_elements_workers,
            "page_elements_batch_size": batch.page_elements_batch_size,
            "page_elements_cpus_per_actor": batch.page_elements_cpus_per_actor,
            "gpu_page_elements": batch.page_elements_gpus_per_actor,
            "ocr_workers": batch.ocr_workers,
            "ocr_inference_batch_size": batch.ocr_batch_size,
            "ocr_cpus_per_actor": batch.ocr_cpus_per_actor,
            "gpu_ocr": batch.ocr_gpus_per_actor,
            "table_structure_workers": batch.table_structure_workers,
            "table_structure_batch_size": batch.table_structure_batch_size,
            "table_structure_cpus_per_actor": batch.table_structure_cpus_per_actor,
            "gpu_table_structure": batch.table_structure_gpus_per_actor,
            "nemotron_parse_workers": batch.nemotron_parse_workers,
            "nemotron_parse_batch_size": batch.nemotron_parse_batch_size,
            "gpu_nemotron_parse": batch.nemotron_parse_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def build_text_chunk_kwargs(
    *,
    enabled: bool,
    text_chunk_max_tokens: int | None = None,
    text_chunk_overlap_tokens: int | None = None,
) -> tuple[bool, dict[str, int]]:
    resolved_enabled = bool(enabled) or text_chunk_max_tokens is not None or text_chunk_overlap_tokens is not None
    if not resolved_enabled:
        return False, {}
    return True, {
        "max_tokens": (
            int(text_chunk_max_tokens) if text_chunk_max_tokens is not None else _DEFAULT_TEXT_CHUNK_MAX_TOKENS
        ),
        "overlap_tokens": (
            int(text_chunk_overlap_tokens)
            if text_chunk_overlap_tokens is not None
            else _DEFAULT_TEXT_CHUNK_OVERLAP_TOKENS
        ),
    }


def _split_config_for_families(
    families: set[str],
    text_chunk_kwargs: dict[str, int],
) -> dict[str, Any] | None:
    if not text_chunk_kwargs:
        return None

    chunk_dict = TextChunkParams(**text_chunk_kwargs).model_dump()
    split_config: dict[str, Any] = {}
    if families & {"pdf", "doc"}:
        split_config["pdf"] = dict(chunk_dict)
    # Txt/html chunking is already passed through text_params/html_params.
    # Keep split_config for families that do not have dedicated params here.
    if "image" in families:
        split_config["image"] = dict(chunk_dict)
    if "audio" in families:
        split_config["audio"] = dict(chunk_dict)
    if "video" in families:
        split_config["video"] = dict(chunk_dict)
        split_config["audio"] = dict(chunk_dict)
    return split_config or None


def resolve_ingest_plan(request: IngestPlanRequest) -> ResolvedIngestPlan:
    """Resolve root ingest options into ordinary params for one extract call.

    ``run_mode`` defaults to Ray-free local execution. Callers that need Ray
    Data scale-out should pass ``run_mode="batch"`` in ``request.runtime``.
    ``input_type`` remains a private expansion/validation constraint;
    extraction still routes from the manifest.
    """

    source = request.source
    runtime = request.runtime
    extract = request.extract
    media = request.media
    chunk = request.chunk
    embed = request.embed
    storage = request.storage

    validated_run_mode = _validate_run_mode(runtime.run_mode)
    validated_profile = validate_ingest_profile(source.profile)
    validated_input_type = validate_ingest_input_type(source.input_type)
    requested_index_mode = validate_ingest_index_mode(storage.index_mode)
    existing_index_mode = (
        None if storage.overwrite else inspect_existing_lancedb_mode(storage.lancedb_uri, storage.table_name)
    )
    resolved_index_mode = resolve_ingest_index_mode(
        requested_index_mode,
        overwrite=storage.overwrite,
        existing_mode=existing_index_mode,
    )
    validated_audio_split_type = _validate_audio_split_type(media.audio_split_type)
    document_list = expand_ingest_documents(source.documents, input_type=validated_input_type)
    branches = plan_extraction_branches(build_input_manifest(document_list))
    _validate_profile_manifest(validated_profile, branches)

    extract_kwargs = profile_extract_defaults(validated_profile)
    extract_kwargs.update(
        {
            key: value
            for key, value in {
                "method": extract.method,
                "dpi": extract.dpi,
                "extract_text": extract.extract_text,
                "extract_images": extract.extract_images,
                "extract_nested_images": extract.extract_nested_images,
                "extract_tables": extract.extract_tables,
                "extract_charts": extract.extract_charts,
                "extract_infographics": extract.extract_infographics,
                "extract_page_as_image": extract.extract_page_as_image,
                "use_page_elements": extract.use_page_elements,
                "use_table_structure": extract.use_table_structure,
                "page_elements_invoke_url": extract.page_elements_invoke_url,
                "ocr_invoke_url": extract.ocr_invoke_url,
                "ocr_version": extract.ocr_version,
                "ocr_lang": extract.ocr_lang,
                "table_structure_invoke_url": extract.table_structure_invoke_url,
                "table_output_format": extract.table_output_format,
                "api_key": extract.extract_api_key,
            }.items()
            if value is not None
        }
    )
    if extract.table_output_format == "markdown":
        extract_kwargs["use_table_structure"] = True

    extract_tuning = _build_extract_batch_tuning(extract.batch)
    if extract_tuning is not None:
        extract_kwargs["batch_tuning"] = extract_tuning

    embedding_model_name = None if resolved_index_mode == "sparse" else resolve_embed_model(embed.embed_model_name)
    embedding_model_revision = None
    if (
        embedding_model_name is not None
        and embedding_model_name != NEMOTRON_3_EMBED_MODEL
        and not str(embed.embed_invoke_url or "").strip()
    ):
        embedding_model_revision = resolve_embed_model_revision(embedding_model_name, None)
    embed_runtime_model_name = (
        embedding_model_name
        if embed.embed_model_name is not None or embed.embed_model_provider_prefix is not None
        else None
    )
    embed_kwargs = build_embed_option_kwargs(
        embed.embed_invoke_url,
        embed_runtime_model_name,
        embed_model_revision=embedding_model_revision if embed_runtime_model_name is not None else None,
        local_ingest_embed_backend=embed.local_ingest_embed_backend,
        embed_api_key=embed.embed_api_key,
        embed_model_provider_prefix=embed.embed_model_provider_prefix,
        embed_modality=embed.embed_modality,
        text_elements_modality=embed.text_elements_modality,
        structured_elements_modality=embed.structured_elements_modality,
        embed_granularity=embed.embed_granularity,
        embed_workers=embed.batch.embed_workers,
        embed_workers_min=embed.batch.embed_workers_min,
        embed_workers_initial=embed.batch.embed_workers_initial,
        embed_workers_max=embed.batch.embed_workers_max,
        embed_batch_size=embed.batch.embed_batch_size,
        embed_cpus_per_actor=embed.batch.embed_cpus_per_actor,
        embed_gpus_per_actor=embed.batch.embed_gpus_per_actor,
    )
    extract_params = ExtractParams(**extract_kwargs)
    embed_params = None if resolved_index_mode == "sparse" else EmbedParams(**embed_kwargs) if embed_kwargs else None
    vdb_upload_kwargs = {
        "uri": storage.lancedb_uri,
        "table_name": storage.table_name,
        "overwrite": bool(storage.overwrite),
    }
    # Keep dense ingest kwargs unchanged unless the index mode needs additional LanceDB behavior.
    if resolved_index_mode == "sparse":
        vdb_upload_kwargs["sparse"] = True
    elif resolved_index_mode == "hybrid":
        vdb_upload_kwargs["hybrid"] = True
    if embedding_model_name is not None:
        vdb_upload_kwargs["embedding_model_name"] = embedding_model_name
        if embed.embed_model_name is not None:
            vdb_upload_kwargs["vector_dim"] = None
    if embedding_model_revision is not None:
        vdb_upload_kwargs["embedding_model_revision"] = embedding_model_revision
    vdb_params = VdbUploadParams(vdb_kwargs=vdb_upload_kwargs)
    caption_params = build_caption_params(
        enabled=request.caption.enabled,
        caption_invoke_url=request.caption.caption_invoke_url,
        caption_api_key=request.caption.caption_api_key,
        caption_model_name=request.caption.caption_model_name,
        caption_device=request.caption.caption_device,
        caption_context_text_max_chars=request.caption.caption_context_text_max_chars,
        caption_gpu_memory_utilization=request.caption.caption_gpu_memory_utilization,
        caption_temperature=request.caption.caption_temperature,
        caption_top_p=request.caption.caption_top_p,
        caption_max_tokens=request.caption.caption_max_tokens,
        caption_infographics=request.caption.caption_infographics,
    )
    dedup_params = build_dedup_params(enabled=request.dedup.enabled, iou_threshold=request.dedup.iou_threshold)
    store_params = build_store_params(images_uri=request.image_store.images_uri, workers=request.image_store.workers)

    families = _branch_families(branches)
    dedup_params = resolve_effective_dedup_params(
        dedup_params,
        caption_enabled=caption_params is not None,
        image_only=families == {"image"},
    )
    text_chunk_enabled, text_chunk_kwargs = build_text_chunk_kwargs(
        enabled=chunk.enabled,
        text_chunk_max_tokens=chunk.text_chunk_max_tokens,
        text_chunk_overlap_tokens=chunk.text_chunk_overlap_tokens,
    )
    text_params = (
        (TextChunkParams(**text_chunk_kwargs) if text_chunk_enabled else TextChunkParams())
        if "txt" in families
        else None
    )
    html_params = (
        (HtmlChunkParams(**text_chunk_kwargs) if text_chunk_enabled else HtmlChunkParams())
        if "html" in families
        else None
    )
    split_config = _split_config_for_families(families, text_chunk_kwargs) if text_chunk_enabled else None
    (
        audio_chunk_params,
        asr_params,
        video_frame_params,
        video_text_dedup_params,
        av_fuse_params,
    ) = _resolve_media_params(
        branches=branches,
        media=media,
        audio_split_type=validated_audio_split_type,
    )

    create_kwargs: dict[str, Any] = {"run_mode": validated_run_mode}
    if runtime.ray_address is not None:
        create_kwargs["ray_address"] = runtime.ray_address
    if runtime.ray_log_to_driver is not None:
        create_kwargs["ray_log_to_driver"] = runtime.ray_log_to_driver

    return ResolvedIngestPlan(
        documents=document_list,
        profile=validated_profile,
        branches=branches,
        create_kwargs=create_kwargs,
        extract_params=extract_params,
        text_params=text_params,
        html_params=html_params,
        audio_chunk_params=audio_chunk_params,
        asr_params=asr_params,
        video_frame_params=video_frame_params,
        video_text_dedup_params=video_text_dedup_params,
        av_fuse_params=av_fuse_params,
        split_config=split_config,
        dedup_params=dedup_params,
        caption_params=caption_params,
        embed_params=embed_params,
        store_params=store_params,
        vdb_params=vdb_params,
        lancedb_uri=storage.lancedb_uri,
        table_name=storage.table_name,
        sparse=resolved_index_mode == "sparse",
    )
