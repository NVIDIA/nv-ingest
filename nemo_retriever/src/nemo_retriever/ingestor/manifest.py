# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest planning for input-aware retriever ingest extraction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from nemo_retriever.common.params import (
    ASRParams,
    AudioChunkParams,
    AudioVisualFuseParams,
    ExtractParams,
    HtmlChunkParams,
    TextChunkParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.common.input_files import _is_explicit_glob_path, input_type_for_path
from nemo_retriever.common.inline_text import is_inline_text_source


DEFAULT_AUDIO_SPLIT_INTERVAL = 500000
DEFAULT_VIDEO_FRAME_FPS = 0.5


@dataclass(frozen=True)
class ExtractionBranchSpec:
    """Canonical policy for one manifest-planned extraction branch."""

    family: str
    input_types: tuple[str, ...]
    extraction_mode: str


_BRANCH_SPECS: tuple[ExtractionBranchSpec, ...] = (
    ExtractionBranchSpec(family="pdf", input_types=("pdf", "doc"), extraction_mode="pdf"),
    ExtractionBranchSpec(family="image", input_types=("image",), extraction_mode="image"),
    ExtractionBranchSpec(family="txt", input_types=("txt",), extraction_mode="text"),
    ExtractionBranchSpec(family="html", input_types=("html",), extraction_mode="html"),
    ExtractionBranchSpec(family="audio", input_types=("audio",), extraction_mode="audio"),
    # Video keeps extraction_mode="auto" because build_graph uses the presence
    # of video params to construct the dedicated video extraction chain.
    ExtractionBranchSpec(family="video", input_types=("video",), extraction_mode="auto"),
)
_BRANCH_SPECS_BY_FAMILY = {spec.family: spec for spec in _BRANCH_SPECS}
_BRANCH_SPECS_BY_INPUT_TYPE = {input_type: spec for spec in _BRANCH_SPECS for input_type in spec.input_types}


@dataclass(frozen=True)
class ManifestEntry:
    """One concrete or optional ingest input in a manifest."""

    path: str
    input_type: str | None
    is_explicit_glob: bool = False


@dataclass(frozen=True)
class InputManifest:
    """Classified input files for planner-backed ingest."""

    entries: tuple[ManifestEntry, ...]
    unsupported_files: tuple[str, ...]

    @property
    def files_by_family(self) -> dict[str, tuple[str, ...]]:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for entry in self.entries:
            if entry.input_type is None:
                continue
            grouped[_BRANCH_SPECS_BY_INPUT_TYPE[entry.input_type].family].append(entry.path)
        return {family: tuple(paths) for family, paths in grouped.items()}

    @property
    def optional_globs(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries if entry.is_explicit_glob)


@dataclass(frozen=True)
class ExtractionBranchPlan:
    """A single typed extraction branch to execute before common stages."""

    spec: ExtractionBranchSpec
    input_paths: tuple[str, ...]

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def extraction_mode(self) -> str:
        return self.spec.extraction_mode


@dataclass(frozen=True)
class ResolvedExtractionInputs:
    """Concrete graph-builder inputs for one extraction branch or explicit mode."""

    extraction_mode: str
    extract_params: Any | None
    text_params: Any | None
    html_params: Any | None
    audio_chunk_params: Any | None
    asr_params: Any | None
    video_frame_params: Any | None
    video_text_dedup_params: Any | None
    av_fuse_params: Any | None


def build_input_manifest(input_paths: Iterable[str]) -> InputManifest:
    """Classify concrete input paths without loading modality dependencies."""

    entries: list[ManifestEntry] = []
    unsupported: list[str] = []
    for path in input_paths:
        is_glob = _is_explicit_glob_path(path)
        input_type = None if is_glob else ("txt" if is_inline_text_source(path) else input_type_for_path(path))
        entries.append(ManifestEntry(path=path, input_type=input_type, is_explicit_glob=is_glob))
        if input_type is None and not is_glob:
            unsupported.append(path)
    return InputManifest(entries=tuple(entries), unsupported_files=tuple(unsupported))


def plan_extraction_branches(manifest: InputManifest) -> tuple[ExtractionBranchPlan, ...]:
    """Emit deterministic extraction branches for a validated manifest."""

    if manifest.unsupported_files:
        examples = ", ".join(manifest.unsupported_files[:3])
        raise ValueError(f"Unsupported input file type(s) for default GraphIngestor.extract(): {examples}")

    files_by_family = manifest.files_by_family
    if not files_by_family:
        # Empty optional globs should preserve the old empty-input behavior
        # without inventing modality branches that require extra dependencies.
        return (
            ExtractionBranchPlan(
                spec=_BRANCH_SPECS_BY_FAMILY["pdf"],
                input_paths=manifest.optional_globs,
            ),
        )

    branches: list[ExtractionBranchPlan] = []
    for spec in _BRANCH_SPECS:
        paths = files_by_family.get(spec.family)
        if not paths:
            continue
        branches.append(ExtractionBranchPlan(spec=spec, input_paths=paths))
    return tuple(branches)


def format_branch_summary(branches: tuple[ExtractionBranchPlan, ...]) -> str:
    return ", ".join(f"{branch.family}:{len(branch.input_paths)}" for branch in branches)


def resolve_branch_extraction_inputs(
    branch: ExtractionBranchPlan,
    *,
    extract_params: Any | None,
    text_params: Any | None,
    html_params: Any | None,
    audio_chunk_params: Any | None,
    asr_params: Any | None,
    video_frame_params: Any | None,
    video_text_dedup_params: Any | None,
    av_fuse_params: Any | None,
) -> ResolvedExtractionInputs:
    """Apply the canonical branch defaults to graph-builder inputs."""

    family = branch.family
    if family in {"pdf", "image"}:
        extract_params = extract_params or ExtractParams()
    elif family == "txt":
        text_params = text_params or TextChunkParams()
    elif family == "html":
        html_params = html_params or HtmlChunkParams()
    elif family == "audio":
        audio_chunk_params = audio_chunk_params or AudioChunkParams(
            split_type="size",
            split_interval=DEFAULT_AUDIO_SPLIT_INTERVAL,
        )
        asr_params = asr_params or default_asr_params()
    elif family == "video":
        extract_params = extract_params or ExtractParams()
        audio_chunk_params = audio_chunk_params or AudioChunkParams(
            enabled=True,
            split_type="size",
            split_interval=DEFAULT_AUDIO_SPLIT_INTERVAL,
        )
        asr_params = asr_params or default_asr_params()
        video_frame_params = video_frame_params or VideoFrameParams(
            enabled=True,
            fps=DEFAULT_VIDEO_FRAME_FPS,
            dedup=True,
        )
        video_text_dedup_params = video_text_dedup_params or VideoFrameTextDedupParams(
            enabled=True,
            max_dropped_frames=2,
        )
        av_fuse_params = av_fuse_params or AudioVisualFuseParams(enabled=True)

    return ResolvedExtractionInputs(
        extraction_mode=branch.extraction_mode,
        extract_params=extract_params,
        text_params=text_params,
        html_params=html_params,
        audio_chunk_params=audio_chunk_params,
        asr_params=asr_params,
        video_frame_params=video_frame_params,
        video_text_dedup_params=video_text_dedup_params,
        av_fuse_params=av_fuse_params,
    )


def default_asr_params() -> ASRParams:
    """Return the default ASR params for audio and video inputs.

    Returns
    -------
    ASRParams
        Params from ``asr_params_from_env()`` (for example ``AUDIO_GRPC_ENDPOINT``), with
        ``segment_audio`` off.
    """
    from nemo_retriever.operators.extract.audio.asr_actor import asr_params_from_env

    return asr_params_from_env().model_copy(update={"segment_audio": False})
