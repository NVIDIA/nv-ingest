# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-native mixed file extraction for Ray Data batches."""

from __future__ import annotations

import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.operators.extract.audio.asr_actor import ASRActor
from nemo_retriever.operators.extract.audio.chunk_actor import MediaChunkActor
from nemo_retriever.operators.extract.audio.asr_actor import asr_params_from_env
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.extract.html.ray_data import HtmlSplitActor
from nemo_retriever.operators.extract.image.ray_data import ImageLoadActor
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.graph.operator_resolution import resolve_operator_class
from nemo_retriever.operators.extract.ocr.ocr import resolve_ocr_archetype
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.common.params import ASRParams
from nemo_retriever.common.params import AudioChunkParams
from nemo_retriever.common.params import CaptionParams
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params import HtmlChunkParams
from nemo_retriever.common.params import PdfSplitParams
from nemo_retriever.common.params import AudioVisualFuseParams
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.common.params import VideoFrameParams
from nemo_retriever.common.params import VideoFrameTextDedupParams
from nemo_retriever.common.params import resolve_split_params
from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor
from nemo_retriever.operators.extract.pdf.extract import PDFExtractionActor, build_pdf_extraction_kwargs
from nemo_retriever.operators.extract.pdf.split import PDFSplitActor
from nemo_retriever.operators.extract.table.table_detection import TableStructureActor
from nemo_retriever.operators.extract.txt.ray_data import TextChunkCPUActor, TxtSplitActor
from nemo_retriever.common.modality.convert.to_pdf import DocToPdfConversionActor
from nemo_retriever.operators.extract.video.audio_visual_fuser import AudioVisualFuser
from nemo_retriever.operators.extract.video.frame_actor import VideoFrameActor
from nemo_retriever.operators.extract.video.ocr_actor import VideoFrameOCRActor
from nemo_retriever.operators.extract.video.text_dedup import VideoFrameTextDedup
from nemo_retriever.operators.extract.video.frame_actor import dedup_video_frames
from nemo_retriever.operators.extract.video.split import video_asr_audio_chunk_params
from nemo_retriever.graph.designer import designer_component
from nemo_retriever.common.input_files import INPUT_TYPE_EXTENSIONS
from nemo_retriever.common import ray_resource_hueristics as _rrh

logger = logging.getLogger(__name__)

# Define file type mappings
PDF_EXTENSIONS = INPUT_TYPE_EXTENSIONS["pdf"] | INPUT_TYPE_EXTENSIONS["doc"]
TEXT_EXTENSIONS = INPUT_TYPE_EXTENSIONS["txt"]
HTML_EXTENSIONS = INPUT_TYPE_EXTENSIONS["html"]
AUDIO_EXTENSIONS = INPUT_TYPE_EXTENSIONS["audio"]
IMAGE_EXTENSIONS = INPUT_TYPE_EXTENSIONS["image"]
VIDEO_EXTENSIONS = INPUT_TYPE_EXTENSIONS["video"]
DEFAULT_AUDIO_SPLIT_INTERVAL = 500000
DEFAULT_VIDEO_FRAME_FPS = 0.5


def _unsupported_extension_message(ext: str) -> str:
    display_ext = ext or "<none>"
    return (
        f"Unsupported file extension '{display_ext}' for extraction_mode='auto'. "
        "Provide a supported extension or set extraction_mode explicitly."
    )


def _has_endpoint(*values: Any) -> bool:
    return any(bool(str(value or "").strip()) for value in values)


def _default_asr_params() -> ASRParams:
    return asr_params_from_env().model_copy(update={"segment_audio": False})


def _default_audio_chunk_params() -> AudioChunkParams:
    return AudioChunkParams(split_type="size", split_interval=DEFAULT_AUDIO_SPLIT_INTERVAL)


def _default_video_frame_params() -> VideoFrameParams:
    return VideoFrameParams(enabled=True, fps=DEFAULT_VIDEO_FRAME_FPS, dedup=True)


def _parse_mode_enabled(extract_params: ExtractParams) -> bool:
    tuning = getattr(extract_params, "batch_tuning", None)
    return extract_params.method == "nemotron_parse" or (
        tuning is not None
        and all(
            getattr(tuning, name, None)
            for name in ("nemotron_parse_workers", "gpu_nemotron_parse", "nemotron_parse_batch_size")
        )
    )


def _ocr_stage_needed(extract_params: ExtractParams) -> bool:
    if extract_params.extract_text:
        return True
    if extract_params.extract_tables:
        # OCR is always needed for table crops: either to produce pseudo-markdown
        # (when use_table_structure=False) or to join against the
        # table_structure_v1 detections published by TableStructureActor.
        return True
    if extract_params.extract_charts:
        return True
    if extract_params.extract_infographics:
        return True
    return False


def _page_elements_stage_needed(extract_params: ExtractParams) -> bool:
    needs_table_structure = extract_params.use_table_structure and extract_params.extract_tables
    return extract_params.use_page_elements and (needs_table_structure or _ocr_stage_needed(extract_params))


def _extract_params_need_local_gpu(extraction_mode: str, extract_params: ExtractParams | None) -> bool:
    if extract_params is None:
        return False
    if extraction_mode not in {"pdf", "image", "auto"}:
        return False

    if _parse_mode_enabled(extract_params):
        return not _has_endpoint(extract_params.nemotron_parse_invoke_url, extract_params.invoke_url)

    if _page_elements_stage_needed(extract_params) and not _has_endpoint(extract_params.page_elements_invoke_url):
        return True
    if (
        extract_params.use_table_structure
        and extract_params.extract_tables
        and not _has_endpoint(extract_params.table_structure_invoke_url)
    ):
        return True
    if _ocr_stage_needed(extract_params) and not _has_endpoint(extract_params.ocr_invoke_url):
        return True
    return False


class _MultiTypeExtractBase(AbstractOperator):
    """Shared implementation for GPU and CPU multi-type extract actors."""

    def __init__(
        self,
        extraction_mode: str = "auto",
        extract_params: ExtractParams | None = None,
        text_params: TextChunkParams | None = None,
        html_params: HtmlChunkParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
        asr_params: ASRParams | None = None,
        caption_params: CaptionParams | None = None,
        video_frame_params: VideoFrameParams | None = None,
        video_text_dedup_params: VideoFrameTextDedupParams | None = None,
        av_fuse_params: AudioVisualFuseParams | None = None,
        split_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.extraction_mode = extraction_mode
        self.extract_params = extract_params or ExtractParams()
        self.text_params = text_params or TextChunkParams()
        self.html_params = html_params or HtmlChunkParams()
        self.audio_chunk_params = audio_chunk_params or _default_audio_chunk_params()
        self.asr_params = asr_params or _default_asr_params()
        self.caption_params = caption_params
        self.video_frame_params = video_frame_params or _default_video_frame_params()
        self.video_text_dedup_params = video_text_dedup_params or VideoFrameTextDedupParams(
            enabled=True,
            max_dropped_frames=2,
        )
        self.av_fuse_params = av_fuse_params or AudioVisualFuseParams()
        self._split_config: dict[str, Any] = split_config if split_config is not None else resolve_split_params(None)
        self._resolved_resources = None

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame | dict[str, list[str]]:
        if isinstance(data, pd.DataFrame):
            return data
        return self._group_file_inputs(data)

    def process(self, batch_df: Any, **kwargs: Any) -> pd.DataFrame | list[Any]:
        if isinstance(batch_df, dict):
            if not any(batch_df.values()):
                return []
            raise ValueError("MultiTypeExtractOperator.process expects a pandas DataFrame for non-empty grouped inputs")

        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()

        grouped = self._group_batches(batch_df)
        outputs: list[pd.DataFrame] = []

        if not grouped["pdf"].empty:
            outputs.append(self._run_pdf_pipeline(grouped["pdf"]))
        if not grouped["image"].empty:
            outputs.append(self._run_image_pipeline(grouped["image"]))
        if not grouped["text"].empty:
            text_params = self._effective_chunk_params("text")
            outputs.append(TxtSplitActor(params=text_params).run(grouped["text"]))
        if not grouped["html"].empty:
            html_params = self._effective_chunk_params("html")
            outputs.append(HtmlSplitActor(params=html_params).run(grouped["html"]))
        if not grouped["audio"].empty:
            audio_work, audio_spill = self._materialize_media_bytes(grouped["audio"])
            try:
                audio_df = MediaChunkActor(params=self.audio_chunk_params).run(audio_work)
                audio_df = ASRActor(params=self.asr_params).run(audio_df)
                outputs.append(self._maybe_chunk(audio_df, "audio"))
            finally:
                if audio_spill is not None:
                    shutil.rmtree(audio_spill, ignore_errors=True)
        if not grouped["video"].empty:
            outputs.append(self._run_video_pipeline(grouped["video"]))

        non_empty = [df for df in outputs if isinstance(df, pd.DataFrame) and not df.empty]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True, sort=False)

    def _group_batches(self, batch_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        grouped: dict[str, list[int]] = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}
        explicit_mode = self.extraction_mode

        for idx, row in batch_df.iterrows():
            path = str(row.get("path") or "")
            ext = Path(path).suffix.lower()
            ext_mode = self._mode_for_extension(ext)
            if explicit_mode == "auto":
                target = ext_mode
            elif explicit_mode in {"text", "html"}:
                # Honor the file suffix so a mis-set extraction_mode does not
                # force HTML bytes through the TXT splitter (or vice versa).
                target = ext_mode or explicit_mode
            else:
                target = explicit_mode
            if explicit_mode == "auto" and target == "":
                logger.warning(
                    _unsupported_extension_message(ext),
                    extra={"path": path},
                )
                continue
            if target in grouped:
                grouped[target].append(idx)

        return {
            key: batch_df.loc[indexes].reset_index(drop=True) if indexes else pd.DataFrame(columns=batch_df.columns)
            for key, indexes in grouped.items()
        }

    def _mode_for_extension(self, ext: str) -> str:
        if ext in PDF_EXTENSIONS:
            return "pdf"
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in TEXT_EXTENSIONS:
            return "text"
        if ext in HTML_EXTENSIONS:
            return "html"
        if ext in AUDIO_EXTENSIONS:
            return "audio"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        return ""

    def _group_file_inputs(self, data: Any) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}

        if isinstance(data, (str, os.PathLike)):
            path = Path(data)
            if path.is_dir():
                files = [str(candidate) for candidate in path.rglob("*") if candidate.is_file()]
            else:
                files = [str(path)]
        elif isinstance(data, (list, tuple)):
            files = [str(item) for item in data]
        else:
            raise ValueError("MultiTypeExtractOperator expects a pandas DataFrame Ray batch")

        explicit_mode = self.extraction_mode
        for path in files:
            ext = Path(path).suffix.lower()
            target = explicit_mode if explicit_mode != "auto" else self._mode_for_extension(ext)
            if explicit_mode == "auto" and target == "":
                logger.warning(
                    _unsupported_extension_message(ext),
                    extra={"path": path},
                )
                continue
            if target in grouped:
                grouped[target].append(path)

        return grouped

    def _run_pdf_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        extract_params = self.extract_params
        split_actor = PDFSplitActor(
            split_params=PdfSplitParams(
                start_page=getattr(extract_params, "start_page", None),
                end_page=getattr(extract_params, "end_page", None),
            )
        )
        batch_df = DocToPdfConversionActor().run(batch_df)
        batch_df = split_actor.run(batch_df)

        parse_mode = _parse_mode_enabled(extract_params)

        if parse_mode:
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
            if extract_params.nemotron_parse_model:
                parse_kwargs["nemotron_parse_model"] = extract_params.nemotron_parse_model
            if extract_params.api_key:
                parse_kwargs["api_key"] = extract_params.api_key
            batch_df = self._instantiate_resolved(NemotronParseActor, **parse_kwargs).run(batch_df)
        else:
            # standard non-parse path: continue with PDFExtractionActor and detection
            extract_kwargs = build_pdf_extraction_kwargs(extract_params)
            batch_df = PDFExtractionActor(**extract_kwargs).run(batch_df)
            batch_df = self._run_detection_pipeline(batch_df)
        return self._maybe_chunk(batch_df, "pdf")

    def _run_image_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df = ImageLoadActor().run(batch_df)
        batch_df = self._run_detection_pipeline(batch_df)
        return self._maybe_chunk(batch_df, "image")

    def _video_ocr_kwargs(self) -> dict[str, Any]:
        """Build OCR kwargs for the video frame branch from ``extract_params``.

        Video OCR shares its endpoint, API key, batch size, and timeout with
        the PDF/image OCR config — the user only configures OCR once via
        :class:`ExtractParams`.
        """
        ep = self.extract_params
        return {
            "ocr_invoke_url": ep.ocr_invoke_url,
            "ocr_version": getattr(ep, "ocr_version", "v2"),
            "ocr_lang": getattr(ep, "ocr_lang", None),
            "api_key": ep.ocr_api_key or ep.api_key,
            "inference_batch_size": int(ep.inference_batch_size),
            "request_timeout_s": float(ep.ocr_request_timeout_s or ep.request_timeout_s),
        }

    @staticmethod
    def _materialize_media_bytes(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
        """Spill in-memory ``bytes`` to temp files when ``path`` is not on disk.

        Returns ``(updated_df, tmpdir)`` where *tmpdir* is ``None`` when no
        spilling was needed (all paths already exist on disk).  The caller
        **must** delete *tmpdir* when finished.
        """
        if "bytes" not in batch_df.columns:
            return batch_df, None

        needs_spill = []
        for idx, row in batch_df.iterrows():
            p = str(row.get("path") or "")
            if p and not Path(p).is_file() and row.get("bytes") is not None:
                needs_spill.append(idx)

        if not needs_spill:
            return batch_df, None

        tmpdir = tempfile.mkdtemp(prefix="retriever_media_spill_")
        df = batch_df.copy()
        for idx in needs_spill:
            row = df.loc[idx]
            original_name = Path(str(row["path"])).name or f"media_{idx}"
            dest = Path(tmpdir) / original_name
            dest.write_bytes(row["bytes"])
            df.at[idx, "path"] = str(dest)
        return df, tmpdir

    def _run_video_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Run audio-from-video ASR + frame OCR + (optional) scene fusion.

        Branch A: ``MediaChunkActor`` demuxes the video's audio track
        before chunking and ``ASRActor`` runs ASR on those audio bytes
        instead of the video container. Emits per-utterance audio rows.

        Branch B: ``VideoFrameActor`` extracts frames at
        ``video_frame_params.fps``; optional content-hash dedup;
        ``VideoFrameOCRActor`` (Nemotron OCR) runs full-frame OCR
        directly — no page-elements detection; ``VideoFrameTextDedup``
        then collapses time-adjacent runs of identical OCR text per
        ``source_path`` (mirrors the Ray graph in ``build_graph``).
        Emits per-frame rows.

        Branch C: ``AudioVisualFuser`` self-joins audio rows against
        concurrent frame OCR rows and appends fused per-utterance rows
        with text = audio + visible OCR. Skipped when
        ``av_fuse_params.enabled`` is False or when neither branch
        produced rows.
        """
        work_df, spill_dir = self._materialize_media_bytes(batch_df)
        try:
            return self._run_video_pipeline_inner(work_df)
        finally:
            if spill_dir is not None:
                shutil.rmtree(spill_dir, ignore_errors=True)

    def _run_video_pipeline_inner(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        # Branch A: audio-from-video → ASR. Skipped when the caller disables
        # audio (visual-only recall benchmarks); mirrors ``build_graph``'s
        # ``audio_enabled`` gate.
        audio_enabled = self.audio_chunk_params.enabled
        if audio_enabled:
            audio_chunks = MediaChunkActor(params=video_asr_audio_chunk_params(self.audio_chunk_params)).run(batch_df)
            audio_out = ASRActor(params=self.asr_params).run(audio_chunks)
        else:
            audio_out = pd.DataFrame()

        # Branch B: frame extraction → optional dedup → full-frame OCR → text dedup
        frame_df = VideoFrameActor(params=self.video_frame_params).run(batch_df)
        if self.video_frame_params.dedup and isinstance(frame_df, pd.DataFrame) and not frame_df.empty:
            frame_df = dedup_video_frames(frame_df)
        if isinstance(frame_df, pd.DataFrame) and not frame_df.empty:
            ocr_kwargs = self._video_ocr_kwargs()
            frame_out = self._instantiate_resolved(VideoFrameOCRActor, **ocr_kwargs).run(frame_df)
            if self.video_text_dedup_params.enabled and isinstance(frame_out, pd.DataFrame) and not frame_out.empty:
                frame_out = VideoFrameTextDedup(params=self.video_text_dedup_params).run(frame_out)
        else:
            frame_out = pd.DataFrame()

        # Concat A + B
        non_empty = [df for df in (audio_out, frame_out) if isinstance(df, pd.DataFrame) and not df.empty]
        if not non_empty:
            return pd.DataFrame()
        combined = pd.concat(non_empty, ignore_index=True, sort=False)

        # Branch C: fused audio+visual rows. Requires audio to have run —
        # the fuser drops all video_frame rows when no audio is present.
        if audio_enabled and self.av_fuse_params.enabled:
            combined = AudioVisualFuser(params=self.av_fuse_params).run(combined)
        return self._maybe_chunk(combined, "video")

    def _run_detection_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        extract_params = self.extract_params
        tuning = getattr(extract_params, "batch_tuning", None)
        ocr_version = getattr(extract_params, "ocr_version", "v2")
        ocr_lang = getattr(extract_params, "ocr_lang", None)

        detect_kwargs: dict[str, Any] = {}
        if extract_params.page_elements_invoke_url:
            detect_kwargs["invoke_url"] = extract_params.page_elements_invoke_url
        if extract_params.api_key:
            detect_kwargs["api_key"] = extract_params.api_key
        inference_batch_size = getattr(extract_params, "inference_batch_size", None) or getattr(
            tuning, "page_elements_batch_size", None
        )
        if inference_batch_size:
            detect_kwargs["inference_batch_size"] = int(inference_batch_size)
        if _page_elements_stage_needed(extract_params):
            batch_df = self._instantiate_resolved(PageElementDetectionActor, **detect_kwargs).run(batch_df)

        if extract_params.use_table_structure and extract_params.extract_tables:
            table_kwargs: dict[str, Any] = {"ocr_version": ocr_version}
            if ocr_lang is not None:
                table_kwargs["ocr_lang"] = ocr_lang
            if extract_params.table_structure_invoke_url:
                table_kwargs["table_structure_invoke_url"] = extract_params.table_structure_invoke_url
            if extract_params.ocr_invoke_url:
                table_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                table_kwargs["api_key"] = extract_params.api_key
            if extract_params.table_output_format:
                table_kwargs["table_output_format"] = extract_params.table_output_format
            batch_df = self._instantiate_resolved(TableStructureActor, **table_kwargs).run(batch_df)

        ocr_kwargs: dict[str, Any] = {
            "use_table_structure": extract_params.use_table_structure,
            "ocr_version": ocr_version,
        }
        if ocr_lang is not None:
            ocr_kwargs["ocr_lang"] = ocr_lang
        if extract_params.extract_text:
            ocr_kwargs["extract_text"] = True
        if extract_params.extract_tables:
            ocr_kwargs["extract_tables"] = True
        if extract_params.extract_charts:
            ocr_kwargs["extract_charts"] = True
        if extract_params.extract_infographics:
            ocr_kwargs["extract_infographics"] = True
        if extract_params.ocr_invoke_url:
            ocr_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
        if extract_params.api_key:
            ocr_kwargs["api_key"] = extract_params.api_key
        ocr_batch_size = getattr(tuning, "ocr_inference_batch_size", None) or getattr(
            extract_params, "inference_batch_size", None
        )
        if ocr_batch_size:
            ocr_kwargs["inference_batch_size"] = int(ocr_batch_size)
        remote_retry = getattr(extract_params, "remote_retry", None)
        if remote_retry is not None:
            for key in ("remote_max_pool_workers", "remote_max_retries", "remote_max_429_retries"):
                ocr_kwargs[key] = int(getattr(remote_retry, key))

        if any(
            ocr_kwargs.get(key) for key in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
        ):
            ocr_archetype = resolve_ocr_archetype(extract_params)
            batch_df = self._instantiate_resolved(ocr_archetype, **ocr_kwargs).run(batch_df)

        return batch_df

    def _effective_chunk_params(self, key: str) -> Any | None:
        """Return the params used for chunking *key*.

        Priority: an explicit ``TextChunkParams`` instance under
        ``split_config[key]`` wins; otherwise fall back to the legacy
        per-extractor param (``text_params`` for ``"text"`` /
        ``html_params`` for ``"html"``). ``split_config[key] is False``
        and "key absent" both fall through to the legacy fallback —
        ``text``/``html`` files always need an extractor, so "no
        chunking" is not a separate state for those keys.
        """
        cfg = self._split_config.get(key)
        if isinstance(cfg, TextChunkParams):
            return cfg
        if key == "text":
            return self.text_params
        if key == "html":
            return self.html_params
        return None

    def _maybe_chunk(self, df: Any, key: str) -> Any:
        """Append a TextChunkActor pass when chunking is on for *key*."""
        params = self._split_config.get(key)
        if not isinstance(params, TextChunkParams):
            return df
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df
        return TextChunkCPUActor(params=params).run(df)

    def _local_resources(self):
        if self._resolved_resources is None:
            self._resolved_resources = _rrh.gather_local_resources()
        return self._resolved_resources

    def _instantiate_resolved(self, operator_class: type[AbstractOperator], **operator_kwargs: Any) -> AbstractOperator:
        resolved_class = resolve_operator_class(
            operator_class, self._local_resources(), operator_kwargs=operator_kwargs
        )
        if issubclass(operator_class, ArchetypeOperator):
            operator_kwargs = operator_class.variant_operator_kwargs(resolved_class, operator_kwargs)
        return resolved_class(**operator_kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class MultiTypeExtractGPUActor(_MultiTypeExtractBase, GPUOperator):
    """GPU variant – used when local models are available."""

    pass


class MultiTypeExtractCPUActor(_MultiTypeExtractBase, CPUOperator):
    """CPU variant – used when all inference is remote."""

    pass


@designer_component(
    name="Multi-Type Extractor",
    category="Document Processing",
    compute="gpu",
    description="Extracts content from multiple file types (PDF, image, text, audio)",
    category_color="#64b4ff",
)
class MultiTypeExtractOperator(ArchetypeOperator):
    """Graph-facing multi-type extraction archetype."""

    _cpu_variant_class = MultiTypeExtractCPUActor
    _gpu_variant_class = MultiTypeExtractGPUActor

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        kwargs = operator_kwargs or {}
        return not _extract_params_need_local_gpu(
            str(kwargs.get("extraction_mode") or "auto"),
            kwargs.get("extract_params"),
        )

    def __init__(
        self,
        extraction_mode: str = "auto",
        extract_params: ExtractParams | None = None,
        text_params: TextChunkParams | None = None,
        html_params: HtmlChunkParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
        asr_params: ASRParams | None = None,
        caption_params: CaptionParams | None = None,
        video_frame_params: VideoFrameParams | None = None,
        video_text_dedup_params: VideoFrameTextDedupParams | None = None,
        av_fuse_params: AudioVisualFuseParams | None = None,
        split_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
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
            **kwargs,
        )
        self.extraction_mode = extraction_mode
        self.extract_params = extract_params
        self.text_params = text_params
        self.html_params = html_params
        self.audio_chunk_params = audio_chunk_params
        self.asr_params = asr_params
        self.caption_params = caption_params
        self.video_frame_params = video_frame_params
        self.video_text_dedup_params = video_text_dedup_params
        self.av_fuse_params = av_fuse_params
        self.split_config = split_config
