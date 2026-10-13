# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer

from nemo_retriever.common.params import CaptionParams
from nemo_retriever.ingest.plan import (
    AudioSplitTypeValue,
    IngestIndexModeValue,
    IngestProfileValue,
    LocalIngestEmbedBackendValue,
    OcrLangValue,
    OcrVersionValue,
    TableOutputFormatValue,
)
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL

DEFAULT_EMBED_MODEL = NEMOTRON_3_EMBED_MODEL
DEFAULT_CAPTION_MODEL = CaptionParams().model_name

DocumentsArgument = Annotated[
    list[str],
    typer.Argument(help="One or more files, directories, or globs. Supported file types are detected automatically."),
]
ProfileOption = Annotated[
    IngestProfileValue,
    typer.Option(
        "--profile",
        help="Ingest profile: auto or fast-text. Use fast-text for large text-only PDF fallback.",
    ),
]
LanceDbUriOption = Annotated[
    str,
    typer.Option("--lancedb-uri", help="LanceDB database URI."),
]
TableNameOption = Annotated[
    str,
    typer.Option("--table-name", help="LanceDB table name."),
]
DryRunOption = Annotated[
    bool,
    typer.Option("--dry-run", help="Print the resolved ingest plan as JSON without creating an ingestor."),
]
MethodOption = Annotated[
    str | None,
    typer.Option(
        "--method",
        help=(
            "PDF text extraction method: pdfium uses native PDF text only; pdfium_hybrid uses native text "
            "with OCR for scanned pages (the auto-profile default); ocr uses OCR for every page; "
            "nemotron_parse uses the Nemotron Parse visual extraction path."
        ),
    ),
]
DpiOption = Annotated[int | None, typer.Option("--dpi", min=72, help="Render DPI for PDF page images.")]
ExtractTextOption = Annotated[
    bool | None,
    typer.Option("--extract-text/--no-extract-text", help="Enable or disable PDF text extraction."),
]
ExtractImagesOption = Annotated[
    bool | None,
    typer.Option("--extract-images/--no-extract-images", help="Enable or disable PDF image extraction."),
]
ExtractNestedImagesOption = Annotated[
    bool | None,
    typer.Option(
        "--extract-nested-images/--no-extract-nested-images",
        help="Also emit source raster images nested in PDF Form XObjects.",
    ),
]
ExtractTablesOption = Annotated[
    bool | None,
    typer.Option("--extract-tables/--no-extract-tables", help="Enable or disable PDF table extraction."),
]
ExtractChartsOption = Annotated[
    bool | None,
    typer.Option("--extract-charts/--no-extract-charts", help="Enable or disable PDF chart extraction."),
]
ExtractInfographicsOption = Annotated[
    bool | None,
    typer.Option("--extract-infographics", help="Enable PDF infographic extraction."),
]
ExtractPageAsImageOption = Annotated[
    bool | None,
    typer.Option(
        "--extract-page-as-image/--no-extract-page-as-image", help="Enable or disable full-page image extraction."
    ),
]
SegmentAudioOption = Annotated[
    bool | None,
    typer.Option("--segment-audio/--no-segment-audio", help="Enable or disable ASR-side audio segmentation."),
]
AudioSplitTypeOption = Annotated[
    AudioSplitTypeValue,
    typer.Option("--audio-split-type", help="Audio/video audio split type: size, time, or frame."),
]
AudioSplitIntervalOption = Annotated[
    int | None,
    typer.Option("--audio-split-interval", min=1, help="Audio/video audio split interval."),
]
VideoExtractAudioOption = Annotated[
    bool | None,
    typer.Option(
        "--video-extract-audio/--no-video-extract-audio", help="Enable or disable audio extraction from video."
    ),
]
VideoExtractFramesOption = Annotated[
    bool | None,
    typer.Option("--video-extract-frames/--no-video-extract-frames", help="Enable or disable video frame extraction."),
]
VideoFrameFpsOption = Annotated[
    float | None,
    typer.Option("--video-frame-fps", min=0.001, help="Video frame extraction frames per second."),
]
VideoFrameDedupOption = Annotated[
    bool | None,
    typer.Option(
        "--video-frame-dedup/--no-video-frame-dedup", help="Enable or disable perceptual video frame deduplication."
    ),
]
VideoFrameTextDedupOption = Annotated[
    bool | None,
    typer.Option(
        "--video-frame-text-dedup/--no-video-frame-text-dedup",
        help="Enable or disable OCR-text deduplication across adjacent video frames.",
    ),
]
VideoFrameTextDedupMaxDroppedFramesOption = Annotated[
    int | None,
    typer.Option(
        "--video-frame-text-dedup-max-dropped-frames",
        min=0,
        help="Maximum dropped frames bridged by video frame text deduplication.",
    ),
]
VideoAvFuseOption = Annotated[
    bool | None,
    typer.Option("--video-av-fuse/--no-video-av-fuse", help="Enable or disable audio/visual fusion rows for video."),
]
CaptionOption = Annotated[
    bool,
    typer.Option("--caption", help="Add an optional VLM captioning stage after extraction."),
]
CaptionInvokeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--caption-invoke-url",
        help=(
            "VLM caption endpoint URL. If omitted with --caption, GPU hosts use local captioning; "
            "CPU-only runs use the hosted default endpoint with NVIDIA_API_KEY/NGC_API_KEY."
        ),
    ),
]
ApiKeyOption = Annotated[
    str | None,
    typer.Option("--api-key", help="Bearer token for remote extract, embed, and caption endpoints."),
]
CaptionModelNameOption = Annotated[
    str | None,
    typer.Option(
        "--caption-model-name",
        help=(
            f"Optional VLM caption model name override. Defaults to {DEFAULT_CAPTION_MODEL} "
            "when --caption is enabled."
        ),
    ),
]
CaptionGpuMemoryUtilizationOption = Annotated[
    float | None,
    typer.Option(
        "--caption-gpu-memory-utilization",
        min=0.001,
        max=1.0,
        help=(
            "Fraction of GPU memory reserved by local vLLM captioning. "
            "Defaults to the selected local caption model profile."
        ),
    ),
]
CaptionContextTextMaxCharsOption = Annotated[
    int | None,
    typer.Option(
        "--caption-context-text-max-chars",
        min=0,
        help="Maximum nearby extracted text characters to include in caption prompts.",
    ),
]
CaptionInfographicsOption = Annotated[
    bool | None,
    typer.Option(
        "--caption-infographics",
        help="Caption infographic crops in addition to extracted images.",
    ),
]
DedupOption = Annotated[
    bool | None,
    typer.Option(
        "--dedup/--no-dedup",
        help=(
            "Enable or disable image deduplication. When omitted, captioning non-image documents "
            "enables default deduplication."
        ),
    ),
]
DedupIouThresholdOption = Annotated[
    float | None,
    typer.Option(
        "--dedup-iou-threshold",
        min=0.0,
        max=1.0,
        help="Image bounding-box IoU threshold for --dedup. Defaults to 0.45.",
    ),
]
StoreImagesUriOption = Annotated[
    str | None,
    typer.Option("--store-images-uri", help="Store extracted images at this local path or fsspec-compatible URI."),
]
OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite/--append",
        help=(
            "Overwrite the target LanceDB table by default. Use --append to add rows to an existing "
            "table without duplicate checks; rerunning the same inputs in append mode creates duplicates."
        ),
    ),
]
IndexModeOption = Annotated[
    IngestIndexModeValue,
    typer.Option(
        "--index-mode",
        help=(
            "Recommended: leave unset. Auto creates a hybrid table for new indexes and preserves an existing "
            "table on append. Dense, hybrid, and sparse are advanced overrides for experiments or specialized "
            "deployments."
        ),
    ),
]
RayAddressOption = Annotated[
    str | None, typer.Option("--ray-address", help="Batch mode only. Ray address for batch ingest.")
]
RayLogToDriverOption = Annotated[
    bool | None,
    typer.Option(
        "--ray-log-to-driver/--no-ray-log-to-driver", help="Batch mode only. Forward Ray worker logs to the driver."
    ),
]
PageElementsInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--page-elements-invoke-url", help="Page-elements NIM endpoint URL."),
]
OcrInvokeUrlOption = Annotated[str | None, typer.Option("--ocr-invoke-url", help="OCR NIM endpoint URL.")]
OcrVersionOption = Annotated[
    OcrVersionValue | None,
    typer.Option("--ocr-version", help="OCR engine version for extraction."),
]
OcrLangOption = Annotated[
    OcrLangValue | None,
    typer.Option("--ocr-lang", help="OCR v2 language selector for local extraction."),
]
TableStructureInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--table-structure-invoke-url", help="Table-structure NIM endpoint URL."),
]
TableOutputFormatOption = Annotated[
    TableOutputFormatValue | None,
    typer.Option(
        "--table-output-format", help="Table text format. 'markdown' enables local table-structure extraction."
    ),
]
EmbedInvokeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--embed-invoke-url",
        envvar="EMBED_INVOKE_URL",
        help=(
            "Embedding endpoint override. On CPU-only hosts, ingest automatically uses NVIDIA's hosted "
            "embedding endpoint with NVIDIA_API_KEY or NGC_API_KEY; pass this only for another endpoint."
        ),
    ),
]
EmbedModelNameOption = Annotated[
    str | None,
    typer.Option(
        "--embed-model-name",
        envvar="EMBED_MODEL_NAME",
        help=(
            "Optional embedding model override. Local inference defaults to Nemotron 3.5 Embed 1B "
            f"(NVFP4 on supported vLLM GPUs, BF16 otherwise); remote defaults to {DEFAULT_EMBED_MODEL}."
        ),
    ),
]
EmbedModelProviderPrefixOption = Annotated[
    str | None,
    typer.Option(
        "--embed-model-provider-prefix",
        envvar="EMBED_MODEL_PROVIDER_PREFIX",
        help="Optional LiteLLM provider prefix prepended to the remote embedding model name.",
    ),
]
LocalIngestEmbedBackendOption = Annotated[
    LocalIngestEmbedBackendValue | None,
    typer.Option(
        "--local-ingest-embed-backend", help="Local ingest-time text embedder when --embed-invoke-url is unset."
    ),
]
EmbedModalityOption = Annotated[
    str | None,
    typer.Option("--embed-modality", help="Embedding modality for emitted rows: text, image, or text_image."),
]
EmbedGranularityOption = Annotated[
    str | None,
    typer.Option("--embed-granularity", help="Embedding granularity: element or page."),
]
TextElementsModalityOption = Annotated[
    str | None,
    typer.Option("--text-elements-modality", help="Embedding modality override for page text elements."),
]
StructuredElementsModalityOption = Annotated[
    str | None,
    typer.Option(
        "--structured-elements-modality", help="Embedding modality override for table/chart/infographic elements."
    ),
]
TextChunkOption = Annotated[
    bool,
    typer.Option(
        "--text-chunk", help="Enable token chunking during extraction. Defaults to 1024 tokens with 150-token overlap."
    ),
]
TextChunkMaxTokensOption = Annotated[
    int | None,
    typer.Option("--text-chunk-max-tokens", min=1, help="Maximum tokens per text chunk. Implies --text-chunk."),
]
TextChunkOverlapTokensOption = Annotated[
    int | None,
    typer.Option(
        "--text-chunk-overlap-tokens", min=0, help="Token overlap between adjacent text chunks. Implies --text-chunk."
    ),
]
PdfSplitBatchSizeOption = Annotated[
    int | None,
    typer.Option("--pdf-split-batch-size", min=1, help="Batch mode only. PDF split batch size."),
]
PdfExtractWorkersOption = Annotated[
    int | None,
    typer.Option("--pdf-extract-workers", min=1, help="Batch mode only. Maximum Ray tasks for PDF extraction."),
]
PdfExtractBatchSizeOption = Annotated[
    int | None,
    typer.Option("--pdf-extract-batch-size", min=1, help="Batch mode only. PDF extraction batch size per Ray task."),
]
PdfExtractCpusPerTaskOption = Annotated[
    float | None,
    typer.Option(
        "--pdf-extract-cpus-per-task", min=0.0, help="Batch mode only. CPUs reserved per PDF extraction Ray task."
    ),
]
PageElementsWorkersOption = Annotated[
    int | None,
    typer.Option(
        "--page-elements-workers", min=1, help="Batch mode only. Number of Ray actors for page-element detection."
    ),
]
PageElementsBatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--page-elements-batch-size", min=1, help="Batch mode only. Page-element detection batch size per actor."
    ),
]
PageElementsCpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--page-elements-cpus-per-actor",
        min=0.0,
        help="Batch mode only. CPUs reserved per page-element detection actor.",
    ),
]
PageElementsGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--page-elements-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local page-element detection actor.",
    ),
]
OcrWorkersOption = Annotated[
    int | None,
    typer.Option("--ocr-workers", min=1, help="Batch mode only. Number of Ray actors for OCR inference."),
]
OcrBatchSizeOption = Annotated[
    int | None,
    typer.Option("--ocr-batch-size", min=1, help="Batch mode only. OCR inference batch size per actor."),
]
OcrCpusPerActorOption = Annotated[
    float | None,
    typer.Option("--ocr-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per OCR actor."),
]
OcrGpusPerActorOption = Annotated[
    float | None,
    typer.Option("--ocr-gpus-per-actor", min=0.0, help="Batch mode only. GPUs reserved per local OCR actor."),
]
TableStructureWorkersOption = Annotated[
    int | None,
    typer.Option(
        "--table-structure-workers", min=1, help="Batch mode only. Number of Ray actors for table-structure extraction."
    ),
]
TableStructureBatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--table-structure-batch-size", min=1, help="Batch mode only. Table-structure extraction batch size per actor."
    ),
]
TableStructureCpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--table-structure-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per table-structure actor."
    ),
]
TableStructureGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--table-structure-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local table-structure actor.",
    ),
]
NemotronParseWorkersOption = Annotated[
    int | None,
    typer.Option("--nemotron-parse-workers", min=1, help="Batch mode only. Number of Ray actors for Nemotron Parse."),
]
NemotronParseBatchSizeOption = Annotated[
    int | None,
    typer.Option("--nemotron-parse-batch-size", min=1, help="Batch mode only. Nemotron Parse batch size per actor."),
]
NemotronParseGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--nemotron-parse-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local Nemotron Parse actor.",
    ),
]
EmbedWorkersOption = Annotated[
    int | None,
    typer.Option("--embed-workers", min=1, help="Batch mode only. Number of Ray actors for embedding."),
]
EmbedBatchSizeOption = Annotated[
    int | None,
    typer.Option("--embed-batch-size", min=1, help="Batch mode only. Embedding batch size per actor."),
]
EmbedCpusPerActorOption = Annotated[
    float | None,
    typer.Option("--embed-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per embedding actor."),
]
EmbedGpusPerActorOption = Annotated[
    float | None,
    typer.Option("--embed-gpus-per-actor", min=0.0, help="Batch mode only. GPUs reserved per local embedding actor."),
]
QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet/--no-quiet",
        help=(
            "Suppress verbose progress output (progress bars, HuggingFace "
            "downloads, vLLM init logs). On success, prints only the final "
            "summary line. On error, flushes all captured output to stderr "
            "for debugging. Enabled by default; pass --no-quiet for the full "
            "verbose output."
        ),
    ),
]
