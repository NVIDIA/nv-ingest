# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Audio extraction stage: chunk + ASR only, write *.audio_extraction.json sidecars.

Provides an extraction-only CLI analogous to `retriever pdf stage page-elements`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer

from retriever.audio.asr_actor import apply_asr_to_df
from retriever.audio.asr_actor import asr_params_from_env
from retriever.audio.chunk_actor import audio_path_to_chunks_df
from retriever.audio.media_interface import is_media_available
from retriever.params import ASRParams
from retriever.params import AudioChunkParams

logger = logging.getLogger(__name__)

app = typer.Typer(help="Audio extraction stage (chunk + ASR, write sidecar JSON).")

# Default globs for discovering audio/video files (ffmpeg-readable).
DEFAULT_AUDIO_GLOBS = ["*.wav", "*.mp3", "*.m4a", "*.mp4", "*.mov", "*.avi", "*.mkv"]


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically write JSON (write temp then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
    tmp_path.replace(path)


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable builtins."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass
    return str(obj)


def _audio_extraction_json_path(source_path: Path, output_dir: Optional[Path]) -> Path:
    """Sidecar path: <source>.audio_extraction.json beside file or under output_dir."""
    sidecar = source_path.with_name(source_path.name + ".audio_extraction.json")
    if output_dir is not None:
        return output_dir / sidecar.name
    return sidecar


@app.command("discover")
def discover(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan for audio/video files.",
    ),
    glob: str = typer.Option(
        "*.mp3,*.wav",
        "--glob",
        help="Comma-separated glob pattern(s) for discovery.",
    ),
) -> None:
    """List files that would be processed and where sidecars would be written (no ASR). Use to verify mount and paths."""
    patterns = [g.strip() for g in glob.split(",") if g.strip()] or ["*.mp3", "*.wav"]
    paths_set: set[Path] = set()
    for pat in patterns:
        for p in input_dir.rglob(pat):
            if p.is_file():
                paths_set.add(p)
    paths = sorted(paths_set)
    if not paths:
        print(f"No files matching {patterns} under {input_dir}", flush=True)
        raise typer.Exit(code=2)
    print(f"Would process {len(paths)} file(s). Sidecar paths:", flush=True)
    for p in paths:
        out = _audio_extraction_json_path(p, None)
        print(f"  {p} -> {out}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _run_extract_one(
    source_path: str,
    chunk_params: AudioChunkParams,
    asr_params: ASRParams,
) -> pd.DataFrame:
    """Chunk one file and run ASR; return DataFrame with path, source_path, duration, chunk_index, text, metadata."""
    chunk_df = audio_path_to_chunks_df(source_path, params=chunk_params)
    if chunk_df.empty:
        return chunk_df
    asr_kw = asr_params.model_dump(mode="python")
    return apply_asr_to_df(chunk_df, asr_params=asr_kw)


@app.command("extract")
def extract(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to scan for audio/video files.",
    ),
    glob: str = typer.Option(
        "*.mp3,*.wav",
        "--glob",
        help="Comma-separated glob(s) for discovery (e.g. '*.mp3', '*.wav'). Default: *.mp3,*.wav.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory to write sidecar JSONs (default: next to each source file).",
    ),
    split_type: str = typer.Option(
        "size",
        "--split-type",
        help="Chunk split type: 'size', 'time', or 'frame'.",
    ),
    split_interval: int = typer.Option(
        450,
        "--split-interval",
        min=1,
        help="Chunk split interval (bytes for size, seconds for time, frames for frame).",
    ),
    audio_only: bool = typer.Option(
        False,
        "--audio-only/--no-audio-only",
        help="If true and file is video, extract audio to MP3 then chunk.",
    ),
    video_audio_separate: bool = typer.Option(
        False,
        "--video-audio-separate/--no-video-audio-separate",
        help="If true and video, also add extracted MP3 as separate item.",
    ),
    use_env_asr: bool = typer.Option(
        True,
        "--use-env-asr/--no-use-env-asr",
        help="Build ASR params from AUDIO_GRPC_ENDPOINT, NGC_API_KEY, AUDIO_FUNCTION_ID when set.",
    ),
    audio_grpc_endpoint: Optional[str] = typer.Option(
        None,
        "--audio-grpc-endpoint",
        help="Override gRPC endpoint for ASR (else from env or local Parakeet).",
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help="Override auth token for cloud ASR.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="Limit number of files processed.",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write one <file>.audio_extraction.json sidecar per source file.",
    ),
) -> None:
    """
    Scan input_dir for audio/video files, run chunk + ASR, and write extraction JSON sidecars.

    Uses local Parakeet when no ASR endpoint is set; use NGC_API_KEY + AUDIO_FUNCTION_ID
    (or --audio-grpc-endpoint) for cloud ASR.
    """
    print(f"Audio stage extract: input_dir={input_dir!s} glob={glob!r} output_dir={output_dir!s}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    if not is_media_available():
        raise typer.BadParameter(
            "Audio stage requires ffmpeg. Install system ffmpeg and ensure it is on PATH."
        )

    if split_type not in ("size", "time", "frame"):
        raise typer.BadParameter("--split-type must be one of: size, time, frame")
    chunk_params = AudioChunkParams(
        split_type=split_type,
        split_interval=split_interval,
        audio_only=audio_only,
        video_audio_separate=video_audio_separate,
    )

    if use_env_asr:
        asr_params = asr_params_from_env()
        if audio_grpc_endpoint is not None:
            asr_params = asr_params.model_copy(
                update={"audio_endpoints": (audio_grpc_endpoint, asr_params.audio_endpoints[1])}
            )
        if auth_token is not None:
            asr_params = asr_params.model_copy(update={"auth_token": auth_token})
    else:
        asr_params = ASRParams(
            audio_endpoints=(audio_grpc_endpoint or "", None),
            auth_token=auth_token,
        )

    # Discover files: comma-separated globs (e.g. *.mp3,*.wav).
    patterns = [g.strip() for g in glob.split(",") if g.strip()] or ["*.mp3", "*.wav"]
    paths_set: set[Path] = set()
    for pat in patterns:
        for p in input_dir.rglob(pat):
            if p.is_file():
                paths_set.add(p)
    paths = sorted(paths_set)
    if not paths:
        typer.echo(f"No files matching {patterns} under {input_dir}", err=True)
        sys.stderr.flush()
        raise typer.Exit(code=2)

    asr_mode = "remote" if (asr_params.audio_endpoints[0] or "").strip() else "local (Parakeet)"
    typer.echo(f"Found {len(paths)} file(s) matching {patterns}. ASR: {asr_mode}.", err=True)
    sys.stderr.flush()

    if limit is not None:
        paths = paths[: int(limit)]

    written: List[Path] = []
    for i, p in enumerate(paths):
        try:
            df = _run_extract_one(str(p), chunk_params, asr_params)
        except Exception as e:
            logger.exception("Extraction failed for %s: %s", p, e)
            typer.echo(f"Extraction failed for {p}: {e}", err=True)
            sys.stderr.flush()
            continue

        if write_json:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "path": str(row.get("path")),
                    "source_path": str(row.get("source_path", p)),
                    "duration": _to_jsonable(row.get("duration")),
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "text": str(row.get("text", "")),
                    "metadata": _to_jsonable(row.get("metadata")),
                })
            out_path = _audio_extraction_json_path(p, output_dir)
            payload: Dict[str, Any] = {
                "schema_version": 1,
                "stage": "audio_extract",
                "source_path": str(p.resolve()),
                "chunks": records,
                "num_chunks": len(records),
            }
            try:
                _atomic_write_json(out_path, payload)
                written.append(out_path)
                typer.echo(f"Wrote {out_path}", err=True)
                sys.stderr.flush()
            except OSError as write_err:
                typer.echo(f"Failed to write {out_path}: {write_err}", err=True)
                sys.stderr.flush()
                logger.exception("Write failed for %s: %s", out_path, write_err)

    print(f"Processed {len(paths)} file(s), wrote {len(written)} JSON sidecar(s).", flush=True)
    sys.stdout.flush()
