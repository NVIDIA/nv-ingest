# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local ASR using nvidia/parakeet-ctc-1.1b via Hugging Face Transformers.

Greedy CTC decoding with word-level timestamps and silence-gap segments so the
``segment_audio`` flow has the time-aligned rows ``recall_match_mode:
audio_segment`` needs. Internal time-windowed chunking keeps each forward pass
under the encoder's ``max_position_embeddings = 5000`` ceiling (≈ 400 s).
Requires transformers>=4.57; expects 16 kHz mono input.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nvtx

from nemo_retriever.models.hf_cache import configure_global_hf_cache_base
from nemo_retriever.models.hf_model_registry import get_hf_revision


logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/parakeet-ctc-1.1b"
SAMPLING_RATE = 16000
# ParakeetEncoderConfig.subsampling_factor for parakeet-ctc-1.1b. Hardcoding it
# avoids calling the model's private ``_get_subsampling_output_length`` which
# can disappear across transformers versions and would silently yield empty
# transcripts (the broad ``except Exception`` upstream swallows the AttributeError).
SUBSAMPLING_FACTOR = 8
# subsampling_factor * mel hop_length (160) / sampling_rate (16000) = 0.08 s
FRAME_STRIDE_SECS = 0.08
SAMPLES_PER_FRAME = 160 * SUBSAMPLING_FACTOR
DEFAULT_SEGMENT_GAP_SECS = 0.5
DEFAULT_MAX_SEGMENT_SECS = 12.0
# Stay well under the encoder's max_position_embeddings = 5000 ceiling (≈ 400 s).
DEFAULT_MAX_CHUNK_SECS = 240.0
MAX_CHUNKS_PER_FORWARD = 4


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def _strip_pad_from_transcript(text: str) -> str:
    """Remove <pad> tokens and normalize spaces (fallback when decode doesn't skip them)."""
    if not text:
        return ""
    t = text.replace("<pad>", "").strip()
    return " ".join(t.split()) if t else ""


def _load_audio_16k(path: str) -> Optional[np.ndarray]:
    """Load audio from path and return mono 16 kHz float32 array, or None on failure."""
    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile not installed; cannot load audio for local ASR.")
        return None

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        # Unsupported format (e.g. mp3); try ffmpeg to 16k mono wav
        try:
            import subprocess

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        path,
                        "-ar",
                        str(SAMPLING_RATE),
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        wav_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                data, sr = sf.read(wav_path, dtype="float32")
            finally:
                Path(wav_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to load or convert audio %s: %s", path, e)
            return None

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLING_RATE:
        from scipy.signal import resample

        n = int(len(data) * SAMPLING_RATE / sr)
        data = resample(data, n).astype(np.float32)
    return data


def _load_model_and_processor(model_id: str, hf_cache_dir: Optional[str] = None):
    """Load Parakeet ASR via AutoModelForCTC + AutoProcessor (explicit decode control)."""
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_cache_dir = configure_global_hf_cache_base(hf_cache_dir)
    _revision = get_hf_revision(model_id)
    kwargs = {}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    processor = AutoProcessor.from_pretrained(model_id, revision=_revision, **kwargs)
    # bfloat16 on GPU: ~2× faster forward and ~half the activation/weight memory vs fp32,
    # which lets Ray Data co-schedule more ASR actors per GPU. Float32 on CPU since
    # bf16 on x86 CPU silently downcasts via emulation and tanks throughput.
    model_dtype = "bfloat16" if device == "cuda" else "auto"
    model = AutoModelForCTC.from_pretrained(
        model_id, revision=_revision, torch_dtype=model_dtype, device_map=device, **kwargs
    )
    return model, processor


def _greedy_word_alignments(frame_ids: np.ndarray, tokenizer) -> List[Tuple[str, Tuple[int, int]]]:
    """Convert frame-level argmax IDs into word-level ``(text, (start_frame, end_frame))`` tuples.

    SentencePiece tokens prefixed with ``▁`` (or the ``<unk>`` token) start a new word.
    """
    if frame_ids.ndim != 1:
        raise ValueError("frame_ids must be 1D")
    pad_id = tokenizer.pad_token_id

    # Collapse consecutive duplicates, drop blanks; track frame ranges per surviving token.
    surviving: List[Tuple[int, int, int]] = []  # (token_id, start_frame, end_frame)
    prev: Optional[int] = None
    run_start = 0
    ids = frame_ids.tolist()
    for i, tid in enumerate(ids):
        if tid == prev:
            continue
        if prev is not None and prev != pad_id:
            surviving.append((prev, run_start, i - 1))
        prev = tid
        run_start = i
    if prev is not None and prev != pad_id:
        surviving.append((prev, run_start, len(ids) - 1))

    if not surviving:
        return []

    # Batch the id→token lookup; ParakeetTokenizerFast.convert_ids_to_tokens accepts a list.
    token_strs = tokenizer.convert_ids_to_tokens([t for t, _, _ in surviving])
    unk_token = tokenizer.unk_token

    words: List[Tuple[str, Tuple[int, int]]] = []
    cur_text = ""
    cur_start_frame = 0
    cur_end_frame = 0
    for token_str, (_, sf, ef) in zip(token_strs, surviving):
        token_str = token_str or ""
        starts_word = token_str.startswith("▁") or token_str == unk_token
        if starts_word and cur_text:
            words.append((cur_text, (cur_start_frame, cur_end_frame)))
            cur_text = ""
        if not cur_text:
            cur_start_frame = sf
            cur_text = token_str.lstrip("▁")
        else:
            cur_text += token_str
        cur_end_frame = ef
    if cur_text:
        words.append((cur_text, (cur_start_frame, cur_end_frame)))
    return [(w.strip(), fr) for w, fr in words if w.strip()]


def _make_segment(words: List[str], start: float, end: float) -> Dict[str, object]:
    return {"start": float(start), "end": float(end), "text": " ".join(words).strip()}


def _segment_words(
    words: List[Tuple[str, Tuple[int, int]]],
    *,
    frame_stride_secs: float = FRAME_STRIDE_SECS,
    gap_secs: Optional[float] = None,
    max_secs: Optional[float] = None,
) -> List[Dict[str, object]]:
    """Group word-level alignments into segments split by silence gaps and capped by duration."""
    if gap_secs is None:
        gap_secs = _env_float("RETRIEVER_PARAKEET_SEGMENT_GAP_SECS", DEFAULT_SEGMENT_GAP_SECS)
    if max_secs is None:
        max_secs = _env_float("RETRIEVER_PARAKEET_MAX_SEGMENT_SECS", DEFAULT_MAX_SEGMENT_SECS)
    if not words:
        return []
    segments: List[Dict[str, object]] = []
    cur_words: List[str] = []
    cur_start = 0.0
    cur_end = 0.0
    last_end: Optional[float] = None
    for word, (frame_start, frame_end) in words:
        word_start = frame_start * frame_stride_secs
        word_end = (frame_end + 1) * frame_stride_secs
        gap = (word_start - last_end) if last_end is not None else 0.0
        too_long = cur_words and (word_end - cur_start) > max_secs
        if cur_words and (gap > gap_secs or too_long):
            segments.append(_make_segment(cur_words, cur_start, cur_end))
            cur_words = []
        if not cur_words:
            cur_start = word_start
        cur_words.append(word)
        cur_end = word_end
        last_end = word_end
    if cur_words:
        segments.append(_make_segment(cur_words, cur_start, cur_end))
    return segments


class ParakeetCTC1B1ASR:
    """
    Local ASR using nvidia/parakeet-ctc-1.1b via Hugging Face Transformers.

    Greedy CTC decoding with word-level timestamps and silence-gap segmentation
    so the audio_segment recall mode can match per-segment hits. Internal time
    chunking keeps each forward pass under the encoder's max_position_embeddings
    ceiling (~6.7 minutes); long audio is sliced and stitched.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._device = device
        self._hf_cache_dir = hf_cache_dir
        self._model_id = model_id or MODEL_ID
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        self._model, self._processor = _load_model_and_processor(self._model_id, self._hf_cache_dir)
        logger.info("ParakeetCTC1B1ASR: loaded %s", self._model_id)

    def transcribe(self, paths: List[str]) -> List[str]:
        """Transcribe one or more audio files (batched). Returns one string per path."""
        return [text for text, _ in self.transcribe_with_segments(paths)]

    def transcribe_with_segments(self, paths: List[str]) -> List[Tuple[str, List[Dict[str, object]]]]:
        """Transcribe and also return per-path silence-gap segments with timestamps."""
        self._ensure_loaded()
        audios: List[Optional[np.ndarray]] = [_load_audio_16k(p) for p in paths]
        return self._transcribe_audio_arrays(audios)

    def transcribe_audios(self, audios: List[Optional[np.ndarray]]) -> List[str]:
        """Transcribe a batch of pre-loaded 16 kHz mono float32 arrays."""
        return [text for text, _ in self._transcribe_audio_arrays(audios)]

    def _split_into_chunks(self, audio: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Non-overlapping windows under max_position_embeddings; ``(sample_offset, chunk)``."""
        chunk_secs = _env_float("RETRIEVER_PARAKEET_MAX_CHUNK_SECS", DEFAULT_MAX_CHUNK_SECS)
        chunk_len = max(1, int(chunk_secs * SAMPLING_RATE))
        min_tail = int(0.1 * SAMPLING_RATE)  # zero-frame logits break the decoder
        chunks: List[Tuple[int, np.ndarray]] = []
        for start in range(0, len(audio), chunk_len):
            end = min(len(audio), start + chunk_len)
            if end - start < min_tail and chunks:
                continue
            chunks.append((start, audio[start:end]))
        if not chunks:
            chunks.append((0, audio))
        return chunks

    def _transcribe_audio_arrays(self, audios: List[Optional[np.ndarray]]) -> List[Tuple[str, List[Dict[str, object]]]]:
        self._ensure_loaded()
        flat: List[Tuple[int, int, np.ndarray]] = []  # (audio_idx, sample_offset, chunk)
        valid_indices: List[int] = []
        for i, audio in enumerate(audios):
            if audio is None or audio.size == 0:
                continue
            valid_indices.append(i)
            for sample_offset, chunk in self._split_into_chunks(audio):
                flat.append((i, sample_offset, chunk))

        results: List[Tuple[str, List[Dict[str, object]]]] = [("", []) for _ in audios]
        if not flat:
            return results

        per_audio_words: Dict[int, List[Tuple[str, Tuple[int, int]]]] = {i: [] for i in valid_indices}
        per_audio_text: Dict[int, List[str]] = {i: [] for i in valid_indices}

        for batch_start in range(0, len(flat), MAX_CHUNKS_PER_FORWARD):
            batch = flat[batch_start : batch_start + MAX_CHUNKS_PER_FORWARD]
            try:
                decoded = self._decode_batch([c for _, _, c in batch])
            except Exception as e:
                logger.warning(
                    "ASR (transformers) chunk forward failed (chunks %d-%d): %s",
                    batch_start,
                    batch_start + len(batch) - 1,
                    e,
                    exc_info=True,
                )
                continue
            for (audio_idx, sample_offset, _), (text, alignments) in zip(batch, decoded):
                frame_offset = sample_offset // SAMPLES_PER_FRAME
                for word, (fs, fe) in alignments:
                    per_audio_words[audio_idx].append((word, (fs + frame_offset, fe + frame_offset)))
                if text.strip():
                    per_audio_text[audio_idx].append(text.strip())

        for i in valid_indices:
            text = _strip_pad_from_transcript(" ".join(per_audio_text[i]))
            results[i] = (text, _segment_words(per_audio_words[i]))
        return results

    def _decode_batch(self, audios: List[np.ndarray]) -> List[Tuple[str, List[Tuple[str, Tuple[int, int]]]]]:
        """Forward pass + greedy CTC for a batch of chunks. Returns ``(text, word_alignments)``."""
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError(
                "ParakeetCTC1B1ASR._decode_batch called before the model was loaded; " "call _ensure_loaded() first."
            )
        inputs = self._processor(
            audios,
            sampling_rate=self._processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        with torch.no_grad(), nvtx.annotate("gpu_inference", color="blue"), nvtx.annotate(
            f"ParakeetCTC1B | bs={len(audios)}", color="blue"
        ):
            outputs = self._model(**inputs)
        logits = outputs.logits  # [B, T, V]

        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            output_lengths = attn_mask.sum(dim=-1) // SUBSAMPLING_FACTOR
        else:
            output_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)

        argmax_ids = logits.argmax(dim=-1).cpu().numpy()
        lengths = output_lengths.cpu().tolist()
        tokenizer = self._processor.tokenizer
        results: List[Tuple[str, List[Tuple[str, Tuple[int, int]]]]] = []
        for i in range(argmax_ids.shape[0]):
            ids = argmax_ids[i, : int(lengths[i])]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
            words = _greedy_word_alignments(ids, tokenizer)
            results.append((text, list(words)))
        return results
