# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local ASR using nvidia/parakeet-ctc-1.1b via Hugging Face Transformers.

Uses AutoModelForCTC + AutoProcessor with batch_decode(skip_special_tokens=True)
to avoid <pad> tokens in output; falls back to post-processing to strip any remaining.
Requires transformers>=5. Model expects 16 kHz mono input.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/parakeet-ctc-1.1b"
SAMPLING_RATE = 16000


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
    model = AutoModelForCTC.from_pretrained(
        model_id, revision=_revision, torch_dtype="auto", device_map=device, **kwargs
    )
    return model, processor


class ParakeetCTC1B1ASR:
    """
    Local ASR using nvidia/parakeet-ctc-1.1b via Hugging Face Transformers.

    Uses AutoModelForCTC + AutoProcessor with batch_decode(skip_special_tokens=True)
    and post-processes to remove any remaining <pad> tokens. Requires transformers>=5.
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
        logger.info(
            "ParakeetCTC1B1ASR: loaded %s via Transformers (AutoModelForCTC + AutoProcessor)",
            self._model_id,
        )

    def transcribe(self, paths: List[str]) -> List[str]:
        """
        Transcribe one or more audio files to text (batched inference).

        Each path is loaded and resampled to 16 kHz mono; then all are processed
        in a single model forward pass. Returns one string per path; empty string
        on load/transcribe failure.
        <pad> tokens are removed via skip_special_tokens and/or post-processing.
        """
        self._ensure_loaded()
        audios: List[Optional[np.ndarray]] = []
        for path in paths:
            audio = _load_audio_16k(path)
            audios.append(audio)
        return self.transcribe_audios(audios)

    def transcribe_audios(self, audios: List[Optional[np.ndarray]]) -> List[str]:
        """
        Transcribe a batch of audio arrays (16 kHz mono float32) in one forward pass.

        Each element can be np.ndarray or None; None yields "" in the output.
        Returns one string per input; empty string for None or on failure.
        """
        self._ensure_loaded()
        valid: List[np.ndarray] = []
        indices: List[int] = []
        for i, audio in enumerate(audios):
            if audio is not None and audio.size > 0:
                valid.append(audio)
                indices.append(i)
        if not valid:
            return [""] * len(audios)
        try:
            transcripts = self._transcribe_audio_batch(valid)
        except Exception as e:
            logger.warning("ASR (transformers) batch failed: %s", e)
            transcripts = [""] * len(valid)
        # Map back to original order; empty string for missing/failed
        result = [""] * len(audios)
        for idx, text in zip(indices, transcripts):
            result[idx] = _strip_pad_from_transcript((text or "").strip())
        return result

    def _transcribe_audio_batch(self, audios: List[np.ndarray]) -> List[str]:
        """Single forward pass for a list of audio arrays; returns one string per array."""
        if self._model is None or self._processor is None or not audios:
            return [""] * len(audios)
        try:
            import torch

            inputs = self._processor(
                audios,
                sampling_rate=self._processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self._model.device, dtype=self._model.dtype)
            with torch.no_grad():
                outputs = self._model.generate(**inputs)
            decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
            return [t.strip() for t in decoded]
        except Exception as e:
            logger.warning("ASR (transformers) batch failed: %s", e)
            return [""] * len(audios)

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Single-sample path for API compatibility; delegates to batch."""
        results = self._transcribe_audio_batch([audio])
        return results[0] if results else ""
