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
    kwargs = {}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    model = AutoModelForCTC.from_pretrained(
        model_id, torch_dtype="auto", device_map=device, **kwargs
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
        self._model, self._processor = _load_model_and_processor(
            self._model_id, self._hf_cache_dir
        )
        logger.info(
            "ParakeetCTC1B1ASR: loaded %s via Transformers (AutoModelForCTC + AutoProcessor)",
            self._model_id,
        )

    def transcribe(self, paths: List[str]) -> List[str]:
        """
        Transcribe one or more audio files to text.

        Each path is loaded and resampled to 16 kHz mono as required by the model.
        Returns one string per path; empty string on load/transcribe failure.
        <pad> tokens are removed via skip_special_tokens and/or post-processing.
        """
        self._ensure_loaded()
        results: List[str] = []
        for path in paths:
            audio = _load_audio_16k(path)
            if audio is None:
                results.append("")
                continue
            results.append(self._transcribe_audio(audio) or "")
        return results

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        if self._model is None or self._processor is None:
            return ""
        try:
            import torch

            # Single sample: wrap in list for processor
            speech = [audio]
            inputs = self._processor(
                speech,
                sampling_rate=self._processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self._model.device, dtype=self._model.dtype)
            with torch.no_grad():
                outputs = self._model.generate(**inputs)
            # batch_decode with skip_special_tokens to drop pad tokens
            decoded = self._processor.batch_decode(
                outputs, skip_special_tokens=True
            )
            text = decoded[0] if decoded else ""
            # Fallback: strip any remaining <pad> and normalize spaces
            return _strip_pad_from_transcript(text.strip())
        except Exception as e:
            logger.warning("ASR (transformers) failed: %s", e)
            return ""
