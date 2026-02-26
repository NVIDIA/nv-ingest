# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local ASR using nvidia/parakeet-ctc-1.1b.

Tries Transformers (released/pip) first; on failure falls back to NeMo.
Model expects 16 kHz mono input.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/parakeet-ctc-1.1b"
SAMPLING_RATE = 16000


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


def _load_with_transformers() -> tuple[Optional[object], Optional[str]]:
    """Try loading with Transformers. Returns (pipe_or_model, None) or (None, error_msg)."""
    try:
        from transformers import pipeline

        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device_map="auto",
        )
        return pipe, None
    except Exception as e:
        return None, str(e)


def _load_with_nemo() -> tuple[Optional[object], Optional[str]]:
    """Try loading with NeMo. Returns (model, None) or (None, error_msg)."""
    try:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_ID)
        return model, None
    except Exception as e:
        return None, str(e)


class ParakeetCTC1B1ASR:
    """
    Local ASR using nvidia/parakeet-ctc-1.1b.

    Tries Transformers first; if the installed Transformers version does not
    support this model, falls back to NeMo (requires nemo_toolkit[all]).
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
        self._backend: Literal["transformers", "nemo"] | None = None
        self._pipe = None
        self._nemo_model = None

    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        pipe_or_model, err = _load_with_transformers()
        if pipe_or_model is not None:
            self._pipe = pipe_or_model
            self._backend = "transformers"
            logger.info("ParakeetCTC1B1ASR: using Transformers backend")
            return
        logger.info("Transformers backend failed (%s); trying NeMo.", err)
        nemo_model, nemo_err = _load_with_nemo()
        if nemo_model is not None:
            self._nemo_model = nemo_model
            self._backend = "nemo"
            logger.info("ParakeetCTC1B1ASR: using NeMo backend")
            return
        raise RuntimeError(
            f"Failed to load {MODEL_ID}: Transformers failed ({err}); NeMo failed ({nemo_err}). "
            "For NeMo fallback install: pip install nemo_toolkit[all]"
        )

    @property
    def backend(self) -> Literal["transformers", "nemo"] | None:
        """Backend in use after first transcribe or _ensure_loaded."""
        return self._backend

    def transcribe(self, paths: List[str]) -> List[str]:
        """
        Transcribe one or more audio files to text.

        Each path is loaded and resampled to 16 kHz mono as required by the model.
        Returns one string per path; empty string on load/transcribe failure.
        """
        self._ensure_loaded()
        results: List[str] = []
        for path in paths:
            audio = _load_audio_16k(path)
            if audio is None:
                results.append("")
                continue
            if self._backend == "transformers":
                out = self._transcribe_transformers(audio)
            else:
                out = self._transcribe_nemo(path, audio)
            results.append(out or "")
        return results

    def _transcribe_transformers(self, audio: np.ndarray) -> str:
        if self._pipe is None:
            return ""
        try:
            # pipeline accepts file path or dict with array/sampling_rate
            result = self._pipe(
                {"array": audio, "sampling_rate": SAMPLING_RATE},
            )
            if isinstance(result, dict) and "text" in result:
                return str(result["text"]).strip()
            if isinstance(result, str):
                return result.strip()
            return ""
        except Exception as e:
            logger.warning("Transformers ASR failed: %s", e)
            return ""

    def _transcribe_nemo(self, path: str, audio: np.ndarray) -> str:
        if self._nemo_model is None:
            return ""
        try:
            # NeMo transcribe accepts list of file paths; model expects 16k.
            # We have audio in memory; write to temp wav then transcribe
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            try:
                sf.write(wav_path, audio, SAMPLING_RATE)
                transcript_list = self._nemo_model.transcribe([wav_path])
                if transcript_list and len(transcript_list) > 0:
                    return str(transcript_list[0]).strip()
                return ""
            finally:
                Path(wav_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("NeMo ASR failed for %s: %s", path, e)
            return ""
