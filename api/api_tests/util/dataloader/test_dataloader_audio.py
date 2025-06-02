# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
import wave
import os
import tempfile
from pathlib import Path
import shutil
from nv_ingest_api.util.dataloader import DataLoader, MediaInterface
import subprocess
import json
import math


test_file_size_mb = 100


def create_test_wav(filepath, duration_seconds=6.0, sample_rate=44100, file_size_mb=test_file_size_mb):
    """
    Create a test WAV file with a simple sine wave.
    The file size is controlled by adjusting the duration and sample rate.

    Args:
        filepath: Path where to save the WAV file
        duration_seconds: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        file_size_mb: Target file size in MB (approximate)
    """
    # Calculate number of channels needed to achieve target file size
    # Each sample is 2 bytes (16-bit audio)
    bytes_per_second = sample_rate * 2  # mono audio
    total_seconds_needed = (file_size_mb * 1024 * 1024) / bytes_per_second

    # Generate time array
    t = np.linspace(0, total_seconds_needed, int(sample_rate * total_seconds_needed))

    # Create a 440 Hz sine wave
    frequency = 440
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Ensure the data is in the correct format
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create WAV file
    with wave.open(str(filepath), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return audio_data, sample_rate


def create_test_mp3(filepath, duration_seconds=6.0, sample_rate=44100, target_size_mb=600):
    """
    Create a test MP3 file with a simple sine wave.
    Uses ffmpeg to create an MP3 with a specific bitrate to achieve target file size.

    Args:
        filepath: Path where to save the MP3 file
        duration_seconds: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
        target_size_mb: Target file size in MB (approximate)
    """
    # Create temporary WAV file first
    temp_wav = str(filepath) + ".temp.wav"
    audio_data, _ = create_test_wav(temp_wav, duration_seconds, sample_rate, target_size_mb)

    # Calculate required bitrate to achieve target size
    # MP3 file size = (bitrate * duration) / 8
    # Solve for bitrate: bitrate = (target_size * 8) / duration
    target_size_bits = target_size_mb * 1024 * 1024 * 8
    bitrate = int(target_size_bits / duration_seconds)

    try:
        # Convert WAV to MP3 with calculated bitrate
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_wav, "-b:a", f"{bitrate}", "-codec:a", "libmp3lame", str(filepath)],
            check=True,
            capture_output=True,
        )
    finally:
        # Clean up temporary WAV file
        os.remove(temp_wav)

    return filepath


def get_audio_info(filepath):
    """Get audio file information using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(filepath)],
        capture_output=True,
        text=True,
    )

    return json.loads(result.stdout)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_dataloader_wav_chunking(temp_dir):
    """Test that DataLoader correctly splits WAV files based on size."""
    # Create a WAV file that's 600MB (larger than default 450MB split size)
    input_wav = temp_dir / "large_input.wav"
    original_audio, sample_rate = create_test_wav(input_wav, file_size_mb=test_file_size_mb)  # 600MB file

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    actual_size_mb = input_wav.stat().st_size * 1e-6

    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    loader = DataLoader(
        paths=[str(input_wav)], output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Collect all chunks
    chunks = []
    for chunk in loader:
        chunks.append(chunk)

    # Verify we got the expected number of chunks (600MB/200MB = 3 chunks)
    assert len(chunks) == 3, f"Expected 3 chunks, but got {len(chunks)}"

    # Verify each chunk is approximately the right size (200MB ± 10%)
    expected_chunk_size = split_size_mb * 1024 * 1024  # Convert MB to bytes
    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        # Allow for some variation in chunk size (±10%)
        assert (
            0.9 * expected_chunk_size <= chunk_size <= 1.1 * expected_chunk_size
        ), f"Chunk {i} size ({chunk_size} bytes) is not within 10% of expected size ({expected_chunk_size} bytes)"


def test_dataloader_wav_content(temp_dir):
    """Test that the content of chunks from DataLoader is valid WAV data."""
    # Create a WAV file that's 600MB
    input_wav = temp_dir / "large_input.wav"
    original_audio, sample_rate = create_test_wav(input_wav, file_size_mb=test_file_size_mb)

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    actual_size_mb = input_wav.stat().st_size / (1024 * 1024)
    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    loader = DataLoader(
        paths=[str(input_wav)], output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Test that each chunk can be read as valid WAV data
    for i, chunk_data in enumerate(loader):
        # Write chunk to temporary file
        chunk_path = chunks_dir / f"test_chunk_{i}.wav"
        with open(chunk_path, "wb") as f:
            f.write(chunk_data)

        # Verify the chunk is valid WAV data
        with wave.open(str(chunk_path), "rb") as wav_file:
            # Check WAV parameters
            assert wav_file.getnchannels() == 1, f"Chunk {i} should be mono"
            assert wav_file.getframerate() == sample_rate, f"Chunk {i} should have correct sample rate"
            assert wav_file.getsampwidth() == 2, f"Chunk {i} should be 16-bit audio"

            # Read some frames to ensure it's valid audio data
            frames = wav_file.readframes(1000)
            assert len(frames) > 0, f"Chunk {i} should contain audio data"


def test_dataloader_mp3_chunking(temp_dir):
    """Test that DataLoader correctly splits MP3 files based on size."""
    # Create an MP3 file that's 600MB
    input_mp3 = temp_dir / "large_input.mp3"
    create_test_mp3(input_mp3, duration_seconds=600, target_size_mb=test_file_size_mb)  # 10 minutes  # 600MB file

    # Verify the input file was created and is roughly the right size
    assert input_mp3.exists(), "Input MP3 file was not created"
    actual_size_mb = input_mp3.stat().st_size / (1024 * 1024)

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    # Initialize DataLoader with MediaInterface
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    loader = DataLoader(
        paths=[str(input_mp3)], output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Collect all chunks
    chunks = []
    for chunk in loader:
        chunks.append(chunk)

    # Verify we got the expected number of chunks
    assert len(chunks) == 3, f"Expected 3 chunks, but got {len(chunks)}"

    # Verify each chunk is approximately the right size (200MB ± 10%)
    expected_chunk_size = split_size_mb * 1024 * 1024
    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        assert (
            0.9 * expected_chunk_size <= chunk_size <= 1.1 * expected_chunk_size
        ), f"Chunk {i} size ({chunk_size} bytes) is not within 10% of expected size ({expected_chunk_size} bytes)"


def test_dataloader_mp3_content(temp_dir):
    """Test that the content of chunks from DataLoader is valid MP3 data."""
    # Create an MP3 file
    input_mp3 = temp_dir / "large_input.mp3"
    create_test_mp3(input_mp3, duration_seconds=600, target_size_mb=test_file_size_mb)

    # Get original audio information
    original_info = get_audio_info(input_mp3)
    original_sample_rate = int(original_info["streams"][0]["sample_rate"])

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    actual_size_mb = input_mp3.stat().st_size * 1e-6

    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    loader = DataLoader(
        paths=[str(input_mp3)], output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Test that each chunk can be read as valid MP3 data
    for i, chunk_data in enumerate(loader):
        # Write chunk to temporary file
        chunk_path = chunks_dir / f"test_chunk_{i}.mp3"
        with open(chunk_path, "wb") as f:
            f.write(chunk_data)

        # Verify the chunk is valid MP3 data using ffprobe
        chunk_info = get_audio_info(chunk_path)

        # Check audio parameters
        assert chunk_info["streams"][0]["codec_name"] == "mp3", f"Chunk {i} should be MP3"
        assert (
            int(chunk_info["streams"][0]["sample_rate"]) == original_sample_rate
        ), f"Chunk {i} should have the same sample rate as original"
        assert float(chunk_info["format"]["duration"]) > 0, f"Chunk {i} should have valid duration"
