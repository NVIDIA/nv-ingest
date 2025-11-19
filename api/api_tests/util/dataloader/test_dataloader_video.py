# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import tempfile
from pathlib import Path
import shutil
from nv_ingest_api.util.dataloader import DataLoader, MediaInterface
import subprocess
import json
import math
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from .dataloader_test_tools import create_test_file

test_file_size_mb = 100

if not DataLoader or not MediaInterface:
    pytest.skip("DataLoader or MediaInterface is not available", allow_module_level=True)


def get_file_info(filepath):
    """Get file information using ffprobe."""
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


@pytest.mark.parametrize("file_type", [".avi", ".mp4", ".mkv", ".webm"])
def test_dataloader_chunking(temp_dir, file_type):
    """Test that DataLoader correctly splits WAV files based on size."""
    # Create a WAV file that's 600MB (larger than default 450MB split size)
    input_file = temp_dir / f"large_input{file_type}"
    # input_file = "jperez@10.20.215.157:/Users/jperez/Downloads/large_input.mp4"
    create_test_file(input_file, file_size_mb=test_file_size_mb)  # 600MB file

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    actual_size_mb = input_file.stat().st_size  # in bytes

    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb) / 3  # Should result in 3 chunks
    num_chunks = math.ceil(actual_size_mb / split_size_mb)
    loader = DataLoader(
        path=str(input_file), output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Collect all chunks
    chunks = []
    for chunk in loader:
        chunks.append(chunk)

    # Verify we got the expected number of chunks (600MB/200MB = 3 chunks)
    assert len(chunks) == num_chunks, f"Expected {num_chunks} chunks, but got {len(chunks)}"

    # Verify each chunk is approximately the right size (200MB ± 10%)
    # Convert MB to bytes
    for i, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        # Allow for some variation in chunk size (±10%)
        assert (
            0.8 * split_size_mb <= chunk_size <= 1.2 * split_size_mb
        ), f"Chunk {i} size ({chunk_size} bytes) is not within 10% of expected size ({split_size_mb} bytes)"


@pytest.mark.parametrize("file_type", [".avi", ".mp4", ".mkv", ".webm"])
def test_dataloader_getitem(temp_dir, file_type):
    """Test that DataLoader.__getitem__ correctly retrieves chunks by index."""
    # Create a WAV file that's 600MB
    input_file = temp_dir / f"large_input{file_type}"
    create_test_file(input_file, file_size_mb=test_file_size_mb)

    # Create output directory for chunks
    chunks_dir = temp_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    actual_size_mb = input_file.stat().st_size
    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    num_chunks = math.ceil(actual_size_mb / split_size_mb)
    loader = DataLoader(
        path=str(input_file), output_dir=str(chunks_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    # Test that we can access chunks by index
    assert (
        len(loader.files_completed) == num_chunks
    ), f"Expected {num_chunks} chunks, but got {len(loader.files_completed)}"

    # Test accessing each chunk by index
    for i in range(len(loader.files_completed)):
        chunk_data = loader[i]
        assert chunk_data is not None, f"Chunk {i} should not be None"
        assert len(chunk_data) > 0, f"Chunk {i} should contain data"

        # Verify the chunk data is valid WAV data
        chunk_path = chunks_dir / f"large_input_chunk_{i:04d}{file_type}"
        with open(chunk_path, "wb") as f:
            f.write(chunk_data)

        # Check that the chunk is valid WAV data
        clip = VideoFileClip(str(chunk_path))
        assert clip.fps == 24, f"Chunk {i} should have correct fps"
        assert clip.size == [100, 100], f"Chunk {i} should have correct size"
        assert clip.duration >= 0.0, f"Chunk {i} should have a duration other than 0"
        clip.close()

    # Test that accessing out-of-bounds index raises an exception
    with pytest.raises(Exception):
        loader[999]  # Should raise an exception for invalid index

    # Test that accessing negative index raises an exception
    with pytest.raises(Exception):
        loader[-4]  # Should raise an exception for negative index


def test_dataloader_audio(temp_dir):
    """Test that DataLoader correctly splits WAV files based on size."""
    # Create a WAV file that's 600MB
    input_file = temp_dir / "large_input.mp4"
    create_test_file(input_file, file_size_mb=test_file_size_mb, add_audio=True)
    actual_size_mb = input_file.stat().st_size
    # Initialize DataLoader
    split_size_mb = math.ceil(actual_size_mb / 3)  # Should result in 3 chunks
    loader = DataLoader(
        path=str(input_file), output_dir=str(temp_dir), split_interval=split_size_mb, interface=MediaInterface()
    )

    output_path = temp_dir / "large_input_audio.mp3"

    out_file = loader.interface.get_audio_from_video(input_file, output_file=str(output_path))
    video_clip = VideoFileClip(str(input_file))
    audio_clip = AudioFileClip(str(out_file))
    assert int(audio_clip.duration) == int(video_clip.duration)
    assert audio_clip.nchannels == 2
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert out_file.suffix == ".mp3"
    assert out_file.stem == "large_input_audio"
    assert out_file.name == "large_input_audio.mp3"


def test_dataloader_video_audio_separate(temp_dir):
    """Test that DataLoader correctly splits WAV files based on size."""
    # Create a WAV file that's 600MB
    input_file = temp_dir / "large_input.mp4"
    create_test_file(input_file, file_size_mb=test_file_size_mb, add_audio=True)
    actual_size_bytes = input_file.stat().st_size
    # Initialize DataLoader
    split_size_bytes = math.ceil(actual_size_bytes / 3)  # Should result in 3 chunks
    num_chunks = math.ceil(actual_size_bytes / split_size_bytes)
    loader = DataLoader(
        path=str(input_file),
        output_dir=str(temp_dir),
        split_interval=split_size_bytes,
        interface=MediaInterface(),
        video_audio_separate=True,
    )

    assert len(loader.files_completed) == num_chunks * 2

    for chunk in loader:
        if isinstance(chunk, tuple):
            assert len(chunk) == 2
            assert len(chunk[0]) > 0
            assert len(chunk[1]) > 0

    for idx, res in enumerate(loader.files_completed):
        file = res[0]
        assert Path(file).exists()

        if idx < 3:
            video_file = Path(file)
            assert video_file.stat().st_size > 0
            assert video_file.suffix == ".mp4"
            assert video_file.stem == f"large_input_chunk_{idx:04d}"
            assert video_file.name == f"large_input_chunk_{idx:04d}.mp4"

        else:
            audio_file = Path(file)
            assert audio_file.stat().st_size > 0
            assert audio_file.suffix == ".mp3"
            assert audio_file.stem == f"large_input_chunk_{idx-3:04d}"
            assert audio_file.name == f"large_input_chunk_{idx-3:04d}.mp3"
