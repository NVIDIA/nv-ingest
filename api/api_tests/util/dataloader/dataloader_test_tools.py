import os
import numpy as np
import subprocess
import wave
from pathlib import Path
import math
from moviepy.video.VideoClip import ColorClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips


def create_test_wav(filepath, duration_seconds=6.0, sample_rate=44100, file_size_mb=100):
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


def create_test_file(filepath, fps=24, width=100, height=100, file_size_mb=100, add_audio=False):
    # Create a ColorClip
    duration = (file_size_mb * 1024 * 1024) / (fps * width * height * 3)
    sub_duration = math.ceil(duration / 3)
    clip1 = ColorClip((width, height), color=(255, 0, 0), duration=sub_duration)
    clip2 = ColorClip((width, height), color=(0, 255, 0), duration=sub_duration)
    clip3 = ColorClip((width, height), color=(0, 0, 255), duration=duration - 2 * sub_duration)

    clip = concatenate_videoclips([clip1, clip2, clip3])
    if add_audio:
        audio_path = Path(filepath)
        audio_path = audio_path.with_suffix(".mp3")
        create_test_mp3(audio_path, duration_seconds=duration, target_size_mb=file_size_mb)
        audio_clip = AudioFileClip(audio_path)
        clip = clip.with_audio(audio_clip.with_duration(duration))
    codec_type = {
        ".mp4": "mpeg4",
        ".mkv": "libvpx",
        ".avi": "rawvideo",
        ".webm": "libvpx",
    }
    # Write the video file
    clip.write_videofile(filepath, fps=fps, codec=codec_type[filepath.suffix.lower()])
