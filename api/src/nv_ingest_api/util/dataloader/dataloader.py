# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, NVIDIA CORPORATION.
from pathlib import Path
from abc import ABC, abstractmethod
import queue
import threading
import subprocess
import json
import logging
import math
import importlib.util
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import glob

from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)

try:
    importlib.util.find_spec("ffmpeg")
    subprocess.run(["ffmpeg", "-version"], capture_output=True)
except Exception:
    logger.error(
        "Unable to load the Dataloader, ffmpeg was not installed, "
        "please install it using `pip install ffmpeg-python` and `apt-get install ffmpeg`"
    )
    ffmpeg = None
else:
    import ffmpeg

if not ffmpeg:
    DataLoader = None
    MediaInterface = None
else:

    class SplitType(Enum):
        FRAME = "frame"
        TIME = "time"
        SIZE = "size"

    class LoaderInterface(ABC):

        @abstractmethod
        def split(self, input_path: str, output_dir: str, split_interval: int = 0):
            pass

        @abstractmethod
        def _get_path_metadata(self, path: str = None):
            pass

    def _probe(filename, format=None, file_handle=None, timeout=None, **kwargs):
        args = ["ffprobe", "-show_format", "-show_streams", "-of", "json"]
        args += ffmpeg._utils.convert_kwargs_to_cmd_line_args(kwargs)
        if file_handle:
            args += ["pipe:"]
        else:
            args += [filename]
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        communicate_kwargs = {}
        if timeout is not None:
            communicate_kwargs["timeout"] = timeout
        if file_handle:
            communicate_kwargs["input"] = file_handle if file_handle else filename
        out, err = p.communicate(**communicate_kwargs)
        if p.returncode != 0:
            raise ffmpeg._run.Error("ffprobe", out, err)
        return json.loads(out.decode("utf-8"))

    def _get_audio_from_video(input_path: str, output_file: str, cache_path: str = None):
        """
        Get the audio from a video file. if audio extraction fails, return None.
        input_path: str, path to the video file
        output_dir: str, path to the output directory
        cache_path: str, path to the cache directory
        """
        output_path = Path(output_file)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            capture_output, capture_error = (
                ffmpeg.input(str(input_path))
                .output(str(output_path), acodec="libmp3lame", map="0:a")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return output_path
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error for file {input_path}: {e.stderr.decode()}")
            return None

    def strip_audio_from_video_files(input_path: str, output_dir: str, cache_path: str = None):
        """
        Strip the audio from a series of video files and return the paths to the new files.
        input_path: str, path to the video file
        output_dir: str, path to the output directory
        cache_path: str, path to the cache directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        futures = []
        results = None
        path = Path(input_path)
        files = [path] if path.is_file() else glob.glob(os.path.join(path, "*.mp4"))
        files = [Path(file) for file in files]
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = [executor.submit(_get_audio_from_video, file, output_path / f"{file.stem}.mp3") for file in files]
            results = [str(future.result()) for future in tqdm(futures)]
        return results

    class MediaInterface(LoaderInterface):

        def __init__(self):
            self.path_metadata = {}

        def probe_media(self, path_file: Path, split_interval: int, split_type: SplitType, file_handle=None):
            num_splits = None
            duration = None
            probe = None
            sample_rate = None
            try:
                file_size = path_file.stat().st_size  # in bytes
                if file_handle:
                    probe = _probe("pipe:", format=path_file.suffix, file_handle=file_handle)
                else:
                    probe = _probe(str(path_file), format=path_file.suffix)
                if probe["streams"][0]["codec_type"] == "video":
                    sample_rate = float(probe["streams"][0]["avg_frame_rate"].split("/")[0])
                    duration = float(probe["format"]["duration"])
                elif probe["streams"][0]["codec_type"] == "audio":
                    sample_rate = float(probe["streams"][0]["sample_rate"])
                    bitrate = probe["format"]["bit_rate"]
                    duration = (file_size * 8) / float(bitrate)
                num_splits = self.find_num_splits(file_size, sample_rate, duration, split_interval, split_type)
            except ffmpeg.Error as e:
                logging.error(f"FFmpeg error for file {path_file}: {e.stderr.decode()}")
            except ValueError as e:
                logging.error(f"Error finding number of splits for file {path_file}: {e}")
            return probe, num_splits, duration

        def get_audio_from_video(self, input_path: str, output_file: str, cache_path: str = None):
            return _get_audio_from_video(input_path, output_file, cache_path)

        def split(
            self,
            input_path: str,
            output_dir: str,
            split_interval: int = 0,
            split_type: SplitType = SplitType.SIZE,
            cache_path: str = None,
            video_audio_separate: bool = False,
            audio_only: bool = False,
        ):
            """
            Split a media file into smaller chunks of `split_interval` size. if
            video_audio_separate is True and the file is a video, the audio will be
            extracted from the video and saved to a separate files. Data can be returned
            as a tuple of (video_files, audio_files) or just files (i.e. audio files).
            input_path: str, path to the media file
            output_dir: str, path to the output directory
            split_interval: the size of the chunk to split the media file into depending on the split type
            split_type: SplitType, type of split to perform, either size, time, or frame
            video_audio_separate: bool, whether to separate the video and audio files
            audio_only: bool, whether to only return the audio files
            """
            import ffmpeg

            files_to_remove = []
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            original_input_path = input_path
            if audio_only and Path(input_path).suffix in [".mp4", ".mov", ".avi", ".mkv"]:
                input_path = self.get_audio_from_video(input_path, output_dir / f"{input_path.stem}.mp3")
                files_to_remove.append(input_path)
            path_file = Path(input_path)
            file_name = path_file.stem
            suffix = path_file.suffix
            output_pattern = output_dir / f"{file_name}_chunk_%04d{suffix}"

            num_splits = 0
            cache_path = cache_path if cache_path else output_dir
            try:
                probe = None
                probe, num_splits, duration = self.probe_media(path_file, split_interval, split_type)
                segment_time = math.ceil(duration / num_splits)
                output_kwargs = {
                    "f": "segment",
                    "segment_time": segment_time,
                    "c": "copy",
                    "map": "0",
                    # use 10% of the available cores, but at least 4 threads
                    # each core has 2 threads
                    "threads": int(max(SystemResourceProbe().get_effective_cores() * 0.2, 4)),
                }
                if suffix == ".mp4":
                    output_kwargs.update(
                        {
                            "force_key_frames": f"expr:gte(t,n_forced*{segment_time})",
                            "crf": 22,
                            "g": 50,
                            "sc_threshold": 0,
                        }
                    )
                capture_output, capture_error = (
                    ffmpeg.input(str(input_path))
                    .output(str(output_pattern), **output_kwargs)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                logging.debug(f"Split {input_path} into {num_splits} chunks")
                self.path_metadata[input_path] = probe
                logging.debug(capture_output)
                logging.debug(f"{original_input_path} -  {capture_error}")
            except ffmpeg.Error as e:
                logging.error(
                    f"FFmpeg error for file {original_input_path}: {e.stderr.decode()} {capture_output} {capture_error}"
                )
                return []
            files = [str(output_dir / f"{file_name}_chunk_{i:04d}{suffix}") for i in range(int(num_splits))]
            if video_audio_separate and suffix in [".mp4", ".mov", ".avi", ".mkv"]:
                video_audio_files = []
                for file in files:
                    file = Path(file)
                    audio_path = self.get_audio_from_video(file, file.with_suffix(".mp3"), cache_path)
                    if audio_path is not None:
                        video_audio_files.append(audio_path)
                    else:
                        logging.error(f"Failed to extract audio from {file}")
                return files + video_audio_files
            for to_remove in files_to_remove:
                to_remove = Path(to_remove)
                if to_remove.is_file():
                    logger.error(f"Removing file {to_remove}")
                    to_remove.unlink()
            return files

        def find_num_splits(
            self,
            file_size: int,
            sample_rate: float,
            duration: float,
            split_interval: int,
            split_type: SplitType,
        ):
            """
            Find the number of splits for a media file based on the split type and interval.
            file_size: int, size of the media file in bytes
            sample_rate: float, sample rate of the media file in samples per second
            duration: float, duration of the media file in seconds
            split_interval: int, size of the chunk to split the media file into depending on the split type
            split_type: SplitType, type of split to perform, either size, time, or frame
            """
            if split_type == SplitType.SIZE:
                return math.ceil(file_size / split_interval)
            elif split_type == SplitType.TIME:
                return math.ceil(duration / split_interval)
            elif split_type == SplitType.FRAME:
                seconds_cap = split_interval / sample_rate
                return math.ceil(duration / seconds_cap)
            else:
                raise ValueError(f"Invalid split type: {split_type}")

        def _get_path_metadata(self):
            """
            Get the metadata for a path.
            path: str, path to get the metadata for if None, get the metadata for all paths
            """
            return self.path_metadata

    def load_data(queue: queue.Queue, paths: list[str], thread_stop: threading.Event):
        file = None
        logger.info(f"Loading data for {len(paths)} files")
        try:
            for file in paths:
                if thread_stop.is_set():
                    return
                with open(file, "rb") as f:
                    queue.put(f.read())
        except Exception as e:
            logging.error(f"Error processing file {file} type: {type(file)} {e}")
            queue.put(RuntimeError(f"Error processing file {file}: {e}"))
        finally:
            queue.put(StopIteration)

    class DataLoader:
        """
        DataLoader is a class that is used to load data from a list of paths and push it to a queue.
        paths: list[str], list of paths to process
        size: int, size of the queue
        """

        def __init__(
            self,
            path: str,
            output_dir: str,
            split_type: SplitType = SplitType.SIZE,
            split_interval: int = 450,
            interface: LoaderInterface = None,
            size: int = 2,
            video_audio_separate: bool = False,
            audio_only: bool = False,
        ):
            interface = interface if interface else MediaInterface()
            self.thread = None
            self.thread_stop = threading.Event()
            self.queue = queue.Queue(size)
            self.path = Path(path)
            self.output_dir = output_dir
            self.split_interval = split_interval
            self.interface = interface
            self.files_completed = []
            self.split_type = split_type
            self.video_audio_separate = video_audio_separate
            self.audio_only = audio_only
            # process the file immediately on instantiation
            self._process()

        def _process(self):
            files_completed = self.interface.split(
                self.path,
                self.output_dir,
                split_interval=self.split_interval,
                split_type=self.split_type,
                video_audio_separate=self.video_audio_separate,
                audio_only=self.audio_only,
            )
            # get durations for files in self.files_completed
            durations = []
            for file in files_completed:
                _, _, duration = self.interface.probe_media(
                    Path(file), split_interval=self.split_interval, split_type=self.split_type
                )
                durations.append(duration)

            self.files_completed = list(zip(files_completed, durations))

        def __next__(self):
            payload = self.queue.get()
            if payload == StopIteration:
                raise payload
            else:
                return payload

        def stop(self):
            """
            Reset itertor by stopping the thread and clearing the queue.
            """
            if self.thread:
                self.thread_stop.set()
                self.thread.join()
                self.thread = None
            try:
                while True:
                    self.queue.get_nowait()
            except Exception:
                pass
            finally:
                self.thread_stop.clear()

        def __iter__(self):
            self.stop()
            self.thread_stop.clear()
            self.thread = threading.Thread(
                target=load_data,
                args=(
                    self.queue,
                    [file for file, _ in self.files_completed],
                    self.thread_stop,
                ),
                daemon=True,
            )
            self.thread.start()
            return self

        def __len__(self):
            return len(self.files_completed)

        def __getitem__(self, index):
            file_path = self.files_completed[index]
            if isinstance(file_path, tuple):
                file_path = file_path[0]
            results = None
            try:
                if isinstance(file_path, tuple):
                    file_path = file_path[0]
                with open(file_path, "rb") as f:
                    results = f.read()
                return results
            except Exception as e:
                logging.error(f"Error getting item {index}: {e}")
                raise e

        def __del__(self):
            self.stop()

        def __exit__(self, exc_type, exc_value, traceback):
            self.stop()

        def get_metadata(self):
            """
            Get the metadata for a path.
            path: str, path to get the metadata for if None, get the metadata for all paths
            """

            return self.interface._get_path_metadata()
