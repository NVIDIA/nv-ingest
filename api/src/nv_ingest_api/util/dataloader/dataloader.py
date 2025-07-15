# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025, NVIDIA CORPORATION.
import fsspec
from upath import UPath as Path
from abc import ABC, abstractmethod
import queue
import threading
import subprocess
import json
import logging
import math
import importlib
from enum import Enum

logger = logging.getLogger(__name__)

if (
    importlib.util.find_spec("ffmpeg") is None
    or subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0
):
    logger.error(
        "Unable to load the Dataloader,ffmpeg was not installed, "
        "please install it using `pip install ffmpeg-python` and `apt-get install ffmpeg`"
    )
else:
    import ffmpeg


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


def _probe(filename="pipe:", cmd="ffprobe", format=None, file_handle=None, timeout=None, **kwargs):
    args = [cmd, "-show_format", "-show_streams", "-of", "json"]
    args += ffmpeg._utils.convert_kwargs_to_cmd_line_args(kwargs)
    args += ["pipe:"]

    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    communicate_kwargs = {}
    if timeout is not None:
        communicate_kwargs["timeout"] = timeout
    if file_handle:
        communicate_kwargs["input"] = file_handle
    out, err = p.communicate(**communicate_kwargs)
    if p.returncode != 0:
        raise ffmpeg._run.Error("ffprobe", out, err)
    return json.loads(out.decode("utf-8"))


class MediaInterface(LoaderInterface):

    def __init__(self):
        self.path_metadata = {}

    def split(self, input_path: str, output_dir: str, split_interval: int = 0, split_type: SplitType = SplitType.SIZE):
        """
        Split a media file into smaller chunks of `split_interval` size.
        input_path: str, path to the media file
        output_dir: str, path to the output directory
        split_interval: the size of the chunk to split the media file into depending on the split type
        split_type: SplitType, type of split to perform, either size, time, or frame
        """
        import ffmpeg
        from upath import UPath as Path

        path_file = Path(input_path)
        file_name = path_file.stem
        suffix = path_file.suffix
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = output_dir / f"{file_name}_chunk_%04d{suffix}"
        num_splits = 0
        try:
            with fsspec.open(input_path, "rb") as f_in:
                probe = _probe("pipe:", format=suffix, file_handle=f_in.read())
                print(f"Probe: {probe}")
                bitrate = float(probe["format"]["bit_rate"])
                sample_rate = float(probe["streams"][0]["sample_rate"])
                file_size = path_file.stat().st_size
                duration = (file_size * 8) / bitrate
                num_splits = self.find_num_splits(file_size, bitrate, sample_rate, duration, split_interval, split_type)
                self.path_metadata[input_path] = {
                    "bitrate": bitrate,
                    "sample_rate": sample_rate,
                    "file_size": file_size,
                    "duration": duration,
                    "num_channels": probe["streams"][0]["channels"],
                }
                segment_time = math.ceil(duration / num_splits)
                f_in.seek(0)
                (
                    ffmpeg.input(
                        "pipe:",
                    )
                    .output(str(output_pattern), f="segment", segment_time=segment_time, c="copy", map="0")
                    .run(capture_stdout=True, capture_stderr=True, input=f_in.read())
                )
                print(f"Split {input_path} into {num_splits} chunks")
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error for file {input_path}: {e.stderr.decode()}")
        except ValueError as e:
            logging.error(f"Error finding number of splits for file {input_path}: {e}")
        files = [str(output_dir / f"{file_name}_chunk_{i:04d}{suffix}") for i in range(int(num_splits))]
        print(f"Files: {files}")
        return files

    def find_num_splits(
        self,
        file_size: int,
        bitrate: float,
        sample_rate: float,
        duration: float,
        split_interval: int,
        split_type: SplitType,
    ):
        """
        Find the number of splits for a media file based on the split type and interval.
        file_size: int, size of the media file in bytes
        bitrate: float, bitrate of the media file in bits per second
        sample_rate: float, sample rate of the media file in samples per second
        duration: float, duration of the media file in seconds
        split_interval: int, size of the chunk to split the media file into depending on the split type
        split_type: SplitType, type of split to perform, either size, time, or frame
        """
        if split_type == SplitType.SIZE:
            return math.ceil(file_size / (split_interval * 1e6))
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
    try:
        for file in paths:
            if thread_stop:
                return
            with open(file, "rb") as f:
                queue.put(f.read())
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")
        queue.put(RuntimeError(f"Error processing file {file}: {e}"))
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
    ):
        self.thread = None
        self.thread_stop = False
        self.queue = queue.Queue(size)
        self.path = Path(path)
        self.output_dir = output_dir
        self.split_interval = split_interval
        self.interface = interface
        self.files_completed = []
        self.split_type = split_type
        # process the file immediately on instantiation
        self.files_completed = self.interface.split(self.path, self.output_dir, self.split_interval, self.split_type)

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
            self.thread_stop = True
            self.thread.join()
        self.thread_stop = False
        while self.queue.qsize() != 0:
            with self.queue.mutex:
                self.queue.queue.clear()

    def __iter__(self):
        self.stop()
        self.thread_stop = False
        self.thread = threading.Thread(
            target=load_data,
            args=(
                self.queue,
                self.files_completed,
                self.thread_stop,
            ),
            daemon=True,
        )
        self.thread.start()
        return self

    def __getitem__(self, index):
        try:
            with open(self.files_completed[index], "rb") as f:
                return f.read()
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
