import ffmpeg
import fsspec
from upath import UPath as Path
from abc import ABC, abstractmethod
import queue
import threading
import subprocess
import json
import logging
import math

logger = logging.getLogger(__name__)


class LoaderInterface(ABC):

    @abstractmethod
    def split(self, input_path: str, output_dir: str, split_interval: int = 0):
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

    def split(self, input_path: str, output_dir: str, split_interval: int = 0):
        """
        Split a video into smaller chunks of `split_interval` size.
        input_path: str, path to the video file
        output_dir: str, path to the output directory
        split_interval: int, size of the chunk in MB
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
            with fsspec.open(input_path, "rb", encoding="utf-8") as f_in:
                probe = _probe("pipe:", format=suffix, file_handle=f_in.read())
                bitrate = float(probe["format"]["bit_rate"])
                file_size = path_file.stat().st_size
                duration = (file_size * 8) / bitrate
                num_splits = math.ceil(file_size // (split_interval * 1e6))
                segment_time = (duration // num_splits) + 1
                f_in.seek(0)
                (
                    ffmpeg.input(
                        "pipe:",
                    )
                    .output(str(output_pattern), f="segment", segment_time=segment_time, c="copy", map="0")
                    .run(capture_stdout=True, capture_stderr=True, input=f_in.read())
                )
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error for file {input_path}: {e.stderr.decode()}")
        files = [str(output_dir / f"{file_name}_chunk_{i:04d}{suffix}") for i in range(int(num_splits))]
        return files


def process_data(
    queue: queue.Queue,
    interface: LoaderInterface,
    paths: list[str],
    output_dir: str,
    thread_stop: threading.Event,
    split_interval: int = 450,
    files_completed: list[str] = None,
):
    for path in paths:
        if thread_stop:
            return
        # check for file size greater than 450MB
        files = []
        if path.stat().st_size > (split_interval * 1e6):
            files = interface.split(path, output_dir, split_interval)
        # either path or files need to be read and pushed to queue
        else:
            files = [path]
        for file in files:
            if thread_stop:
                return
            with open(file, "rb") as f:
                queue.put(f.read())
        files_completed.extend(files)
    queue.put(StopIteration)
    return files_completed


class DataLoader:
    def __init__(
        self,
        paths: list[str],
        output_dir: str,
        split_interval: int = 450,
        interface: LoaderInterface = None,
        size: int = 2,
    ):
        self.thread = None
        self.thread_stop = False
        self.queue = queue.Queue(size)
        self.paths = [Path(path) for path in paths]
        self.output_dir = output_dir
        self.split_interval = split_interval
        self.chunk_count = 0
        self.interface = interface
        self.files_completed = []

    def __getitem__(self, index: int):
        return self.__next__()

    def __next__(self):
        payload = self.queue.get()
        if payload == StopIteration:
            raise payload
        else:
            return payload

    def stop(self):
        if self.thread:
            self.thread_stop = True
            self.thread.join()
        self.thread_stop = False
        while self.queue.qsize() != 0:
            with self.queue.mutex:
                self.queue.queue.clear()

    def __iter__(self):
        self.stop()
        self.files_completed = []
        self.thread = threading.Thread(
            target=process_data,
            args=(
                self.queue,
                self.interface,
                self.paths,
                self.output_dir,
                self.thread_stop,
                self.split_interval,
                self.files_completed,
            ),
            daemon=True,
        )
        self.thread.start()
        return self
