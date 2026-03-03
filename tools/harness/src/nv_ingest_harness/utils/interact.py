import glob
import inspect
import json
import os
import shutil
import time
import zipfile
import urllib.request
import urllib.error

from pathlib import Path

import subprocess
import datetime
import socket


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, cwd=cwd)


def embed_info(
    max_retries: int = 5,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
    request_timeout: float = 2.0,
):
    """Get embedding model information from the embedding service.

    This function attempts to query the embedding service API at localhost:8012
    to get the model name with retry logic and exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        initial_backoff: Initial backoff time in seconds (default: 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        request_timeout: Timeout for each request in seconds (default: 2.0)

    Returns:
        tuple: A tuple containing (model_name: str, embedding_dimension: int).
               Returns a default model if the embedding service is not available after retries.
    """
    # Model name to embedding dimension mapping
    MODEL_DIMENSIONS = {
        "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": 2048,
        "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
        "nvidia/llama-3.2-nemoretriever-300m-embed-v1": 2048,
        "nvidia/llama-nemotron-embed-vl-1b-v2": 2048,
        "nvidia/nv-embedqa-e5-v5": 1024,
    }

    # Default model
    DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"
    DEFAULT_DIMENSION = 1024

    url = "http://localhost:8012/v1/models"

    # Try to fetch model info from embedding service API with retry/backoff
    for attempt in range(max_retries):
        should_retry = False

        try:
            with urllib.request.urlopen(url, timeout=request_timeout) as response:
                # Check if we got a successful response
                if response.status != 200:
                    # Non-200 response should trigger retry
                    should_retry = True
                else:
                    data = json.loads(response.read().decode("utf-8"))
                    # Check if we got valid data
                    if data.get("data") and len(data["data"]) > 0:
                        model_name = data["data"][0].get("id")
                        if model_name:
                            dimension = MODEL_DIMENSIONS.get(model_name, DEFAULT_DIMENSION)
                            return model_name, dimension
                    # Got 200 but incomplete/invalid data - retry
                    should_retry = True

        except Exception:
            # Any exception should trigger retry
            should_retry = True

        # If we need to retry and haven't exhausted attempts, backoff and continue
        if should_retry:
            if attempt == max_retries - 1:
                # Last attempt failed, fall back to defaults
                return DEFAULT_MODEL, DEFAULT_DIMENSION

            # Calculate backoff time with exponential increase
            backoff_time = initial_backoff * (backoff_multiplier**attempt)
            time.sleep(backoff_time)

    # Fallback if we somehow exit the loop without returning
    return DEFAULT_MODEL, DEFAULT_DIMENSION


def clean_spill(path: str):
    """Clean up temporary spill directories by removing all subdirectories.

    Args:
        path (str): The base path containing spill directories to clean up.
    """
    for path in glob.glob(f"{path}/*"):
        shutil.rmtree(path, ignore_errors=True)


def save_extracts(save_dir: str, results: list):
    """Save extraction results to a JSON file with timestamp and timing information.

    Args:
        save_dir (str): Directory where the results file will be saved.
        results (list): List of extraction results to save.
    """
    parent_fn = get_parent_script_filename()
    dt_hour = date_hour()
    t0 = time.time()
    with open(f"{save_dir}/{parent_fn}_{dt_hour}.json", "w") as fp:
        fp.write(json.dumps(results))
    t1 = time.time()
    kv_event_log("disk_write_time", t1 - t0)


def get_parent_script_filename():
    """Get the filename of the script that initiated the call stack.

    Returns:
        str: The basename of the original calling script filename.
    """
    stack = inspect.stack()
    caller_frame = stack[-1]
    caller_filename = caller_frame.filename
    return os.path.basename(caller_filename)


def get_directory_size(path):
    """
    Calculate the total size of all files in a directory (including subdirectories).

    Args:
        path (str): The path to the directory.

    Returns:
        int: Total size of all files in bytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            full_path = os.path.join(dirpath, f)
            # Skip if the file is a broken symbolic link
            if os.path.isfile(full_path):
                try:
                    total_size += os.path.getsize(full_path)
                except OSError:
                    pass  # Ignore files that can't be accessed
    return total_size


def zip(path, fn):
    """Create a ZIP archive containing all files from a directory.

    Args:
        path (str): The directory path to compress.
        fn (str): The filename for the output ZIP file.
    """
    # create a new ZipFile object in write mode
    zip_obj = zipfile.ZipFile(fn, "w")

    # loop through all the files in the directory and add them to the ZIP file
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            zip_obj.write(file_path)

    # close the ZIP file
    zip_obj.close()


def _load_env_file_from_same_directory() -> None:
    """Load environment variables from a `.env` file beside this script.

    Variables already present in the environment are not overridden. The file is
    parsed as simple KEY=VALUE pairs; lines starting with `#` are ignored.
    Surrounding quotes in values are stripped.
    """
    dotenv_path = Path(__file__).parent / ".env"
    if not dotenv_path.exists():
        return

    try:
        with dotenv_path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Do not override existing env vars
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Fail-soft: if loading the file fails for any reason, continue without env vars
        pass


def get_gpu_name():
    """Get the name of the first GPU device.

    This function uses pynvml to query the GPU device name. If pynvml is not
    available on the system, it will print warning messages and return a
    placeholder string.

    Returns:
      str: The name of the first GPU device (index 0) if available,
           or "GPU_INFO_UNAVAILABLE" if pynvml is not installed.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetName(handle)
    except ImportError:
        print("WARNING: pynvml is not available on this system. GPU information cannot be retrieved.")
        print("WARNING: Please install pynvml to enable GPU monitoring functionality.")
        return "GPU_INFO_UNAVAILABLE"


def kv_event_log(key, val, log_path: str = "test_results"):
    """Log a key-value pair event with timestamp to both console and JSON file.

    Args:
        key: The event key/name to log.
        val: The event value to log.
        log_path (str, optional): The directory path where log files will be stored.
                                 Defaults to "test_results".
    """
    print(f"{date_hour()}: {key}: {val}")
    caller_fn = get_parent_script_filename()

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Handle special cases like <stdin> or invalid filenames
    base_name = caller_fn.split(".")[0]
    if base_name.startswith("<") or not base_name.replace("_", "").replace("-", "").isalnum():
        base_name = "test_events"

    log_file = f"{log_path}/{base_name}.json"

    # Read existing data or create new dict
    data = {}
    if os.path.exists(log_file):
        try:
            with open(log_file) as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

    # Update with new key-value pair
    data[key] = val

    # Write back as simple JSON object
    with open(log_file, "w") as fp:
        json.dump(data, fp, indent=2)


def segment_results(results):
    """Segment nv-ingest results into text, table, and chart categories.

    This function takes the results from an nv-ingest invocation and separates
    them into three distinct categories based on their document type and content
    subtype: text documents, tables, and charts.

    Args:
      results: The results data structure from nv-ingest containing extracted
               content with document types and metadata.

    Returns:
      tuple: A tuple containing three lists:
             - text_results: Elements with document_type == 'text'
             - table_results: Elements with content subtype == 'table'
             - chart_results: Elements with content subtype == 'chart'
    """
    text_results = [[element for element in results if element["document_type"] == "text"] for results in results]
    table_results = [
        [element for element in results if element["metadata"]["content_metadata"]["subtype"] == "table"]
        for results in results
    ]
    chart_results = [
        [element for element in results if element["metadata"]["content_metadata"]["subtype"] == "chart"]
        for results in results
    ]
    return text_results, table_results, chart_results


def run_system_command(command: list, print_output=False):
    """Run a system command and capture its output.

    This function executes a system command using subprocess.Popen and captures
    both the return code and all output from stdout. Optionally prints the output
    in real-time as it's generated.

    Args:
        command (list): A list of command arguments to execute (e.g., ['ls', '-la']).
        print_output (bool, optional): Whether to print output to console in real-time.
                                     Defaults to False.

    Returns:
        tuple: A tuple containing:
            - return_code (int): The exit code of the executed command
            - all_output (str): All stdout output from the command concatenated together
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)

    return_code = -1
    all_output = ""
    while True:
        output = process.stdout.readline()
        if print_output:
            print(output.strip())
        all_output = all_output + output.strip()
        return_code = process.poll()
        if return_code is not None:
            for output in process.stdout.readlines():
                print(output.strip())
                all_output = all_output + output.strip()
            break

    return return_code, all_output


def date_hour():
    """Get the current date and hour formatted as a string.

    Returns:
      str: The current date and hour in MM-DD-YY_HH format (12-hour format).
    """
    now = datetime.datetime.now()
    return now.strftime("%m-%d-%y_%I")


def time_delta(fn):
    """Calculate the time difference in minutes between file creation and now.

    This function gets the creation time of a file and calculates how many
    minutes have elapsed since the file was created.

    Args:
      fn (str): The file path to check the creation time for.

    Returns:
      int: The number of minutes since the file was created.
    """
    file_ctime = os.path.getctime(fn)
    file_creation_time = datetime.datetime.fromtimestamp(file_ctime)
    current_time = datetime.datetime.now()
    time_difference = current_time - file_creation_time
    minutes = int(time_difference.total_seconds() / 60)
    return minutes


def timestr_fn(test_type):
    """Generate a timestamp string with test type and hostname.

    Creates a formatted string containing the current date/time, test type,
    and hostname for use in naming files or logging.

    Args:
      test_type (str): The type of test being performed.

    Returns:
      str: A formatted string in the format "YYYY-MM-DD_HH_testtype_hostname".
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H") + "_" + test_type + "_" + socket.gethostname()
