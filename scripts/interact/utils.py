import docker

import glob
import inspect
import json
import os
import shutil
import time
import zipfile

from pathlib import Path

from pymilvus import MilvusClient

import pypdfium2 as pdfium
import subprocess
import datetime
import socket


def check_container_running(container_name):
    """Check if a container with the specified name is currently running.

    Args:
        container_name (str): The name or partial name to search for in container image tags.

    Returns:
        bool: True if a container with the specified name is running, False otherwise.
    """
    client = docker.from_env()
    containers = client.containers.list()

    for container in containers:
        if container.image.tags and container_name in container.image.tags[0]:
            return True

    return False


def embed_info():
    """Get embedding model information based on currently running containers.

    This function checks for specific embedding model containers and returns
    the appropriate model name and embedding dimension based on which container
    is currently running.

    Returns:
        tuple: A tuple containing (model_name: str, embedding_dimension: int).
               Returns a default model if no specific containers are found.
    """
    if check_container_running("llama-3.2-nemoretriever-1b-vlm-embed-v1"):
        return "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1", 2048
    elif check_container_running("llama-3.2-nv-embedqa-1b-v2"):
        return "nvidia/llama-3.2-nv-embedqa-1b-v2", 2048
    elif check_container_running("llama-3.2-nemoretriever-300m-embed-v1"):
        return "nvidia/llama-3.2-nemoretriever-300m-embed-v1", 2048
    else:
        return "nvidia/nv-embedqa-e5-v5", 1024


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
            with open(log_file, "r") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}

    # Update with new key-value pair
    data[key] = val

    # Write back as simple JSON object
    with open(log_file, "w") as fp:
        json.dump(data, fp, indent=2)


def unload_collection(milvus_uri: str, collection_name: str):
    """Unload/release a Milvus collection from memory.

    This function connects to a Milvus instance and releases the specified
    collection from memory, freeing up resources.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to unload.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        client.release_collection(collection_name=collection_name)
    except Exception as e:
        kv_event_log(f"{collection_name}_unload_error", str(e))
        print(f"Error unloading collection {collection_name}: {e}")


def load_collection(milvus_uri: str, collection_name: str):
    """Load a Milvus collection into memory.

    This function connects to a Milvus instance and loads the specified
    collection into memory for querying and operations.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to load.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        client.load_collection(collection_name)
    except Exception as e:
        kv_event_log(f"{collection_name}_load_error", str(e))
        print(f"Error loading collection {collection_name}: {e}")


def milvus_chunks(milvus_uri: str, collection_name: str):
    """Get and log statistics about chunks in a Milvus collection.

    This function connects to a Milvus instance, retrieves collection statistics,
    logs the chunk information using the event logging system, and closes the connection.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to get statistics for.

    Returns:
      dict: A dictionary containing the collection statistics.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        stats = client.get_collection_stats(collection_name)
        kv_event_log(f"{collection_name}_chunks", f"{stats}")
        client.close()
    except Exception as e:
        kv_event_log(f"{collection_name}_chunks_error", str(e))
        print(f"Error getting collection stats for {collection_name}: {e}")
        stats = {}
    return stats


def pdf_page_count_glob(pattern: str) -> int:
    """Count the total number of pages in all PDF files matching a glob pattern.

    This function searches for PDF files using a glob pattern and counts the total
    number of pages across all matching files.

    Args:
        pattern (str): A glob pattern to match PDF files (supports recursive search).

    Returns:
        int: The total number of pages across all matching PDF files.
    """
    total_pages = 0
    for filepath in glob.glob(pattern, recursive=True):
        if filepath.lower().endswith(".pdf"):
            pdf = pdfium.PdfDocument(filepath)
            total_pages += len(pdf)
    return total_pages


def pdf_page_count(directory: str) -> int:
    """Count the total number of pages in all PDF files within a directory.

    This function scans a directory for PDF files and counts the total number
    of pages across all PDF files found. If a PDF file cannot be processed,
    an error message is printed and the file is skipped.

    Args:
        directory (str): The directory path to search for PDF files.

    Returns:
        int: The total number of pages across all PDF files in the directory.
    """
    total_pages = 0
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            try:
                pdf = pdfium.PdfDocument(filepath)
                total_pages += len(pdf)
            except Exception as e:
                print(f"{filepath} failed: {e}")
                continue
    return total_pages


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
