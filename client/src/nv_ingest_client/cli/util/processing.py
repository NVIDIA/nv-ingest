# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import as_completed
from statistics import mean
from statistics import median
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from nv_ingest_client.util.processing import handle_future_result
from nv_ingest_client.util.util import estimate_page_count
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def report_stage_statistics(stage_elapsed_times: defaultdict, total_trace_elapsed: float, abs_elapsed: float) -> None:
    """
    Reports the statistics for each processing stage, including average, median, total time spent,
    and their respective percentages of the total processing time.

    Parameters
    ----------
    stage_elapsed_times : defaultdict(list)
        A defaultdict containing lists of elapsed times for each processing stage, in nanoseconds.
    total_trace_elapsed : float
        The total elapsed time across all processing stages, in nanoseconds.
    abs_elapsed : float
        The absolute elapsed time from the start to the end of processing, in nanoseconds.

    Notes
    -----
    This function logs the average, median, and total time for each stage, along with the percentage of total
    computation.
    It also calculates and logs the unresolved time, if any, that is not accounted for by the recorded stages.
    """

    for stage, times in stage_elapsed_times.items():
        if times:
            avg_time = mean(times)
            med_time = median(times)
            total_stage_time = sum(times)
            percent_of_total = (total_stage_time / total_trace_elapsed * 100) if total_trace_elapsed > 0 else 0
            logger.info(
                f"{stage}: Avg: {avg_time / 1e6:.2f} ms, Median: {med_time / 1e6:.2f} ms, "
                f"Total Time: {total_stage_time / 1e6:.2f} ms, Total % of Trace Computation: {percent_of_total:.2f}%"
            )

    unresolved_time = abs_elapsed - total_trace_elapsed
    if unresolved_time > 0:
        percent_unresolved = unresolved_time / abs_elapsed * 100
        logger.info(
            f"Unresolved time: {unresolved_time / 1e6:.2f} ms, Percent of Total Elapsed: {percent_unresolved:.2f}%"
        )
    else:
        logger.info("No unresolved time detected. Trace times account for the entire elapsed duration.")


def report_overall_speed(total_pages_processed: int, start_time_ns: int, total_files: int) -> None:
    """
    Report the overall processing speed based on the number of pages and files processed.

    This function calculates the total elapsed time from the start of processing and reports the throughput
    in terms of pages and files processed per second.

    Parameters
    ----------
    total_pages_processed : int
        The total number of pages processed.
    start_time_ns : int
        The nanosecond timestamp marking the start of processing.
    total_files : int
        The total number of files processed.

    Notes
    -----
    The function converts the elapsed time from nanoseconds to seconds and logs the overall throughput.
    """
    total_elapsed_time_ns: int = time.time_ns() - start_time_ns
    total_elapsed_time_s: float = total_elapsed_time_ns / 1_000_000_000  # Convert nanoseconds to seconds

    throughput_pages: float = total_pages_processed / total_elapsed_time_s  # pages/sec
    throughput_files: float = total_files / total_elapsed_time_s  # files/sec

    logger.info(f"Processed {total_files} files in {total_elapsed_time_s:.2f} seconds.")
    logger.info(f"Total pages processed: {total_pages_processed}")
    logger.info(f"Throughput (Pages/sec): {throughput_pages:.2f}")
    logger.info(f"Throughput (Files/sec): {throughput_files:.2f}")


def report_statistics(
    start_time_ns: int,
    stage_elapsed_times: defaultdict,
    total_pages_processed: int,
    total_files: int,
) -> None:
    """
    Aggregate and report statistics for the entire processing session.

    This function calculates the absolute elapsed time from the start of processing to the current time and
    the total time taken by all stages. It then reports detailed stage statistics along with overall
    processing throughput.

    Parameters
    ----------
    start_time_ns : int
        The nanosecond timestamp marking the start of the processing.
    stage_elapsed_times : defaultdict
        A defaultdict where each key is a processing stage (str) and each value is a list of elapsed times
        (int, in nanoseconds) for that stage.
    total_pages_processed : int
        The total number of pages processed during the session.
    total_files : int
        The total number of files processed during the session.

    Notes
    -----
    The function calls `report_stage_statistics` to log detailed timing information per stage, then calls
    `report_overall_speed` to log the overall throughput.
    """
    abs_elapsed: int = time.time_ns() - start_time_ns
    total_trace_elapsed: int = sum(sum(times) for times in stage_elapsed_times.values())
    report_stage_statistics(stage_elapsed_times, total_trace_elapsed, abs_elapsed)  # Assumes implementation exists
    report_overall_speed(total_pages_processed, start_time_ns, total_files)


def process_response(response: Dict[str, Any], stage_elapsed_times: defaultdict) -> None:
    """
    Process the response to extract trace data and calculate elapsed time for each stage.

    This function iterates over trace data in the response, identifies entry and exit times for each stage,
    calculates the elapsed time, and appends the elapsed time to the corresponding stage in the provided
    `stage_elapsed_times` dictionary.

    Parameters
    ----------
    response : Dict[str, Any]
        The response dictionary containing trace information for processing stages.
    stage_elapsed_times : defaultdict
        A defaultdict where keys are stage names (str) and values are lists of elapsed times (int, in nanoseconds).

    Notes
    -----
    The function expects trace keys to include "entry" and "exit" substrings. For each entry key, the corresponding
    exit key is determined by replacing "entry" with "exit". The stage name is assumed to be the third element when
    splitting the key by "::".
    """
    trace_data: Dict[str, Any] = response.get("trace", {})
    for key, entry_time in trace_data.items():
        if "entry" in key:
            exit_key: str = key.replace("entry", "exit")
            exit_time: Any = trace_data.get(exit_key)
            if exit_time:
                # Assumes the stage name is in the third position when splitting the key
                stage_parts = key.split("::")
                if len(stage_parts) >= 3:
                    stage_name: str = stage_parts[2]
                    elapsed_time: int = exit_time - entry_time
                    stage_elapsed_times[stage_name].append(elapsed_time)


def organize_documents_by_type(response_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize documents by their content type.

    This function takes a list of response documents, extracts the content type from each document's metadata,
    and organizes the documents into a dictionary, where the keys are content types and the values are lists of
    documents belonging to each type.

    Parameters
    ----------
    response_data : List[Dict[str, Any]]
        A list of documents, where each document is represented as a dictionary. Each dictionary must contain
        a 'metadata' field that may be either a JSON string or a dictionary. The metadata is expected to have a
        "content_metadata" field containing the document's type.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A dictionary mapping document types (as strings) to lists of documents. Each key represents a document type,
        and the associated value is a list of documents that belong to that type.

    Notes
    -----
    - If the 'metadata' field of a document is a string, it is parsed into a dictionary using `json.loads`.
    - The function assumes that each document's metadata has a valid "content_metadata" field with a "type" key.
    - Documents are grouped by the value of the "type" key in their "content_metadata".

    Examples
    --------
    >>> response_data = [
    ...     {"metadata": {"content_metadata": {"type": "report"}}},
    ...     {"metadata": '{"content_metadata": {"type": "summary"}}'},
    ...     {"metadata": {"content_metadata": {"type": "report"}}}
    ... ]
    >>> organize_documents_by_type(response_data)
    {'report': [{'metadata': {'content_metadata': {'type': 'report'}}},
                {'metadata': {'content_metadata': {'type': 'report'}}}],
     'summary': [{'metadata': {'content_metadata': {'type': 'summary'}}}]}
    """
    doc_map: Dict[str, List[Dict[str, Any]]] = {}
    for document in response_data:
        doc_meta: Any = document["metadata"]
        if isinstance(doc_meta, str):
            doc_meta = json.loads(doc_meta)
        doc_content_metadata: Dict[str, Any] = doc_meta["content_metadata"]
        doc_type: str = doc_content_metadata["type"]
        if doc_type not in doc_map:
            doc_map[doc_type] = []
        doc_map[doc_type].append(document)
    return doc_map


def save_response_data(response: Dict[str, Any], output_directory: str, images_to_disk: bool = False) -> None:
    """
    Save the response data into categorized metadata JSON files and optionally save images to disk.

    This function processes the response data, organizes it based on document types, and saves the organized data
    into a specified output directory as JSON files. If 'images_to_disk' is True and the document type is 'image',
    it decodes and writes base64 encoded images to disk.

    Parameters
    ----------
    response : Dict[str, Any]
        A dictionary containing the API response data. It must contain a "data" field, which is expected to be a
        list of document entries. Each document entry should contain metadata, which includes information about
        the document's source.
    output_directory : str
        The path to the directory where the JSON metadata files should be saved. Subdirectories will be created based
        on the document types, and the metadata files will be stored within these subdirectories.
    images_to_disk : bool, optional
        If True, base64 encoded images in the 'metadata.content' field will be decoded and saved to disk.
        Default is False.

    Returns
    -------
    None
        This function does not return any values. It writes output to the filesystem.

    Notes
    -----
    - If 'images_to_disk' is True and 'doc_type' is 'image', images will be decoded and saved to disk with appropriate
      file types based on 'metadata.image_metadata.image_type'.
    """
    if ("data" not in response) or (not response["data"]):
        logger.debug("Data is not in the response or response.data is empty")
        return

    response_data = response["data"]

    if not isinstance(response_data, list) or len(response_data) == 0:
        logger.debug("Response data is not a list or the list is empty.")
        return

    doc_meta_base = response_data[0]["metadata"]
    source_meta = doc_meta_base["source_metadata"]
    doc_name = source_meta["source_id"]
    clean_doc_name = get_valid_filename(os.path.basename(doc_name))
    output_name = f"{clean_doc_name}.metadata.json"

    doc_map = organize_documents_by_type(response_data)
    for doc_type, documents in doc_map.items():
        doc_type_path = os.path.join(output_directory, doc_type)
        if not os.path.exists(doc_type_path):
            os.makedirs(doc_type_path)

        if doc_type in ("image", "structured") and images_to_disk:
            for i, doc in enumerate(documents):
                meta: Dict[str, Any] = doc.get("metadata", {})
                image_content = meta.get("content")
                if doc_type == "image":
                    image_type = meta.get("image_metadata", {}).get("image_type", "png").lower()
                else:
                    image_type = "png"

                if image_content and image_type in {"png", "bmp", "jpeg", "jpg", "tiff"}:
                    try:
                        # Decode the base64 content
                        image_data = base64.b64decode(image_content)
                        image = Image.open(io.BytesIO(image_data))

                        # Define the output file path
                        image_ext = "jpg" if image_type == "jpeg" else image_type
                        image_filename = f"{clean_doc_name}_{i}.{image_ext}"
                        image_output_path = os.path.join(doc_type_path, "media", image_filename)

                        # Ensure the media directory exists
                        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)

                        # Save the image to disk
                        image.save(image_output_path, format=image_ext.upper())

                        # Update the metadata content with the image path
                        meta["content"] = ""
                        meta["content_url"] = os.path.realpath(image_output_path)
                        logger.debug(f"Saved image to {image_output_path}")

                    except Exception as e:
                        logger.error(f"Failed to save image {i} for {clean_doc_name}: {e}")

        # Write the metadata JSON file
        with open(os.path.join(doc_type_path, output_name), "w") as f:
            f.write(json.dumps(documents, indent=2))


def generate_job_batch_for_iteration(
    client: Any,
    pbar: Any,
    files: List[str],
    tasks: Dict[str, Any],
    processed: int,
    batch_size: int,
    retry_job_ids: List[str],
    fail_on_error: bool = False,
) -> Tuple[List[str], Dict[str, str], int]:
    """
    Generates a batch of job specifications for the current iteration of file processing.
    This function handles retrying failed jobs and creating new jobs for unprocessed files.
    The job specifications are then submitted for processing.

    Parameters
    ----------
    client : Any
        The client object used to submit jobs asynchronously.
    pbar : Any
        The progress bar object used to update the progress as jobs are processed.
    files : List[str]
        The list of file paths to be processed.
    tasks : Dict[str, Any]
        A dictionary of tasks to be executed as part of the job specifications.
    processed : int
        The number of files that have been processed so far.
    batch_size : int
        The maximum number of jobs to process in one batch.
    retry_job_ids : List[str]
        A list of job IDs that need to be retried due to previous failures.
    fail_on_error : bool, optional
        Whether to raise an error and stop processing if job specifications are missing. Default is False.

    Returns
    -------
    Tuple[List[str], Dict[str, str], int]
        A tuple containing:
        - job_ids (List[str]): The list of job IDs created or retried in this iteration.
        - job_id_map_updates (Dict[str, str]): A dictionary mapping job IDs to their corresponding file names.
        - processed (int): The updated number of files processed.

    Raises
    ------
    RuntimeError
        If `fail_on_error` is True and there are missing job specifications, a RuntimeError is raised.
    """
    job_indices: List[str] = []
    job_index_map_updates: Dict[str, str] = {}
    cur_job_count: int = 0

    if retry_job_ids:
        job_indices.extend(retry_job_ids)
        cur_job_count = len(job_indices)

    if (cur_job_count < batch_size) and (processed < len(files)):
        new_job_count: int = min(batch_size - cur_job_count, len(files) - processed)
        batch_files: List[str] = files[processed : processed + new_job_count]

        new_job_indices: List[str] = client.create_jobs_for_batch(batch_files, tasks)
        if len(new_job_indices) != new_job_count:
            missing_jobs: int = new_job_count - len(new_job_indices)
            error_msg: str = f"Missing {missing_jobs} job specs -- this is likely due to bad reads or file corruption"
            logger.warning(error_msg)

            if fail_on_error:
                raise RuntimeError(error_msg)

            pbar.update(missing_jobs)

        job_index_map_updates = {job_index: file for job_index, file in zip(new_job_indices, batch_files)}
        processed += new_job_count
        _ = client.submit_job_async(new_job_indices, "ingest_task_queue")
        job_indices.extend(new_job_indices)

    return job_indices, job_index_map_updates, processed


def create_and_process_jobs(
    files: List[str],
    client: Any,
    tasks: Dict[str, Any],
    output_directory: str,
    batch_size: int,
    fail_on_error: bool = False,
    save_images_separately: bool = False,
) -> Tuple[int, Dict[str, List[float]], int, Dict[str, str]]:
    """
    Process a list of files by creating and submitting jobs for each file, then fetching
    and handling the results asynchronously.

    This function creates job specifications (JobSpecs) for the provided list of files,
    submits the jobs to the client, and processes the results asynchronously. It handles
    job retries for timeouts, logs failures, and limits the number of JobSpecs in memory to
    `batch_size * 2`. Progress is reported on a per-file basis, including the pages processed
    per second.

    Parameters
    ----------
    files : List[str]
        A list of file paths to be processed. Each file is used to create a job which is then
        submitted to the client.
    client : Any
        An instance of NvIngestClient used to submit jobs and fetch results asynchronously.
    tasks : Dict[str, Any]
        A dictionary of tasks to be added to each job. The keys represent task names (e.g., "split",
        "extract", "store", "caption", etc.) and the values represent task configurations.
    output_directory : str
        The directory path where the processed job results will be saved. If an empty string or None
        is provided, results will not be saved.
    batch_size : int
        The number of jobs to process in each batch. Memory is limited to `batch_size * 2` jobs at
        any time.
    fail_on_error : bool, optional
        If True, the function will raise an error and stop processing when encountering an unrecoverable
        error. If False, the function logs the error and continues processing other jobs. Default is False.
    save_images_separately : bool, optional
        If True, images will be saved separately to disk. Default is False.

    Returns
    -------
    Tuple[int, Dict[str, List[float]], int, Dict[str, str]]
        A tuple containing:
        - total_files (int): The total number of files processed.
        - trace_times (Dict[str, List[float]]): A dictionary mapping job IDs to a list of trace times
          for diagnostic purposes.
        - total_pages_processed (int): The total number of pages processed from the files.
        - trace_ids (Dict[str, str]): A dictionary mapping a source file to its correlating trace_id.

    Raises
    ------
    RuntimeError
        If `fail_on_error` is True and an error occurs during job submission or processing.
    """
    total_files: int = len(files)
    total_pages_processed: int = 0
    trace_times: Dict[str, List[float]] = defaultdict(list)
    trace_ids: Dict[str, str] = defaultdict(list)  # type: ignore
    failed_jobs: List[str] = []
    retry_job_ids: List[str] = []
    job_id_map: Dict[str, str] = {}
    retry_counts: Dict[str, int] = defaultdict(int)
    file_page_counts: Dict[str, int] = {file: estimate_page_count(file) for file in files}

    start_time_ns: int = time.time_ns()
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        processed: int = 0
        while (processed < len(files)) or retry_job_ids:
            # Process new batch of files or retry failed job IDs
            job_ids, job_id_map_updates, processed = generate_job_batch_for_iteration(
                client, pbar, files, tasks, processed, batch_size, retry_job_ids, fail_on_error
            )
            job_id_map.update(job_id_map_updates)
            retry_job_ids = []

            futures_dict: Dict[Any, str] = client.fetch_job_result_async(job_ids, data_only=False)
            for future in as_completed(futures_dict.keys()):
                retry: bool = False
                job_id: str = futures_dict[future]
                source_name: str = job_id_map[job_id]
                try:
                    future_response, trace_id = handle_future_result(future)
                    trace_ids[source_name] = trace_id

                    if output_directory:
                        save_response_data(future_response, output_directory, images_to_disk=save_images_separately)

                    total_pages_processed += file_page_counts[source_name]
                    elapsed_time: float = (time.time_ns() - start_time_ns) / 1e9
                    pages_per_sec: float = total_pages_processed / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}")

                    process_response(future_response, trace_times)

                except TimeoutError:
                    retry_counts[source_name] += 1
                    retry_job_ids.append(job_id)  # Add job_id back to retry list
                    retry = True
                except json.JSONDecodeError as e:
                    logger.error(f"Decoding error while processing {job_id}({source_name}): {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except RuntimeError as e:
                    logger.error(f"Error while processing '{job_id}' - ({source_name}):\n{e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except Exception as e:
                    logger.exception(f"Unhandled error while processing {job_id}({source_name}): {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                finally:
                    # Do not update progress bar if we're going to retry the job.
                    if not retry:
                        pbar.update(1)

    return total_files, trace_times, total_pages_processed, trace_ids


def get_valid_filename(name: Any) -> str:
    """
    Return a sanitized version of the given filename.

    This function, adapted from Django (https://github.com/django/django/blob/main/django/utils/text.py),
    converts the input string to a form that is safe to use as a filename. It trims leading and trailing spaces,
    replaces remaining spaces with underscores, and removes any characters that are not alphanumeric, dashes,
    underscores, or dots.

    Parameters
    ----------
    name : Any
        The input value to be converted into a valid filename. It will be converted to a string.

    Returns
    -------
    str
        A sanitized string that can be used as a filename.

    Raises
    ------
    ValueError
        If a valid filename cannot be derived from the input.

    Examples
    --------
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s: str = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise ValueError("Could not derive file name from '%s'" % name)
    return s
