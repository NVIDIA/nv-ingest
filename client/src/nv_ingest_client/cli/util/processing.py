# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import as_completed
from statistics import mean
from statistics import median
from typing import Dict
from typing import List
from typing import Type

from click import style
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.util.file_processing.extract import extract_file_content
from nv_ingest_client.util.util import check_ingest_result
from nv_ingest_client.util.util import estimate_page_count
from pydantic import BaseModel
from pydantic import ValidationError
from tqdm import tqdm

logger = logging.getLogger(__name__)


def highlight_error_in_original(original_str: str, task_name: str, error_detail: dict) -> str:
    """
    Directly highlights the error-causing text in the original JSON string based on the error type.
    For 'extra fields' errors, it attempts to colorize the specific field name in the original string.
    For 'missing fields', it appends a clear message indicating the missing field.
    """
    error_type = error_detail["type"]
    error_location = "->".join(map(str, error_detail["loc"]))
    if error_type == "value_error.extra":
        error_key = error_detail["loc"][-1]
        highlighted_key = style(error_key, fg="blue", bold=True)
        highlighted_str = original_str.replace(f'"{error_key}"', highlighted_key)
    elif error_type in ["value_error.missing", "value_error.any_str.min_length"]:
        missing_message = style(f"'{error_location}'", fg="blue", bold=True)
        highlighted_str = (
            f"{original_str}\n(Schema Error): Missing required parameter for task '{task_name}'"
            f" {missing_message}\n -> {original_str}"
        )
    else:
        error_key = error_detail["loc"][-1]
        highlighted_key = style(error_key, fg="blue", bold=True)
        highlighted_str = original_str.replace(f'"{error_key}"', highlighted_key)

    return highlighted_str


def format_validation_error(e: ValidationError, task_id, original_str: str) -> str:
    """
    Formats validation errors with appropriate highlights and returns a detailed error message.
    """
    error_messages = []
    for error in e.errors():
        error_message = f"(Schema Error): {error['msg']}"
        highlighted_str = highlight_error_in_original(original_str, task_id, error)
        error_messages.append(f"{error_message}\n -> {highlighted_str}")

    return "\n".join(error_messages)


def check_schema(schema: Type[BaseModel], options: dict, task_id: str, original_str: str) -> BaseModel:
    try:
        return schema(**options)
    except ValidationError as e:
        error_message = format_validation_error(e, task_id, original_str)
        # logger.error(error_message)
        raise ValueError(error_message) from e


def report_stage_statistics(
    stage_elapsed_times: defaultdict(list), total_trace_elapsed: float, abs_elapsed: float
) -> None:
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
    Reports the overall processing speed based on the number of pages and files processed.

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
    This function calculates the total elapsed time from the start of processing and reports the throughput
    in terms of pages and files processed per second.
    """

    total_elapsed_time_ns = time.time_ns() - start_time_ns
    total_elapsed_time_s = total_elapsed_time_ns / 1_000_000_000  # Convert nanoseconds to seconds

    throughput_pages = total_pages_processed / total_elapsed_time_s  # pages/sec
    throughput_files = total_files / total_elapsed_time_s  # files/sec

    logger.info(f"Processed {total_files} files in {total_elapsed_time_s:.2f} seconds.")
    logger.info(f"Total pages processed: {total_pages_processed}")
    logger.info(f"Throughput (Pages/sec): {throughput_pages:.2f}")
    logger.info(f"Throughput (Files/sec): {throughput_files:.2f}")


def report_statistics(
    start_time_ns: int,
    stage_elapsed_times: defaultdict,
    total_pages_processed: int,
    total_files: int,
    total_timeouts: int,
) -> None:
    """
    Aggregates and reports statistics for the entire processing session.

    Parameters
    ----------
    start_time_ns : int
        The nanosecond timestamp marking the start of the processing.
    stage_elapsed_times : defaultdict(list)
        A defaultdict where each key is a processing stage and each value is a list
        of elapsed times in nanoseconds for that stage.
    total_pages_processed : int
        The total number of pages processed during the session.
    total_files : int
        The total number of files processed during the session.
    total_timeouts : int
        The total number of timeouts that occurred during processing.

    Notes
    -----
    This function calculates the absolute elapsed time from the start of processing to the current
    time and the total time taken by all stages.
    """

    abs_elapsed = time.time_ns() - start_time_ns
    total_trace_elapsed = sum(sum(times) for times in stage_elapsed_times.values())
    report_stage_statistics(stage_elapsed_times, total_trace_elapsed, abs_elapsed)
    report_overall_speed(total_pages_processed, start_time_ns, total_files)
    logger.info(f"Total timeouts: {total_timeouts}")


def process_response(response, stage_elapsed_times):
    """
    Process the response to extract trace data and calculate elapsed time for each stage.

    Parameters
    ----------
    response : dict
        The response dictionary containing trace information for processing stages.
    stage_elapsed_times : defaultdict(list)
        A defaultdict to accumulate elapsed times for each processing stage.

    Notes
    -----
    The function iterates over trace data in the response, identifying entry and exit times for
    each stage, and calculates the elapsed time which is then appended to the respective stage in
    `stage_elapsed_times`.
    """

    trace_data = response.get("trace", {})
    for key, entry_time in trace_data.items():
        if "entry" in key:
            exit_key = key.replace("entry", "exit")
            exit_time = trace_data.get(exit_key)
            if exit_time:
                stage_name = key.split("::")[2]
                elapsed_time = exit_time - entry_time
                stage_elapsed_times[stage_name].append(elapsed_time)


def organize_documents_by_type(response_data):
    doc_map = {}
    for document in response_data:
        doc_meta = document["metadata"]
        # TODO: fix this. doc_meta can be a json string or a dict.
        if isinstance(doc_meta, str):
            doc_meta = json.loads(doc_meta)
        doc_content_metadata = doc_meta["content_metadata"]
        doc_type = doc_content_metadata["type"]
        if doc_type not in doc_map:
            doc_map[doc_type] = []
        doc_map[doc_type].append(document)
    return doc_map


def save_response_data(response, output_directory):
    if ("data" not in response) or (not response["data"]):
        return

    response_data = response["data"]

    if not isinstance(response_data, list) or len(response_data) == 0:
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

        with open(os.path.join(doc_type_path, output_name), "w") as f:
            f.write(json.dumps(documents, indent=2))


def create_job_specs_for_batch(files_batch: List[str], tasks: Dict, client: NvIngestClient) -> List[str]:
    """
    Creates JobSpecs for a batch of files and submits them, returning job IDs.
    """
    job_ids = []
    for file_name in files_batch:
        try:
            file_content, file_type = extract_file_content(file_name)  # Assume these are defined
        except ValueError as ve:
            logger.error(f"Error extracting content from {file_name}: {ve}")
            continue

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=file_name,
            source_name=file_name,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )

        logger.debug(f"Tasks: {tasks.keys()}")
        for task in tasks:
            logger.debug(f"Task: {task}")

        # TODO(Devin): Formalize this later, don't have time right now.
        if "split" in tasks:
            job_spec.add_task(tasks["split"])

        if f"extract_{file_type}" in tasks:
            job_spec.add_task(tasks[f"extract_{file_type}"])

        if "store" in tasks:
            job_spec.add_task(tasks["store"])

        if "caption" in tasks:
            job_spec.add_task(tasks["caption"])

        if "dedup" in tasks:
            job_spec.add_task(tasks["dedup"])

        if "filter" in tasks:
            job_spec.add_task(tasks["filter"])

        if "embed" in tasks:
            job_spec.add_task(tasks["embed"])

        if "vdb_upload" in tasks:
            job_spec.add_task(tasks["vdb_upload"])

        job_id = client.add_job(job_spec)
        job_ids.append(job_id)

    return job_ids


# TODO(Devin): Circle back on this, we can refactor to be better at keeping as many jobs in-flight as possible.
def create_and_process_jobs(
    files: List[str],
    client: NvIngestClient,
    tasks: Dict,
    output_directory: str,
    batch_size: int,
    timeout: int = 10,
    fail_on_error: bool = False,
):
    """
    Processes a list of files, creating and submitting jobs for each file, then fetching results.
    Manages retries for timeouts and logs failures for decoding errors.
    Limits the number of JobSpecs in memory to batch_size * 2. Progress is reported per file.
    """
    total_files = len(files)
    total_pages_processed = 0
    total_timeouts = 0
    trace_times = defaultdict(list)
    failed_jobs = []
    retry_job_ids = []
    job_id_map = {}
    retry_counts = defaultdict(int)
    file_page_counts = {file: estimate_page_count(file) for file in files}

    start_time_ns = time.time_ns()
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        processed = 0
        while (processed < len(files)) or retry_job_ids:
            # Process new batch of files or retry failed job IDs
            job_ids = []
            cur_job_count = 0
            if retry_job_ids:
                # logger.info(f"Adding retry jobs: {[job_id_map[jid] for jid in retry_job_ids]}")
                job_ids.extend(retry_job_ids)
                cur_job_count = len(job_ids)
                retry_job_ids = []  # Clear retry list after assigning

            if (cur_job_count < batch_size) and (processed < len(files)):
                new_job_count = min(batch_size - cur_job_count, len(files) - processed)
                batch_files = files[processed : processed + new_job_count]  # noqa: E203

                new_job_ids = create_job_specs_for_batch(batch_files, tasks, client)
                if len(new_job_ids) != new_job_count:
                    missing_jobs = new_job_count - len(new_job_ids)
                    error_msg = (
                        f"Missing {missing_jobs} job specs -- this is likely due to bad reads or file corruption"
                    )
                    if fail_on_error:
                        raise RuntimeError(error_msg)

                    logger.warning(error_msg)
                    pbar.update(missing_jobs)

                job_id_map.update({job_id: file for job_id, file in zip(new_job_ids, batch_files)})

                processed += new_job_count
                _ = client.submit_job_async(new_job_ids, "morpheus_task_queue")
                job_ids.extend(new_job_ids)

            futures_dict = client.fetch_job_result_async(job_ids, timeout=timeout, data_only=False)

            for future in as_completed(futures_dict.keys()):
                retry = False
                job_id = futures_dict[future]
                try:
                    result, _ = future.result()[0]
                    if ("annotations" in result) and result["annotations"]:
                        annotations = result["annotations"]
                        for key, value in annotations.items():
                            logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

                    valid_result, description = check_ingest_result(result)

                    if valid_result:
                        raise RuntimeError(f"Failed to process job {job_id}: {description}")

                    source_name = job_id_map[job_id]

                    if output_directory:
                        save_response_data(result, output_directory)

                    total_pages_processed += file_page_counts[source_name]
                    elapsed_time = (time.time_ns() - start_time_ns) / 1e9
                    pages_per_sec = total_pages_processed / elapsed_time if elapsed_time > 0 else 0
                    pbar.set_postfix(pages_per_sec=f"{pages_per_sec:.2f}")

                    process_response(result, trace_times)
                except TimeoutError:
                    source_name = job_id_map[job_id]
                    retry_counts[source_name] += 1

                    # TODO(Devin): not sure if we actually want a retry limit; if we don't get an actual failure
                    #  condition just assume we should continue waiting.
                    # if retry_counts[source_name] > 10:
                    #    logger.error(f"Timeout error for job {job_id} after {retry_counts[source_name]} retries.")
                    #    failed_jobs.append(f"{job_id}::{source_name}")
                    # else:
                    retry_job_ids.append(job_id)  # Add job_id back to retry list
                    total_timeouts += 1
                    retry = True
                except json.JSONDecodeError as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Decoding error for job {job_id}::{source_name} {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except RuntimeError as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Processing error was reported for {job_id}::{source_name} {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                except Exception as e:
                    source_name = job_id_map[job_id]
                    logger.error(f"Unhandled error occurred processing {job_id}:{source_name} {e}")
                    failed_jobs.append(f"{job_id}::{source_name}")
                finally:
                    if not retry:
                        pbar.update(1)

    if failed_jobs:
        logger.error(f"Failed jobs due to decoding or other errors: {failed_jobs}")

    return total_files, trace_times, total_pages_processed, total_timeouts


def get_valid_filename(name):
    """
    Taken from https://github.com/django/django/blob/main/django/utils/text.py.
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise ValueError("Could not derive file name from '%s'" % name)
    return s
