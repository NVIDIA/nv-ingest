# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import time
from collections import defaultdict
from statistics import mean
from statistics import median
from typing import Any

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
