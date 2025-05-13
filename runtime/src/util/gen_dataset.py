# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import os
import random
from collections import defaultdict

import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


def format_size(size_in_bytes):
    """
    Formats a size in bytes into a human-readable string with a maximum of three digits.

    Parameters:
    ----------
    size_in_bytes : int
        The size in bytes to format.

    Returns:
    -------
    str
        The formatted size string with appropriate size suffix.
    """
    for unit in ["", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(size_in_bytes) < 1024.0:
            return f"{size_in_bytes:3.1f}{unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f}YB"


def parse_size(size):
    """Parse the size string with optional suffix (KB, MB) into bytes."""
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
    if size.isdigit():
        return int(size)
    unit = size[-2:].upper()
    number = size[:-2]
    if unit in units and number.isdigit():
        return int(number) * units[unit]
    raise ValueError("Size must be in the format <number>[KB|MB|GB].")


def load_or_scan_files(source_directory, cache_file=None):
    """Load file list from cache or scan the source directory."""
    if cache_file and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    file_list = defaultdict(list)
    all_files = [os.path.join(root, file) for root, _, files in os.walk(source_directory) for file in files]
    for file_path in tqdm(all_files, desc="Scanning files", leave=True):
        ext = os.path.splitext(file_path)[-1].lower().strip(".")
        file_list[ext].append(file_path)

    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(file_list, f)

    return file_list


def sample_files(file_list, samples, total_size_bytes, with_replacement=True):
    """
    Samples files to meet a target total size, respecting the specified proportions for each
    file type, and allows for sampling with or without replacement.

    Parameters
    ----------
    file_list : dict
        A dictionary mapping file types to lists of file paths.
    samples : dict
        A dictionary mapping file types to their target proportions of the total size.
    total_size_bytes : int
        The target total size of the sampled files in bytes.
    with_replacement : bool, optional
        Flag to determine if sampling should be done with replacement. Defaults to True.

    Returns
    -------
    dict
        A dictionary containing 'sampled_files', a list of paths to sampled files,
        and 'metadata', a dictionary with details about the sampling process.
    """
    sampled_files = []
    metadata = {
        "total_sampled_size_bytes": 0,
        "file_type_proportions": {},
        "sampling_method": ("with_replacement" if with_replacement else "without_replacement"),
    }
    selected_files = set()

    for ftype, proportion in samples.items():
        ftype_target_bytes = total_size_bytes * (proportion / 100)
        available_files = file_list.get(ftype, [])

        if not with_replacement:
            available_files = [f for f in available_files if f not in selected_files]

        ftype_sampled_size = 0

        while available_files and ftype_sampled_size < ftype_target_bytes:
            if with_replacement:
                file_path = random.choice(available_files)
            else:
                file_path = available_files.pop(0)  # Treat as a queue for without replacement
                selected_files.add(file_path)

            file_size = os.path.getsize(file_path)

            if ftype_sampled_size + file_size > ftype_target_bytes:
                # Stop if adding this file exceeds the target size for this file type
                if not with_replacement:
                    # Put the file back if we're not sampling with replacement
                    available_files.insert(0, file_path)
                    selected_files.remove(file_path)
                break

            sampled_files.append(file_path)
            ftype_sampled_size += file_size

        metadata["file_type_proportions"][ftype] = {
            "target_proportion": proportion,
            "achieved_size_bytes": ftype_sampled_size,
        }
        metadata["total_sampled_size_bytes"] += ftype_sampled_size

    return {"sampled_files": sampled_files, "metadata": metadata}


def process_samples(sample_options):
    """
    Converts a list of 'key=value' strings from the --sample option into a dictionary,
    mapping file extensions to their corresponding proportions for sampling.

    Parameters
    ----------
    sample_options : tuple of str
        A tuple containing strings, each in the format 'key=value', where 'key'
        is a file extension (without the dot) and 'value' is the proportion
        (as an integer) of the total size that files of this type should occupy.

    Returns
    -------
    dict
        A dictionary where keys are file extensions (as strings) and values are the
        proportions (as integers) of the total size that files of each type should occupy.

    Raises
    ------
    ValueError
        If any string in `sample_options` does not follow the 'key=value' format or
        if the 'value' part cannot be converted to an integer.
    """
    sample_dict = {}
    for sample_str in sample_options:
        try:
            key, value = sample_str.split("=")
            sample_dict[key] = int(value)
        except ValueError:
            raise ValueError(f"Invalid sample format: {sample_str}. Expected format: 'key=value'.")
    return sample_dict


def validate_output_file(output_file_path):
    """
    Validates the output file by calculating and logging the total bytes for each file type
    identified in the sampled files listed within the specified JSON file, and checks if the
    actual proportions match the expected proportions within a 5% tolerance.
    """
    try:
        with open(output_file_path, "r") as f:
            data = json.load(f)
            sampled_files = data["sampled_files"]
            metadata = data["metadata"]
    except FileNotFoundError:
        logger.error(f"Output file '{output_file_path}' not found.")
        return
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from '{output_file_path}'.")
        return
    except KeyError:
        logger.error("Missing 'sampled_files' or 'metadata' in the output file.")
        return

    total_bytes_by_type = defaultdict(int)

    for file_path in sampled_files:
        file_type = os.path.splitext(file_path)[-1].lower()
        file_type = file_type[1:] if file_type.startswith(".") else "unknown"
        file_size = os.path.getsize(file_path)
        total_bytes_by_type[file_type] += file_size

    total_sampled_size = sum(total_bytes_by_type.values())
    expected_proportions = metadata.get("sampling_proportions", {})
    sampling_method = metadata.get("sampling_method", "unknown")

    # Validate proportions
    for file_type, expected_prop in expected_proportions.items():
        actual_size = total_bytes_by_type.get(file_type, 0)
        actual_prop = (actual_size / total_sampled_size) * 100 if total_sampled_size else 0
        if not (expected_prop * 0.95 <= actual_prop <= expected_prop * 1.05):
            logger.warning(
                f"Proportion of {file_type} files is off by more than 5%: "
                f"Expected {expected_prop}%, got {actual_prop:.2f}%."
            )

    # Log total sizes
    for file_type, total_size in total_bytes_by_type.items():
        logger.info(f"Total size for {file_type}: {format_size(total_size)}")
    logger.info(f"Total size of all sampled files: {format_size(total_sampled_size)}")
    logger.info(f"Sampling method used: {sampling_method}")


@click.command()
@click.option("--source_directory", required=True, help="Path to the source directory.")
@click.option("--size", required=True, help="Total size of files to sample, e.g., 1024, 1KB, 1MB.")
@click.option("--sample", multiple=True, type=str, help="File type and proportion, e.g., pdf=40.")
@click.option("--cache_file", help="Path to cache the file list as JSON.")
@click.option("--output_file", help="Path to output the sampled file list as JSON.")
@click.option(
    "--validate-output",
    is_flag=True,
    help="Re-validate the 'output_file' JSON and log total bytes for each file type.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Sets the logging level.",
)
@click.option(
    "--with-replacement",
    is_flag=True,
    default=True,
    help="Sample with replacement. Files can be selected multiple times.",
)
def main(
    source_directory,
    size,
    sample,
    cache_file,
    output_file,
    validate_output,
    log_level,
    with_replacement,
):
    """
    Samples files from a source directory based on specified proportions and total size,
    optionally caches the file list, outputs a sampled file list, and validates output.

    Parameters
    ----------
    source_directory : str
        The directory to scan for files to sample.
    size : str
        The total size of the files to sample, with optional suffix (KB, MB, GB).
    sample : tuple of str
        Each string specifies a file type and its target proportion of the total size
        (e.g., "pdf=40").
    cache_file : str, optional
        If specified, caches the scanned file list as JSON at this path.
    output_file : str, optional
        If specified, outputs the list of sampled files as JSON at this path.
    validate_output : bool
        If set, validates the `output_file` by logging total bytes for each file type.
    log_level : str
        The logging level for the script's output
        ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    with_replacement : bool
        If set, samples files with replacement. Files can be selected multiple times.

    Notes
    -----
    The script performs a sampling process that respects the specified size and type proportions,
    generates a detailed file list, and provides options for caching and validation to facilitate
    efficient data handling and integrity checking.
    """  # Configure logging

    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        total_size_bytes = parse_size(size)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    samples = {}
    for sample_str in sample:
        try:
            key, value = sample_str.split("=")
            samples[key] = int(value)
        except ValueError:
            click.echo(f"Error: Invalid sample format: {sample_str}. Expected format: 'key=value'.")
            return

    if sum(samples.values()) != 100:
        click.echo("Error: The sum of the sample proportions must be 100.")
        return

    file_list = load_or_scan_files(source_directory, cache_file)
    sampled_files = sample_files(file_list, samples, total_size_bytes, with_replacement)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(sampled_files, f)

    click.echo(f"Sampled files written to {output_file}" if output_file else "Sampling complete.")

    if validate_output:
        if output_file:
            validate_output_file(output_file)
        else:
            logger.warning("No output file specified for validation.")
    return


if __name__ == "__main__":
    main()
