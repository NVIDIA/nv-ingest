# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import glob
import json
import logging
import os
import random
from enum import Enum
from pprint import pprint

import click
from nv_ingest_client.cli.util.processing import check_schema
from nv_ingest_client.primitives.tasks import CaptionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import VdbUploadTask
from nv_ingest_client.primitives.tasks.caption import CaptionTaskSchema
from nv_ingest_client.primitives.tasks.dedup import DedupTaskSchema
from nv_ingest_client.primitives.tasks.embed import EmbedTaskSchema
from nv_ingest_client.primitives.tasks.extract import ExtractTaskSchema
from nv_ingest_client.primitives.tasks.filter import FilterTaskSchema
from nv_ingest_client.primitives.tasks.split import SplitTaskSchema
from nv_ingest_client.primitives.tasks.store import StoreTaskSchema
from nv_ingest_client.primitives.tasks.vdb_upload import VdbUploadTaskSchema

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ClientType(str, Enum):
    REST = "REST"
    REDIS = "REDIS"
    KAFKA = "KAFKA"


# Example TaskId validation set
VALID_TASK_IDS = {"task1", "task2", "task3"}

_MODULE_UNDER_TEST = "nv_ingest_client.cli.util.click"


def debug_print_click_options(ctx):
    """
    Retrieves all options from the Click context and pretty prints them.

    Parameters
    ----------
    ctx : click.Context
        The Click context object from which to retrieve the command options.
    """
    click_options = {}
    for param in ctx.command.params:
        if isinstance(param, click.Option):
            value = ctx.params[param.name]
            click_options[param.name] = value

    pprint(click_options)


def click_validate_file_exists(ctx, param, value):
    if not value:
        return []

    if isinstance(value, str):
        value = [value]
    else:
        value = list(value)

    for filepath in value:
        if not os.path.exists(filepath):
            raise click.BadParameter(f"File does not exist: {filepath}")

    return value


def click_validate_task(ctx, param, value):
    validated_tasks = {}
    validation_errors = []

    for task_str in value:
        task_split = task_str.split(":", 1)
        if len(task_split) != 2:
            task_id, json_options = task_str, "{}"
        else:
            task_id, json_options = task_split

        try:
            options = json.loads(json_options)

            if task_id == "split":
                task_options = check_schema(SplitTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = SplitTask(**task_options.dict())
            elif task_id == "extract":
                task_options = check_schema(ExtractTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}_{task_options.document_type}"
                new_task = ExtractTask(**task_options.dict())
            elif task_id == "store":
                task_options = check_schema(StoreTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = StoreTask(**task_options.dict())
            elif task_id == "caption":
                task_options = check_schema(CaptionTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = CaptionTask(**task_options.dict())
            elif task_id == "dedup":
                task_options = check_schema(DedupTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = DedupTask(**task_options.dict())
            elif task_id == "filter":
                task_options = check_schema(FilterTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = FilterTask(**task_options.dict())
            elif task_id == "embed":
                task_options = check_schema(EmbedTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = EmbedTask(**task_options.dict())
            elif task_id == "vdb_upload":
                task_options = check_schema(VdbUploadTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = VdbUploadTask(**task_options.dict())

            else:
                raise ValueError(f"Unsupported task type: {task_id}")

            logger.debug("Adding task: %s", new_task_id)
            validated_tasks[new_task_id] = new_task
        except ValueError as e:
            validation_errors.append(str(e))

    if validation_errors:
        # Aggregate error messages with original values highlighted
        error_message = "\n".join(validation_errors)
        # logger.error(error_message)
        raise click.BadParameter(error_message)

    return validated_tasks


def click_validate_batch_size(ctx, param, value):
    if value < 1:
        raise click.BadParameter("Batch size must be >= 1.")
    return value


def pre_process_dataset(dataset_json: str, shuffle_dataset: bool):
    """
    Loads a dataset from a JSON file and optionally shuffles the list of files contained within.

    Parameters
    ----------
    dataset_json : str
        The path to the dataset JSON file.
    shuffle_dataset : bool, optional
        Whether to shuffle the dataset before processing. Defaults to True.

    Returns
    -------
    list
        The list of files from the dataset, possibly shuffled.
    """
    try:
        with open(dataset_json, "r") as f:
            file_source = json.load(f)
    except FileNotFoundError:
        raise click.BadParameter(f"Dataset JSON file not found: {dataset_json}")
    except json.JSONDecodeError:
        raise click.BadParameter(f"Invalid JSON format in file: {dataset_json}")

    # Extract the list of files and optionally shuffle them
    file_source = file_source.get("sampled_files", [])

    if shuffle_dataset:
        random.shuffle(file_source)

    return file_source


def _generate_matching_files(file_sources):
    """
    Generates a list of file paths that match the given patterns specified in file_sources.

    Parameters
    ----------
    file_sources : list of str
        A list containing the file source patterns to match against.

    Returns
    -------
    generator
        A generator yielding paths to files that match the specified patterns.

    Notes
    -----
    This function utilizes glob pattern matching to find files that match the specified patterns.
    It yields each matching file path, allowing for efficient processing of potentially large
    sets of files.
    """

    files = [
        file_path
        for pattern in file_sources
        for file_path in glob.glob(pattern, recursive=True)
        if os.path.isfile(file_path)
    ]
    for file_path in files:
        yield file_path


def click_match_and_validate_files(ctx, param, value):
    """
    Matches and validates files based on the provided file source patterns.

    Parameters
    ----------
    value : list of str
        A list containing file source patterns to match against.

    Returns
    -------
    list of str or None
        A list of matching file paths if any matches are found; otherwise, None.
    """

    if not value:
        return []

    matching_files = list(_generate_matching_files(value))
    if not matching_files:
        logger.warning("No files found matching the specified patterns.")
        return []

    return matching_files
