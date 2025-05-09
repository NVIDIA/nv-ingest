# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import os
import random
from enum import Enum
from pprint import pprint
from typing import Union, List, Any, Dict

import click
from nv_ingest_client.util.processing import check_schema
from nv_ingest_client.primitives.tasks import CaptionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreEmbedTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks.caption import CaptionTaskSchema
from nv_ingest_client.primitives.tasks.dedup import DedupTaskSchema
from nv_ingest_client.primitives.tasks.embed import EmbedTaskSchema
from nv_ingest_client.primitives.tasks.extract import ExtractTaskSchema
from nv_ingest_client.primitives.tasks.filter import FilterTaskSchema
from nv_ingest_client.primitives.tasks.split import SplitTaskSchema
from nv_ingest_client.primitives.tasks.store import StoreEmbedTaskSchema
from nv_ingest_client.primitives.tasks.store import StoreTaskSchema
from nv_ingest_client.util.util import generate_matching_files

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """
    Enum for specifying logging levels.

    Attributes
    ----------
    DEBUG : str
        Debug logging level.
    INFO : str
        Informational logging level.
    WARNING : str
        Warning logging level.
    ERROR : str
        Error logging level.
    CRITICAL : str
        Critical logging level.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ClientType(str, Enum):
    """
    Enum for specifying client types.

    Attributes
    ----------
    REST : str
        Represents a REST client.
    REDIS : str
        Represents a Redis client.
    KAFKA : str
        Represents a Kafka client.
    """

    REST = "REST"
    REDIS = "REDIS"
    KAFKA = "KAFKA"


# Example TaskId validation set
VALID_TASK_IDS = {"task1", "task2", "task3"}

_MODULE_UNDER_TEST = "nv_ingest_client.cli.util.click"


def debug_print_click_options(ctx: click.Context) -> None:
    """
    Retrieves all options from the Click context and pretty prints them.

    Parameters
    ----------
    ctx : click.Context
        The Click context object from which to retrieve the command options.
    """
    click_options: Dict[str, Any] = {}
    for param in ctx.command.params:
        if isinstance(param, click.Option):
            value = ctx.params[param.name]
            click_options[param.name] = value

    pprint(click_options)


def click_validate_file_exists(
    ctx: click.Context, param: click.Parameter, value: Union[str, List[str], None]
) -> List[str]:
    """
    Validates that the given file(s) exist.

    Parameters
    ----------
    ctx : click.Context
        The Click context.
    param : click.Parameter
        The parameter associated with the file option.
    value : Union[str, List[str], None]
        A file path or a list of file paths.

    Returns
    -------
    List[str]
        A list of validated file paths.

    Raises
    ------
    click.BadParameter
        If any file path does not exist.
    """
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


# Define a union type for all supported task types.
TaskType = Union[
    CaptionTask,
    DedupTask,
    EmbedTask,
    ExtractTask,
    FilterTask,
    SplitTask,
    StoreEmbedTask,
    StoreTask,
]


def parse_task_options(task_id: str, options_str: str) -> Dict[str, Any]:
    """
    Parse the task options string as JSON.

    Parameters
    ----------
    task_id : str
        The identifier of the task for which options are being parsed.
    options_str : str
        The string containing JSON options.

    Returns
    -------
    Dict[str, Any]
        The parsed options as a dictionary.

    Raises
    ------
    ValueError
        If the JSON string is not well formatted. The error message will indicate the task,
        the error details (e.g., expected property format), and show the input that was provided.
    """
    try:
        return json.loads(options_str)
    except json.JSONDecodeError as e:
        error_message = (
            f"Invalid JSON format for task '{task_id}': {e.msg} at line {e.lineno} column {e.colno} (char {e.pos}). "
            f"Input was: {options_str}"
        )
        raise ValueError(error_message)


def click_validate_task(ctx: click.Context, param: click.Parameter, value: List[str]) -> Dict[str, TaskType]:
    """
    Validates and processes task definitions provided as strings.

    Each task definition should be in the format "<task_id>:<json_options>".
    If the separator ':' is missing, an empty JSON options dictionary is assumed.
    The function uses a schema check (via check_schema) for validation and
    instantiates the corresponding task.

    Parameters
    ----------
    ctx : click.Context
        The Click context.
    param : click.Parameter
        The parameter associated with the task option.
    value : List[str]
        A list of task strings to validate.

    Returns
    -------
    Dict[str, TaskType]
        A dictionary mapping task IDs to their corresponding task objects.

    Raises
    ------
    click.BadParameter
        If any task fails validation (including malformed JSON) or if duplicate tasks are detected.
    """
    validated_tasks: Dict[str, TaskType] = {}
    validation_errors: List[str] = []

    for task_str in value:
        task_split = task_str.split(":", 1)
        if len(task_split) != 2:
            task_id, json_options = task_str, "{}"
        else:
            task_id, json_options = task_split

        try:
            options: Dict[str, Any] = parse_task_options(task_id, json_options)

            if task_id == "split":
                task_options = check_schema(SplitTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, SplitTask(**task_options.model_dump()))]
            elif task_id == "extract":
                task_options = check_schema(ExtractTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}_{task_options.document_type}"
                new_task = [(new_task_id, ExtractTask(**task_options.model_dump()))]
            elif task_id == "store":
                task_options = check_schema(StoreTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, StoreTask(**task_options.model_dump()))]
            elif task_id == "store_embedding":
                task_options = check_schema(StoreEmbedTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, StoreEmbedTask(**task_options.model_dump()))]
            elif task_id == "caption":
                task_options = check_schema(CaptionTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, CaptionTask(**task_options.model_dump()))]
            elif task_id == "dedup":
                task_options = check_schema(DedupTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, DedupTask(**task_options.model_dump()))]
            elif task_id == "filter":
                task_options = check_schema(FilterTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, FilterTask(**task_options.model_dump()))]
            elif task_id == "embed":
                task_options = check_schema(EmbedTaskSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, EmbedTask(**task_options.model_dump()))]
            else:
                raise ValueError(f"Unsupported task type: {task_id}")

            if new_task_id in validated_tasks:
                raise ValueError(f"Duplicate task detected: {new_task_id}")

            logger.debug("Adding task: %s", new_task_id)
            for task_tuple in new_task:
                validated_tasks[task_tuple[0]] = task_tuple[1]
        except ValueError as e:
            validation_errors.append(str(e))

    if validation_errors:
        error_message = "\n".join(validation_errors)
        raise click.BadParameter(error_message)

    return validated_tasks


def click_validate_batch_size(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """
    Validates that the batch size is at least 1.

    Parameters
    ----------
    ctx : click.Context
        The Click context.
    param : click.Parameter
        The parameter associated with the batch size option.
    value : int
        The batch size value provided.

    Returns
    -------
    int
        The validated batch size.

    Raises
    ------
    click.BadParameter
        If the batch size is less than 1.
    """
    if value < 1:
        raise click.BadParameter("Batch size must be >= 1.")
    return value


def pre_process_dataset(dataset_json: str, shuffle_dataset: bool) -> List[str]:
    """
    Loads a dataset from a JSON file and optionally shuffles the list of files.

    Parameters
    ----------
    dataset_json : str
        The path to the dataset JSON file.
    shuffle_dataset : bool
        Whether to shuffle the dataset before processing.

    Returns
    -------
    List[str]
        The list of file paths from the dataset. If 'shuffle_dataset' is True,
        the list will be shuffled.

    Raises
    ------
    click.BadParameter
        If the dataset file is not found or if its contents are not valid JSON.
    """
    try:
        with open(dataset_json, "r") as f:
            file_source = json.load(f)
    except FileNotFoundError:
        raise click.BadParameter(f"Dataset JSON file not found: {dataset_json}")
    except json.JSONDecodeError:
        raise click.BadParameter(f"Invalid JSON format in file: {dataset_json}")

    file_source = file_source.get("sampled_files", [])
    if shuffle_dataset:
        random.shuffle(file_source)

    return file_source


def click_match_and_validate_files(ctx: click.Context, param: click.Parameter, value: List[str]) -> List[str]:
    """
    Matches and validates files based on the provided file source patterns.

    Parameters
    ----------
    ctx : click.Context
        The Click context.
    param : click.Parameter
        The parameter associated with the file matching option.
    value : List[str]
        A list of file source patterns to match against.

    Returns
    -------
    List[str]
        A list of matching file paths. If no files match, an empty list is returned.
    """
    if not value:
        return []

    matching_files = list(generate_matching_files(value))
    if not matching_files:
        logger.warning("No files found matching the specified patterns.")
        return []

    return matching_files
