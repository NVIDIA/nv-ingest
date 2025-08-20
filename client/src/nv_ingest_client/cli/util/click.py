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

from nv_ingest_api.internal.enums.common import PipelinePhase
from nv_ingest_api.util.introspection.function_inspect import infer_udf_function_name
from nv_ingest_client.util.processing import check_schema
from nv_ingest_client.primitives.tasks import CaptionTask
from nv_ingest_client.primitives.tasks import DedupTask
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import FilterTask
from nv_ingest_client.primitives.tasks import InfographicExtractionTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.primitives.tasks import StoreEmbedTask
from nv_ingest_client.primitives.tasks import StoreTask
from nv_ingest_client.primitives.tasks import UDFTask
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskCaptionSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskDedupSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskEmbedSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskExtractSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskFilterSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskInfographicExtraction
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskSplitSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskStoreEmbedSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskStoreSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskUDFSchema
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
    InfographicExtractionTask,
    SplitTask,
    StoreEmbedTask,
    StoreTask,
    UDFTask,
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
        options = json.loads(options_str)

        # Convert string boolean values to actual booleans for extract tasks
        if task_id == "extract":
            boolean_fields = [
                "extract_text",
                "extract_images",
                "extract_tables",
                "extract_charts",
                "extract_infographics",
                "extract_page_as_image",
            ]
            for field in boolean_fields:
                if field in options:
                    value = options[field]
                    if isinstance(value, str):
                        if value.lower() in ("true", "1", "yes", "on"):
                            options[field] = True
                        elif value.lower() in ("false", "0", "no", "off"):
                            options[field] = False
                        else:
                            raise ValueError(
                                f"Invalid boolean value for {field}: '{value}'. Use true/false, 1/0, yes/no, or on/off."
                            )

        return options
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
                task_options = check_schema(IngestTaskSplitSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, SplitTask(**task_options.model_dump()))]
            elif task_id == "extract":
                # Map CLI parameters to API schema structure
                method = options.pop("extract_method", None)
                if method is None:
                    method = "pdfium"  # Default fallback

                # Build params dict for API schema
                params = {k: v for k, v in options.items() if k != "document_type"}

                # Validate with API schema
                api_options = {
                    "document_type": options.get("document_type"),
                    "method": method,
                    "params": params,
                }
                task_options = check_schema(IngestTaskExtractSchema, api_options, task_id, json_options)
                new_task_id = f"{task_id}_{task_options.document_type.value}"

                # Create ExtractTask with original CLI parameters
                extract_task_params = {
                    "document_type": task_options.document_type,
                    "extract_method": task_options.method,
                    **task_options.params,
                }

                # Start with the main extract task
                new_task = [(new_task_id, ExtractTask(**extract_task_params))]

                # Add ChartExtractionTask if extract_charts is True
                if task_options.params.get("extract_charts", False):
                    from nv_ingest_client.primitives.tasks import ChartExtractionTask

                    chart_task_id = "chart_data_extract"
                    chart_params = {"params": {}}  # ChartExtractionTask takes params dict
                    new_task.append((chart_task_id, ChartExtractionTask(chart_params)))

                # Add TableExtractionTask if extract_tables is True
                if task_options.params.get("extract_tables", False):
                    from nv_ingest_client.primitives.tasks import TableExtractionTask

                    table_task_id = "table_data_extract"
                    new_task.append((table_task_id, TableExtractionTask()))
            elif task_id == "store":
                task_options = check_schema(IngestTaskStoreSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, StoreTask(**task_options.model_dump()))]
            elif task_id == "store_embedding":
                task_options = check_schema(IngestTaskStoreEmbedSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, StoreEmbedTask(**task_options.model_dump()))]
            elif task_id == "caption":
                task_options = check_schema(IngestTaskCaptionSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                # Extract individual parameters from API schema for CaptionTask constructor
                caption_params = {
                    "api_key": task_options.api_key,
                    "endpoint_url": task_options.endpoint_url,
                    "prompt": task_options.prompt,
                    "model_name": task_options.model_name,
                }
                new_task = [(new_task_id, CaptionTask(**caption_params))]
            elif task_id == "dedup":
                task_options = check_schema(IngestTaskDedupSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                # Extract individual parameters from API schema for DedupTask constructor
                dedup_params = {
                    "content_type": task_options.content_type,
                    "filter": task_options.params.filter,
                }
                new_task = [(new_task_id, DedupTask(**dedup_params))]
            elif task_id == "filter":
                task_options = check_schema(IngestTaskFilterSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                # Extract individual parameters from API schema for FilterTask constructor
                filter_params = {
                    "content_type": task_options.content_type,
                    "min_size": task_options.params.min_size,
                    "max_aspect_ratio": task_options.params.max_aspect_ratio,
                    "min_aspect_ratio": task_options.params.min_aspect_ratio,
                    "filter": task_options.params.filter,
                }
                new_task = [(new_task_id, FilterTask(**filter_params))]
            elif task_id == "embed":
                task_options = check_schema(IngestTaskEmbedSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, EmbedTask(**task_options.model_dump()))]
            elif task_id == "infographic":
                task_options = check_schema(IngestTaskInfographicExtraction, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, InfographicExtractionTask(**task_options.model_dump()))]
            elif task_id == "udf":
                # Validate mutual exclusivity of target_stage and phase
                has_target_stage = "target_stage" in options and options["target_stage"] is not None
                has_phase = "phase" in options and options["phase"] is not None

                if has_target_stage and has_phase:
                    raise ValueError(
                        "UDF task cannot specify both 'target_stage' and 'phase'. Please specify only one."
                    )
                elif not has_target_stage and not has_phase:
                    raise ValueError("UDF task must specify either 'target_stage' or 'phase'.")

                # Pre-process UDF task options to convert phase names to integers
                if "phase" in options and isinstance(options["phase"], str):
                    # Convert phase string to integer using the same logic as UDFTask
                    phase_str = options["phase"].upper()
                    phase_aliases = {
                        "PRE_PROCESSING": PipelinePhase.PRE_PROCESSING,
                        "PREPROCESSING": PipelinePhase.PRE_PROCESSING,
                        "PRE": PipelinePhase.PRE_PROCESSING,
                        "EXTRACTION": PipelinePhase.EXTRACTION,
                        "EXTRACT": PipelinePhase.EXTRACTION,
                        "POST_PROCESSING": PipelinePhase.POST_PROCESSING,
                        "POSTPROCESSING": PipelinePhase.POST_PROCESSING,
                        "POST": PipelinePhase.POST_PROCESSING,
                        "MUTATION": PipelinePhase.MUTATION,
                        "MUTATE": PipelinePhase.MUTATION,
                        "TRANSFORM": PipelinePhase.TRANSFORM,
                        "RESPONSE": PipelinePhase.RESPONSE,
                        "RESP": PipelinePhase.RESPONSE,
                    }

                    if phase_str in phase_aliases:
                        options["phase"] = phase_aliases[phase_str].value
                    else:
                        raise ValueError(f"Invalid phase name: {options['phase']}")

                # Try to infer udf_function_name if not provided
                if "udf_function_name" not in options or not options["udf_function_name"]:
                    udf_function = options.get("udf_function", "")
                    if udf_function:
                        inferred_name = infer_udf_function_name(udf_function)
                        if inferred_name:
                            options["udf_function_name"] = inferred_name
                            logger.info(f"Inferred UDF function name: {inferred_name}")
                        else:
                            raise ValueError(
                                f"Could not infer UDF function name from '{udf_function}'. "
                                "Please specify 'udf_function_name' explicitly."
                            )

                task_options = check_schema(IngestTaskUDFSchema, options, task_id, json_options)
                new_task_id = f"{task_id}"
                new_task = [(new_task_id, UDFTask(**task_options.model_dump()))]
            else:
                raise ValueError(f"Unsupported task type: {task_id}")

            # Check for duplicate tasks - now allowing multiple tasks of the same type
            if new_task_id in validated_tasks:
                logger.debug(f"Multiple tasks detected for {new_task_id}, storing as list")

            logger.debug("Adding task: %s", new_task_id)
            for task_tuple in new_task:
                if task_tuple[0] in validated_tasks:
                    # Convert single task to list if needed, then append
                    existing_task = validated_tasks[task_tuple[0]]
                    if not isinstance(existing_task, list):
                        validated_tasks[task_tuple[0]] = [existing_task]
                    validated_tasks[task_tuple[0]].append(task_tuple[1])
                else:
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
