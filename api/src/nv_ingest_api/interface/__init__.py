# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import functools
import inspect
import pprint
from typing import Dict, Any, Optional, List

from pydantic import BaseModel

from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFiumConfigSchema, NemoRetrieverParseConfigSchema
from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

logger = logging.getLogger(__name__)

## CONFIG_SCHEMAS is a global dictionary that maps extraction methods to Pydantic schemas.
CONFIG_SCHEMAS: Dict[str, Any] = {
    "adobe": PDFiumConfigSchema,
    "llama": PDFiumConfigSchema,
    "nemoretriever_parse": NemoRetrieverParseConfigSchema,
    "pdfium": PDFiumConfigSchema,
    "tika": PDFiumConfigSchema,
    "unstructured_io": PDFiumConfigSchema,
}


def _build_config_from_schema(schema_class: type[BaseModel], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build and validate a configuration dictionary from the provided arguments using a Pydantic schema.

    This function filters the supplied arguments to include only those keys defined in the given
    Pydantic schema (using Pydantic v2's `model_fields`), instantiates the schema for validation,
    and returns the validated configuration as a dictionary.

    Parameters
    ----------
    schema_class : type[BaseModel]
        The Pydantic BaseModel subclass used for validating the configuration.
    args : dict
        A dictionary of arguments from which to extract and validate configuration data.

    Returns
    -------
    dict
        A dictionary containing the validated configuration data as defined by the schema.

    Raises
    ------
    pydantic.ValidationError
        If the provided arguments do not conform to the schema.
    """
    field_names = schema_class.model_fields.keys()
    config_data = {k: v for k, v in args.items() if k in field_names}
    # Instantiate the schema to perform validation, then return the model's dictionary representation.

    return schema_class(**config_data).dict()


def extraction_interface_relay_constructor(api_fn, task_keys: Optional[List[str]] = None):
    """
    Decorator for constructing and validating configuration using Pydantic schemas.

    This decorator wraps a user-facing interface function. It extracts common task parameters
    (using the provided task_keys, or defaults if not specified) and method-specific configuration
    parameters based on a required 'extract_method' keyword argument. It then uses the corresponding
    Pydantic schema (from the global CONFIG_SCHEMAS registry) to validate and build a method-specific
    configuration. The resulting composite configuration, along with the extraction ledger and
    execution trace log, is passed to the backend API function.

    Parameters
    ----------
    api_fn : callable
        The backend API function that will be called with the extraction ledger, the task configuration
        dictionary, the extractor configuration, and the execution trace log. This function must conform
        to the signature:

            extract_primitives_from_pdf_internal(df_extraction_ledger: pd.DataFrame,
                                        task_config: Dict[str, Any],
                                        extractor_config: Any,
                                        execution_trace_log: Optional[List[Any]] = None)
    task_keys : list of str, optional
        A list of keyword names that should be extracted from the user function as common task parameters.
        If not provided, defaults to ["extract_text", "extract_images", "extract_tables", "extract_charts"].

    Returns
    -------
    callable
        A wrapped function that builds and validates the configuration before invoking the backend API function.

    Raises
    ------
    ValueError
        If the extraction method specified is not supported (i.e., no corresponding Pydantic schema exists
        in CONFIG_SCHEMAS), if api_fn does not conform to the expected signature, or if the required
        'extract_method' parameter is not provided.
    """
    # Verify that api_fn conforms to the expected signature.
    try:
        # Try binding four arguments: ledger, task_config, extractor_config, and execution_trace_log.
        inspect.signature(api_fn).bind("dummy_ledger", {"dummy": True}, {"dummy": True}, {})
    except TypeError as e:
        raise ValueError(
            "api_fn must conform to the signature: "
            "extract_primitives_from_pdf(df_extraction_ledger, task_config, extractor_config, execution_trace_log)"
        ) from e

    if task_keys is None:
        task_keys = []

    def decorator(user_fn):
        @functools.wraps(user_fn)
        def wrapper(*args, **kwargs):
            # Use bind_partial so that missing required arguments can be handled gracefully.
            sig = inspect.signature(user_fn)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # The first parameter is assumed to be the extraction ledger.
            param_names = list(sig.parameters.keys())
            if param_names[0] not in bound.arguments:
                raise ValueError("Missing required ledger argument.")
            ledger = bound.arguments[param_names[0]]

            # Process reserved 'execution_trace_log'.
            execution_trace_log = bound.arguments.get("execution_trace_log", None)
            if execution_trace_log is None:
                execution_trace_log = {}  # Replace None with an empty dict.
            if "execution_trace_log" in bound.arguments:
                del bound.arguments["execution_trace_log"]

            # Ensure that 'extract_method' is provided.
            if "extract_method" not in bound.arguments or bound.arguments["extract_method"] is None:
                raise ValueError("The 'extract_method' parameter is required.")
            extract_method = bound.arguments["extract_method"]
            del bound.arguments["extract_method"]

            # Extract common task parameters using the specified task_keys.
            task_params = {key: bound.arguments[key] for key in task_keys if key in bound.arguments}
            task_params["extract_method"] = extract_method
            task_config = {"params": task_params}

            # Look up the appropriate Pydantic schema.
            schema_class = CONFIG_SCHEMAS.get(extract_method)
            if schema_class is None:
                raise ValueError(f"Unsupported extraction method: {extract_method}")

            # Build the method-specific configuration using the schema class.
            extraction_config_dict = _build_config_from_schema(schema_class, bound.arguments)

            # Create a Pydantic object instead of a dictionary for the specific extractor config
            extractor_schema = None
            try:
                # Find the appropriate extractor schema class based on the extraction method
                extractor_schema_name = f"{extract_method.capitalize()}ExtractorSchema"
                extractor_schema_class = globals().get(extractor_schema_name)

                if extractor_schema_class is None:
                    # Try another common naming pattern
                    extractor_schema_name = f"{extract_method.upper()}ExtractorSchema"
                    extractor_schema_class = globals().get(extractor_schema_name)

                if extractor_schema_class is None:
                    # Final fallback attempt with camelCase
                    extractor_schema_name = f"{extract_method[0].upper() + extract_method[1:]}ExtractorSchema"
                    extractor_schema_class = globals().get(extractor_schema_name)

                if extractor_schema_class is not None:
                    # Create the extractor schema with the method-specific config
                    config_key = f"{extract_method}_config"
                    extractor_schema = extractor_schema_class(**{config_key: extraction_config_dict})
                else:
                    logger.warning(f"Could not find extractor schema class for method: {extract_method}")
            except Exception as e:
                logger.warning(f"Error creating extractor schema: {str(e)}")
                # Fall back to dictionary approach if schema creation fails
                extractor_schema = {f"{extract_method}_config": extraction_config_dict}

            # If schema creation failed, fall back to dictionary
            if extractor_schema is None:
                extractor_schema = {f"{extract_method}_config": extraction_config_dict}

            # Log the task and extractor configurations for debugging (sanitized)
            logger.debug("\n" + "=" * 80)
            logger.debug(f"DEBUG - API Function: {api_fn.__name__}")
            logger.debug(f"DEBUG - Extract Method: {extract_method}")
            logger.debug("-" * 80)

            # Sanitize and format the task config as a string and log it
            sanitized_task_config = sanitize_for_logging(task_config)
            task_config_str = pprint.pformat(sanitized_task_config, width=100, sort_dicts=False)
            logger.debug(f"DEBUG - Task Config (sanitized):\n{task_config_str}")
            logger.debug("-" * 80)

            # Sanitize and format the extractor config as a string and log it
            if hasattr(extractor_schema, "model_dump"):
                sanitized_extractor_config = sanitize_for_logging(extractor_schema.model_dump())
            else:
                sanitized_extractor_config = sanitize_for_logging(extractor_schema)
            extractor_config_str = pprint.pformat(sanitized_extractor_config, width=100, sort_dicts=False)
            logger.debug(f"DEBUG - Extractor Config Type: {type(extractor_schema)}")
            logger.debug(f"DEBUG - Extractor Config (sanitized):\n{extractor_config_str}")
            logger.debug("=" * 80 + "\n")

            # Call the backend API function. Print sanitized configs for any debug consumers of stdout.
            pprint.pprint(sanitized_task_config)
            pprint.pprint(sanitized_extractor_config)
            result = api_fn(ledger, task_config, extractor_schema, execution_trace_log)

            # If the result is a tuple, return only the first element
            if isinstance(result, tuple):
                return result[0]
            return result

        return wrapper

    return decorator
