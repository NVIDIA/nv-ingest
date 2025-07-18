# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import importlib.util
import logging
import importlib
import inspect
from typing import Dict
from typing import Optional
from typing import Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from nv_ingest.pipeline.pipeline_schema import PipelinePhase

from .task_base import Task

logger = logging.getLogger(__name__)


class UDFTaskSchema(BaseModel):
    udf_function: Optional[str] = Field(
        None,
        description="UDF function specification. Supports three formats:\n"
        "1. Inline function string: 'def my_func(control_message): ...'\n"
        "2. Import path: 'my_module.my_function'\n"
        "3. File path: '/path/to/file.py:my_function'",
    )
    phase: Union[PipelinePhase, int, str] = Field(
        PipelinePhase.RESPONSE,
        description="Pipeline phase where this UDF should be executed. "
        "Can be specified as phase name (e.g., 'EXTRACTION', 'TRANSFORM') or numeric value. "
        "Defaults to RESPONSE phase.",
    )

    model_config = ConfigDict(extra="forbid")
    model_config["protected_namespaces"] = ()

    @field_validator("udf_function")
    @classmethod
    def validate_udf_function(cls, v):
        """Validate UDF function specification format."""
        if v is None:
            return v

        if not isinstance(v, str):
            raise ValueError("udf_function must be a string")

        if not v.strip():
            raise ValueError("udf_function cannot be empty")

        return v

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v):
        """Validate and convert phase to PipelinePhase enum."""
        if isinstance(v, PipelinePhase):
            return v

        if isinstance(v, int):
            try:
                return PipelinePhase(v)
            except ValueError:
                valid_values = [phase.value for phase in PipelinePhase]
                raise ValueError(f"Invalid phase number {v}. Valid values are: {valid_values}")

        if isinstance(v, str):
            # Convert string to uppercase and try to match enum name
            phase_name = v.upper().strip()

            # Handle common aliases and variations
            phase_aliases = {
                "EXTRACT": "EXTRACTION",
                "PREPROCESS": "PRE_PROCESSING",
                "PRE_PROCESS": "PRE_PROCESSING",
                "PREPROCESSING": "PRE_PROCESSING",
                "POSTPROCESS": "POST_PROCESSING",
                "POST_PROCESS": "POST_PROCESSING",
                "POSTPROCESSING": "POST_PROCESSING",
                "MUTATE": "MUTATION",
            }

            # Apply alias if exists
            if phase_name in phase_aliases:
                phase_name = phase_aliases[phase_name]

            try:
                return PipelinePhase[phase_name]
            except KeyError:
                valid_names = [phase.name for phase in PipelinePhase]
                valid_aliases = list(phase_aliases.keys())
                raise ValueError(
                    f"Invalid phase name '{v}'. Valid phase names are: {valid_names}. "
                    f"Also supported aliases: {valid_aliases}"
                )

        raise ValueError(f"Phase must be a PipelinePhase enum, integer, or string, got {type(v)}")


def _load_function_from_import_path(import_path: str):
    """Load a function from an import path like 'module.submodule.function'."""
    try:
        parts = import_path.split(".")
        module_path = ".".join(parts[:-1])
        function_name = parts[-1]

        module = importlib.import_module(module_path)
        func = getattr(module, function_name)

        if not callable(func):
            raise ValueError(f"'{function_name}' is not callable in module '{module_path}'")

        return func
    except ImportError as e:
        raise ValueError(f"Failed to import module from '{import_path}': {e}")
    except AttributeError as e:
        raise ValueError(f"Function '{function_name}' not found in module '{module_path}': {e}")


def _load_function_from_file_path(file_path: str, function_name: str):
    """Load a function from a file path."""
    try:

        # Create a module spec from the file
        spec = importlib.util.spec_from_file_location("udf_module", file_path)
        if spec is None:
            raise ValueError(f"Could not create module spec from file: {file_path}")

        module = importlib.util.module_from_spec(spec)

        # Execute the module to load its contents
        spec.loader.exec_module(module)

        # Get the function
        func = getattr(module, function_name)

        if not callable(func):
            raise ValueError(f"'{function_name}' is not callable in file '{file_path}'")

        return func
    except Exception as e:
        raise ValueError(f"Failed to load function '{function_name}' from file '{file_path}': {e}")


def _resolve_udf_function(udf_function_spec: str) -> str:
    """
    Resolve UDF function specification to function string.

    Supports three formats:
    1. Inline function string: 'def my_func(control_message): ...'
    2. Import path: 'my_module.my_function'
    3. File path: '/path/to/file.py:my_function'

    Returns the function as a string for execution.
    """
    if udf_function_spec.strip().startswith("def "):
        # Already an inline function string
        return udf_function_spec

    elif ".py:" in udf_function_spec:
        # File path format: /path/to/file.py:function_name
        file_path, function_name = udf_function_spec.split(":", 1)
        func = _load_function_from_file_path(file_path, function_name)

        # Get the source code of the function
        try:
            source = inspect.getsource(func)
            return source
        except OSError as e:
            raise ValueError(f"Could not get source code for function '{function_name}': {e}")

    elif "." in udf_function_spec:
        # Import path format: module.submodule.function
        func = _load_function_from_import_path(udf_function_spec)

        # Get the source code of the function
        try:
            source = inspect.getsource(func)
            return source
        except OSError as e:
            raise ValueError(f"Could not get source code for function from '{udf_function_spec}': {e}")

    else:
        raise ValueError(f"Invalid UDF function specification: {udf_function_spec}")


class UDFTask(Task):
    """
    User-Defined Function (UDF) task for custom processing logic.

    This task allows users to provide custom Python functions that will be executed
    during the ingestion pipeline. The UDF function must accept a control_message
    parameter and return an IngestControlMessage.

    Supports three UDF function specification formats:
    1. Inline function string: 'def my_func(control_message): ...'
    2. Import path: 'my_module.my_function'
    3. File path: '/path/to/file.py:my_function'
    """

    def __init__(
        self,
        udf_function: str = None,
        phase: Union[PipelinePhase, int, str] = PipelinePhase.RESPONSE,
    ) -> None:
        super().__init__()
        self._udf_function = udf_function

        # Use the schema validation to convert phase string to enum if needed
        validated_data = UDFTaskSchema(udf_function=udf_function, phase=phase)
        self._phase = validated_data.phase
        self._resolved_udf_function = None

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "User-Defined Function (UDF) Task:\n"

        if self._udf_function:
            # Show first 100 characters of the function for brevity
            function_preview = self._udf_function[:100]
            if len(self._udf_function) > 100:
                function_preview += "..."
            info += f"  udf_function: {function_preview}\n"
        else:
            info += "  udf_function: None\n"

        # Display phase information
        if isinstance(self._phase, PipelinePhase):
            info += f"  phase: {self._phase.name} ({self._phase.value})\n"
        else:
            info += f"  phase: {self._phase}\n"

        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        task_properties = {}

        if self._udf_function:
            # Resolve the UDF function specification to function string
            resolved_function = self._resolve_udf_function()
            task_properties["udf_function"] = resolved_function

        # Convert phase to integer value for serialization
        if isinstance(self._phase, PipelinePhase):
            task_properties["phase"] = self._phase.value
        else:
            task_properties["phase"] = self._phase

        return {
            "type": "udf",
            "task_properties": task_properties,
        }

    def _resolve_udf_function(self):
        """Resolve UDF function specification to function string."""
        if self._resolved_udf_function is None and self._udf_function:
            self._resolved_udf_function = _resolve_udf_function(self._udf_function)
        return self._resolved_udf_function
