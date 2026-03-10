# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import importlib.util
import logging
import importlib
import inspect
import ast
import re
from typing import Dict, Optional, Union

from nv_ingest_api.internal.enums.common import PipelinePhase
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskUDFSchema
from nv_ingest_client.primitives.tasks.task_base import Task

logger = logging.getLogger(__name__)


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


def _extract_function_with_context(file_path: str, function_name: str) -> str:
    """
    Extract a function from a file while preserving the full module context.

    This includes all imports, module-level variables, and other functions
    that the target function might depend on.

    Parameters
    ----------
    file_path : str
        Path to the Python file containing the function
    function_name : str
        Name of the function to extract

    Returns
    -------
    str
        Complete module source code with the target function
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            module_source = f.read()

        # Parse the module to verify the function exists
        try:
            tree = ast.parse(module_source)
            function_found = False

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    function_found = True
                    break

            if not function_found:
                raise ValueError(f"Function '{function_name}' not found in file '{file_path}'")

        except SyntaxError as e:
            raise ValueError(f"Syntax error in file '{file_path}': {e}")

        return module_source

    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to read file '{file_path}': {e}")


def _resolve_udf_function(udf_function_spec: str) -> str:
    """
    Resolve UDF function specification to function string.

    Supports four formats:
    1. Inline function string: 'def my_func(control_message): ...'
    2. Module path with colon: 'my_module.my_submodule:my_function' (preserves imports)
    3. File path: '/path/to/file.py:my_function'
    4. Legacy import path: 'my_module.my_function' (function name only, no imports)
    """
    # Default to treating as inline unless it clearly matches a
    # module/file specification. This avoids misclassifying inline code that
    # contains colons, imports, or annotations before the def line.

    spec = udf_function_spec.strip()

    # 1) File path with function: /path/to/file.py:function_name
    if ".py:" in spec:
        file_path, function_name = spec.split(":", 1)
        return _extract_function_with_context(file_path, function_name)

    # 2) File path without function name is an explicit error
    if spec.endswith(".py"):
        raise ValueError(
            f"File path '{udf_function_spec}' is missing function name. Use format 'file.py:function_name'."
        )

    # 3) Module path with colon: my.module:function
    # Be strict: only letters, numbers, underscore, and dots on the left; valid identifier on the right;
    # no whitespace/newlines.
    module_colon_pattern = re.compile(r"^[A-Za-z_][\w\.]*:[A-Za-z_][\w]*$")
    if module_colon_pattern.match(spec):
        module_path, function_name = spec.split(":", 1)
        try:
            module = importlib.import_module(module_path)
            module_file = inspect.getfile(module)
            return _extract_function_with_context(module_file, function_name)
        except ImportError as e:
            raise ValueError(f"Failed to import module '{module_path}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to resolve module path '{module_path}': {e}")

    # 4) Legacy import path: my.module.function (no colon)
    legacy_import_pattern = re.compile(r"^[A-Za-z_][\w\.]*\.[A-Za-z_][\w]*$")
    if legacy_import_pattern.match(spec):
        func = _load_function_from_import_path(spec)
        try:
            source = inspect.getsource(func)
            return source
        except (OSError, TypeError) as e:
            raise ValueError(f"Could not get source code for function from '{udf_function_spec}': {e}")

    # 5) Default: treat as inline UDF source (entire string)
    return udf_function_spec


class UDFTask(Task):
    """
    User-Defined Function (UDF) task for custom processing logic.

    This task allows users to provide custom Python functions that will be executed
    during the ingestion pipeline. The UDF function must accept a control_message
    parameter and return an IngestControlMessage.

    Supports four UDF function specification formats:
    1. Inline function string: 'def my_func(control_message): ...'
    2. Module path with colon: 'my_module.my_submodule:my_function' (preserves imports)
    3. File path: '/path/to/file.py:my_function'
    4. Legacy import path: 'my_module.my_function' (function name only, no imports)
    """

    def __init__(
        self,
        udf_function: Optional[str] = None,
        udf_function_name: Optional[str] = None,
        phase: Union[PipelinePhase, int, str, None] = PipelinePhase.RESPONSE,
        target_stage: Optional[str] = None,
        run_before: bool = False,
        run_after: bool = False,
    ) -> None:
        super().__init__()
        self._udf_function = udf_function
        self._udf_function_name = udf_function_name
        self._target_stage = target_stage
        self._run_before = run_before
        self._run_after = run_after

        # Convert phase to the appropriate format for API schema
        # If target_stage is provided and phase is None, don't convert phase
        if target_stage is not None and phase is None:
            converted_phase = None
            self._phase = None  # Set to None when using target_stage
        else:
            converted_phase = self._convert_phase(phase)
            self._phase = PipelinePhase(converted_phase)  # Convert back to enum for internal use

        # Use the API schema for validation
        _ = IngestTaskUDFSchema(
            udf_function=udf_function or "",
            udf_function_name=udf_function_name or "",
            phase=converted_phase,
            target_stage=target_stage,
            run_before=run_before,
            run_after=run_after,
        )
        self._resolved_udf_function = None

    def _convert_phase(self, phase: Union[PipelinePhase, int, str]) -> int:
        """Convert phase to integer for API schema validation."""
        if isinstance(phase, PipelinePhase):
            return phase.value

        if isinstance(phase, int):
            try:
                PipelinePhase(phase)  # Validate it's a valid phase number
                return phase
            except ValueError:
                valid_values = [p.value for p in PipelinePhase]
                raise ValueError(f"Invalid phase number {phase}. Valid values are: {valid_values}")

        if isinstance(phase, str):
            # Convert string to uppercase and try to match enum name
            phase_name = phase.upper().strip()

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
                return PipelinePhase[phase_name].value
            except KeyError:
                valid_names = [p.name for p in PipelinePhase]
                valid_aliases = list(phase_aliases.keys())
                raise ValueError(
                    f"Invalid phase name '{phase}'. Valid phase names are: {valid_names}. "
                    f"Also supported aliases: {valid_aliases}"
                )

        raise ValueError(f"Phase must be a PipelinePhase enum, integer, or string, got {type(phase)}")

    @property
    def udf_function(self) -> Optional[str]:
        """
        Returns the UDF function string or specification.
        """
        return self._udf_function

    @property
    def udf_function_name(self) -> Optional[str]:
        """
        Returns the UDF function name.
        """
        return self._udf_function_name

    @property
    def phase(self) -> PipelinePhase:
        """
        Returns the pipeline phase for this UDF task.
        """
        return self._phase

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

        if self._udf_function_name:
            task_properties["udf_function_name"] = self._udf_function_name

        # Convert phase to integer value for serialization
        if isinstance(self._phase, PipelinePhase):
            task_properties["phase"] = self._phase.value
        else:
            task_properties["phase"] = self._phase

        # Add new stage targeting parameters
        if self._target_stage:
            task_properties["target_stage"] = self._target_stage

        task_properties["run_before"] = self._run_before
        task_properties["run_after"] = self._run_after

        return {
            "type": "udf",
            "task_properties": task_properties,
        }

    def _resolve_udf_function(self):
        """Resolve UDF function specification to function string."""
        if self._resolved_udf_function is None and self._udf_function:
            self._resolved_udf_function = _resolve_udf_function(self._udf_function)
        return self._resolved_udf_function
