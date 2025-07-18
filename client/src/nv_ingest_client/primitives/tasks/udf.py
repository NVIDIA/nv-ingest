# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict
from typing import Optional
from typing import Union

from pydantic import BaseModel, ConfigDict, Field
from nv_ingest.pipeline.pipeline_schema import PipelinePhase

from .task_base import Task

logger = logging.getLogger(__name__)


class UDFTaskSchema(BaseModel):
    udf_function: Optional[str] = Field(
        None,
        description="Python function string to execute as UDF. Must accept control_message parameter "
        "and return IngestControlMessage.",
    )
    phase: Union[PipelinePhase, int] = Field(
        PipelinePhase.RESPONSE,
        description="Pipeline phase where this UDF should be executed. Defaults to RESPONSE phase.",
    )

    model_config = ConfigDict(extra="forbid")
    model_config["protected_namespaces"] = ()


class UDFTask(Task):
    """
    User-Defined Function (UDF) task for custom processing logic.

    This task allows users to provide custom Python functions that will be executed
    during the ingestion pipeline. The UDF function must accept a control_message
    parameter and return an IngestControlMessage.
    """

    def __init__(
        self,
        udf_function: str = None,
        phase: Union[PipelinePhase, int] = PipelinePhase.RESPONSE,
    ) -> None:
        super().__init__()
        self._udf_function = udf_function
        self._phase = phase

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
            task_properties["udf_function"] = self._udf_function

        # Convert phase to integer value for serialization
        if isinstance(self._phase, PipelinePhase):
            task_properties["phase"] = self._phase.value
        else:
            task_properties["phase"] = self._phase

        return {
            "type": "udf",
            "task_properties": task_properties,
        }
