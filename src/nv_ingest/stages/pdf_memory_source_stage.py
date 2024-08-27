# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import logging
import time
import typing
import uuid

# pylint: disable=morpheus-incorrect-lib-from-import
import mrc
from morpheus.cli import register_stage
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.preallocator_mixin import PreallocatorMixin
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stage_schema import StageSchema

import cudf

from nv_ingest.schemas import validate_ingest_job

logger = logging.getLogger(__name__)


@register_stage(
    "pdf-memory-source",
    modes=[PipelineModes.FIL, PipelineModes.NLP, PipelineModes.OTHER],
)
class PdfMemoryFileSource(PreallocatorMixin, SingleOutputSource):
    """
    Load messages from a file.

    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    source_config : str
        Path to json file containing paths to pdf documents.
    repeat : int, default = 1, min = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    """

    def __init__(self, c: Config, source_config, repeat=4):
        super().__init__(c)

        self._source_config = source_config
        self._repeat = repeat

        self._load_pdfs()

    def _load_pdfs(self):
        pdf_files = [
            "/workspace/data/LEGO_EBook_US_Fall_2023_Small.pdf"
        ]  # self._source_config["sampled_files"]  # [5:6]#[:1]
        self._jobs = []

        for pdf_path in pdf_files[:]:
            job_desc = {}

            job_id = str(uuid.uuid4())

            with open(pdf_path, "rb") as file:
                encoding = "utf-8"
                content = base64.b64encode(file.read()).decode(encoding)
                job_data = {
                    "content": [content],
                    "source_name": [job_id],
                    "source_id": [job_id],
                    "document_type": ["pdf"],
                }

                job_id = str(uuid.uuid4())
                job_desc["job_payload"] = job_data
                job_desc["job_id"] = job_id
                tasks = [
                    {
                        "type": "split",
                        "task_properties": {
                            "split_by": "word",
                            "split_length": 250,
                            "split_overlap": 30,
                            "max_character_length": 1900,
                            "sentence_window_size": 0,
                        },
                    },
                    {
                        "type": "extract",
                        "task_properties": {
                            "method": "pdfium",
                            "document_type": "pdf",
                            "params": {
                                "extract_text": True,
                                "extract_images": True,
                                "extract_tables": False,
                                "text_depth": "document",
                            },
                        },
                    },
                    {
                        "type": "dedup",
                        "task_properties": {
                            "content_type": "image",
                            "params": {
                                "filter": True,
                            },
                        },
                    },
                    {
                        "type": "filter",
                        "task_properties": {
                            "content_type": "image",
                            "params": {
                                "min_size": 256,
                                "max_aspect_ratio": 5.0,
                                "min_aspect_ratio": 0.2,
                                "filter": False,
                            },
                        },
                    },
                    {
                        "type": "caption",
                        "task_properties": {"n_neighbors": 5},
                    },
                ]

                job_desc["tasks"] = tasks
                job_desc["tracing_options"] = {"trace": True, "ts_send": time.time_ns()}

                validate_ingest_job(job_desc)

                self._jobs.append(job_desc)

    @property
    def name(self) -> str:
        """Return the name of the stage"""
        return "from-pdf-file"

    def supports_cpp_node(self) -> bool:
        """Indicates whether this stage supports a C++ node"""
        return False

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def _build_source(self, builder: mrc.Builder) -> mrc.SegmentObject:
        node = builder.make_source(self.unique_name, self._generate_frames())

        return node

    def _generate_frames(self) -> typing.Iterable[ControlMessage]:
        for i in range(self._repeat):
            for job in self._jobs:
                job = job.copy()

                job_id = job.pop("job_id")
                job_payload = job.pop("job_payload", {})
                job_tasks = job.pop("tasks", [])

                tracing_options = job.pop("tracing_options", {})
                tracing_options.get("trace", False)
                tracing_options.get("ts_send", None)

                response_channel = f"response_{job_id}"

                df = cudf.DataFrame(job_payload)
                message_meta = MessageMeta(df=df)

                control_message = ControlMessage()
                control_message.payload(message_meta)
                control_message.set_metadata("response_channel", response_channel)
                control_message.set_metadata("job_id", job_id)

                for task in job_tasks:
                    control_message.add_task(task["type"], task["task_properties"])

                yield control_message
