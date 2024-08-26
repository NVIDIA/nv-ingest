# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import mrc
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module

import cudf

from nv_ingest.schemas import MetadataInjectorSchema
from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.util.converters.type_mappings import doc_type_to_content_type
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "metadata_injection"
MODULE_NAMESPACE = "nv_ingest"

MetadataInjectorLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def on_data(message: ControlMessage):
    with message.payload().mutable_dataframe() as mdf:
        df = mdf.to_pandas()

    update_required = False
    rows = []
    for _, row in df.iterrows():
        content_type = doc_type_to_content_type(DocumentTypeEnum(row["document_type"]))
        if "metadata" not in row or "content" not in row["metadata"]:
            update_required = True
            row["metadata"] = {
                "content": row["content"],
                "content_metadata": {
                    "type": content_type.name.lower(),
                },
                "error_metadata": None,
                "image_metadata": (
                    None if content_type != ContentTypeEnum.IMAGE else {"image_type": row["document_type"]}
                ),
                "source_metadata": {
                    "source_id": row["source_id"],
                    "source_name": row["source_name"],
                    "source_type": row["document_type"],
                },
                "text_metadata": (None if (content_type != ContentTypeEnum.TEXT) else {"text_type": "document"}),
            }

        rows.append(row)

    if update_required:
        docs = pd.DataFrame(rows)
        gdf = cudf.from_pandas(docs)
        message_meta = MessageMeta(df=gdf)
        message.payload(message_meta)

    return message


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _metadata_injection(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, MetadataInjectorSchema)

    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def _on_data(message: ControlMessage):
        return on_data(message)

    node = builder.make_node("metadata_injector", _on_data)

    builder.register_module_input("input", node)
    builder.register_module_output("output", node)
