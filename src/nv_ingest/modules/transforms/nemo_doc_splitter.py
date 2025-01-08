# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import copy
import logging
import traceback
import uuid
from typing import Any
from typing import List
from typing import Literal

import mrc
import pandas as pd
from transformers import AutoTokenizer
from more_itertools import windowed
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from pydantic import BaseModel

import cudf

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.nemo_doc_splitter_schema import DocumentSplitterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)


def _build_split_documents(row, chunks: List[str]) -> List[dict[str, Any]]:
    """Build documents from text chunks"""
    documents: List[dict] = []

    for i, text in enumerate(chunks):
        if text is None or not text.strip():
            continue

        metadata = row.metadata if hasattr(row, "metadata") and isinstance(row.metadata, dict) else {}
        metadata = copy.deepcopy(metadata)

        metadata["content"] = text

        documents.append({"document_type": ContentTypeEnum.TEXT.value, "metadata": metadata, "uuid": str(uuid.uuid4())})

    return documents


def _split_into_chunks(text, tokenizer, chunk_size=300):
    # Tokenize the text into token IDs
    encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)

    # Get the token IDs and offsets for splitting
    tokens = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    # Split the tokens into chunks of the desired size
    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    # Convert token chunks back to text while preserving original spacing and case
    text_chunks = []
    for chunk in chunks:
        # Find the start and end offsets for the current chunk
        chunk_offsets = offsets[: len(chunk)]
        start_offset = chunk_offsets[0][0]
        end_offset = chunk_offsets[-1][1]

        # Extract the original text for this chunk based on offsets
        text_chunk = text[start_offset:end_offset]
        text_chunks.append(text_chunk)

        # Remove processed offsets for the next iteration
        offsets = offsets[len(chunk) :]

    return text_chunks


MODULE_NAME = "nemo_document_splitter"
MODULE_NAMESPACE = "nv_ingest"

NemoDocSplitterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, DocumentSplitterSchema)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _nemo_document_splitter(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """

    validated_config = fetch_and_validate_module_config(builder, DocumentSplitterSchema)

    @filter_by_task(["split"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def split_and_forward(message: ControlMessage):
        try:
            # Assume that df is going to have a 'content' column
            task_props = message.remove_task("split")

            if isinstance(task_props, BaseModel):
                task_props = task_props.model_dump()

            # Validate that all 'content' values are not None
            with message.payload().mutable_dataframe() as mdf:
                df = mdf.to_pandas()

            # Filter to text only
            bool_index = df["document_type"] == ContentTypeEnum.TEXT
            df_filtered = df.loc[bool_index]

            if df_filtered.empty:
                gdf = cudf.from_pandas(df)
                message_meta = MessageMeta(df=gdf)
                message.payload(message_meta)

                return message

            # Override parameters if set
            tokenizer = task_props.get("tokenizer", validated_config.tokenizer)
            chunk_size = task_props.get("chunk_size", validated_config.chunk_size)
            chunk_overlap = task_props.get("chunk_overlap", validated_config.chunk_overlap)

            logger.info(
                f"Splitting text with tokenizer: {tokenizer}, chunk_size: {chunk_size} tokens, chunk_overlap: {chunk_overlap}"
            )

            tokenizer_model = AutoTokenizer.from_pretrained(tokenizer, token="")

            split_docs = []
            for _, row in df_filtered.iterrows():
                content = row["metadata"]["content"]

                if content is None:
                    raise ValueError(
                        "DocumentSplitter only works with text documents but one or more " "'content' values are None."
                    )

                chunks = _split_into_chunks(content, tokenizer_model, chunk_size)
                split_docs.extend(_build_split_documents(row, chunks))

            split_docs_df = pd.DataFrame(split_docs)

            # Return both processed text and other document types
            split_docs_df = pd.concat([split_docs_df, df[~bool_index]], axis=0).reset_index(drop=True)
            # Update control message with new payload
            split_docs_gdf = cudf.from_pandas(split_docs_df)

            message_meta = MessageMeta(df=split_docs_gdf)
            message.payload(message_meta)

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to split documents: {e}")

    split_node = builder.make_node("split_and_forward", ops.map(split_and_forward))

    # Register the input and output of the module
    builder.register_module_input("input", split_node)
    builder.register_module_output("output", split_node)
