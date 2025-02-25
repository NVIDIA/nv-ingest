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

import mrc
import pandas as pd
from transformers import AutoTokenizer
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.text_splitter_schema import TextSplitterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type

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


def _split_into_chunks(text, tokenizer, chunk_size=1024, chunk_overlap=20):
    # Tokenize the text into token IDs
    encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)

    # Get the token IDs and offsets for splitting
    offsets = encoding["offset_mapping"]

    # Split the tokens into chunks of the desired size with the desired overlap
    chunks = [offsets[i : i + chunk_size] for i in range(0, len(offsets), chunk_size - chunk_overlap)]

    # Convert token chunks back to text while preserving original spacing and case
    text_chunks = []
    for chunk in chunks:
        text_chunk = text[chunk[0][0] : chunk[-1][0]]
        text_chunks.append(text_chunk)

    return text_chunks


MODULE_NAME = "text_splitter"
MODULE_NAMESPACE = "nv_ingest"

TextSplitterLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, TextSplitterSchema)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _text_splitter(builder: mrc.Builder):
    """
    A pipeline module that splits documents into smaller parts based on the specified criteria.
    """

    validated_config = fetch_and_validate_module_config(builder, TextSplitterSchema)

    @filter_by_task(["split"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def split_and_forward(message: IngestControlMessage):
        try:
            # Assume that df is going to have a 'content' column
            task_props = remove_task_by_type(message, "split")

            # Validate that all 'content' values are not None
            df = message.payload()

            # Override parameters if set
            tokenizer = task_props.get("tokenizer", validated_config.tokenizer)
            chunk_size = task_props.get("chunk_size", validated_config.chunk_size)
            chunk_overlap = task_props.get("chunk_overlap", validated_config.chunk_overlap)
            params = task_props.get("params", {})

            hf_access_token = params.get("hf_access_token", None)
            split_source_types = params.get("split_source_types", ["text"])

            logger.debug(
                f"Splitting text with tokenizer: {tokenizer}, "
                f"chunk_size: {chunk_size} tokens, "
                f"chunk_overlap: {chunk_overlap}"
            )

            # Filter to document and file type
            bool_index = (df["document_type"] == ContentTypeEnum.TEXT) & (
                pd.json_normalize(df["metadata"])["source_metadata.source_type"].isin(split_source_types)
            )
            df_filtered = df.loc[bool_index]

            if df_filtered.empty:
                return message

            if os.path.exists("/workspace/models/llama-3.2-1b/tokenizer/tokenizer.json") and (
                tokenizer is None or tokenizer == "meta-llama/Llama-3.2-1B"
            ):
                tokenizer = "/workspace/models/llama-3.2-1b/tokenizer/"
            elif os.path.exists("/workspace/models/e5-unsupervised-large/tokenizer/tokenizer.json") and (
                tokenizer is None or tokenizer == "intfloat/e5-large-unsupervised"
            ):
                tokenizer = "/workspace/models/e5-unsupervised-large/tokenizer/"

            tokenizer_model = AutoTokenizer.from_pretrained(tokenizer, token=hf_access_token)

            split_docs = []
            for _, row in df_filtered.iterrows():
                content = row["metadata"]["content"] if row["metadata"]["content"] is not None else ""

                chunks = _split_into_chunks(content, tokenizer_model, chunk_size, chunk_overlap)
                split_docs.extend(_build_split_documents(row, chunks))

            split_docs_df = pd.DataFrame(split_docs)

            # Return both processed text and other document types
            split_docs_df = pd.concat([split_docs_df, df[~bool_index]], axis=0).reset_index(drop=True)

            message.payload(split_docs_df)

            return message
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to split documents: {e}")

    split_node = builder.make_node("split_and_forward", ops.map(split_and_forward))

    # Register the input and output of the module
    builder.register_module_input("input", split_node)
    builder.register_module_output("output", split_node)
