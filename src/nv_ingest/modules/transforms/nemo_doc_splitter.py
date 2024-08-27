# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import copy
import logging
import traceback
import uuid
from typing import Any
from typing import List
from typing import Literal

import mrc
import pandas as pd
from more_itertools import windowed
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops

import cudf

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.nemo_doc_splitter_schema import DocumentSplitterSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)


def _build_split_documents(row, text_splits: List[str], sentence_window_size: int) -> List[dict[str, Any]]:
    """Build documents from text splits with window text."""
    documents: List[dict] = []

    window_size = sentence_window_size
    for i, text in enumerate(text_splits):
        if text is None or not text.strip():
            continue

        metadata = row.metadata if hasattr(row, "metadata") and isinstance(row.metadata, dict) else {}
        metadata = copy.deepcopy(metadata)
        if window_size > 0:
            window_text = "".join(
                text_splits[max(0, i - window_size) : min(i + 1 + window_size, len(text_splits))]  # noqa: E203
            )

            metadata["window"] = window_text
            metadata["original_text"] = text

        metadata["content"] = text

        documents.append({"document_type": ContentTypeEnum.TEXT.value, "metadata": metadata, "uuid": str(uuid.uuid4())})

    return documents


def _split_into_units(text: str, split_by: Literal["word", "sentence", "passage"]) -> List[str]:
    if split_by == "passage":
        split_at = "\n\n"
    elif split_by == "sentence":
        split_at = "."  # why not ?,!, etc..?
    elif split_by == "word":
        split_at = " "
    else:
        raise NotImplementedError("DocumentSplitter only supports 'passage', 'sentence'" " or 'word' split_by options.")
    units = text.split(split_at)
    # Add the delimiter back to all units except the last one
    for i in range(len(units) - 1):
        units[i] += split_at

    return units


def _concatenate_units(units: List[str], split_length: int, split_overlap: int, max_character_length: int) -> List[str]:
    text_splits = []
    segments = windowed(units, n=split_length, step=split_length - split_overlap)
    for seg in segments:
        current_units = [unit for unit in seg if unit is not None]
        txt = "".join(current_units)
        if max_character_length and len(txt) > max_character_length:
            text_splits.extend(_split_long_text(txt, max_character_length))
        elif len(txt) > 0:
            text_splits.append(txt)

    return text_splits


def _split_long_text(text: str, max_character_length: int) -> List[str]:
    """
    Splits a long text into smaller segments that
    do not exceed max_character_length.
    """
    split_texts = []
    while text:
        # Take the maximum possible substring without exceeding max_character_length
        segment = text[:max_character_length]
        split_texts.append(segment)
        text = text[max_character_length:]  # noqa: E203

    return split_texts


def _process_content(row, validated_config):
    content = row["metadata"]["content"]

    if content is None:
        raise ValueError(
            "DocumentSplitter only works with text documents but one or more 'content' " "values are None."
        )

    units = _split_into_units(content, validated_config.split_by)
    text_splits = _concatenate_units(
        units,
        validated_config.split_length,
        validated_config.split_overlap,
        max_character_length=validated_config.max_character_length,
    )
    split_docs = _build_split_documents(row, text_splits, sentence_window_size=validated_config.sentence_window_size)

    return split_docs


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
            split_by = task_props.get("split_by", validated_config.split_by)
            split_length = task_props.get("split_length", validated_config.split_length)
            split_overlap = task_props.get("split_overlap", validated_config.split_overlap)
            max_character_length = task_props.get("max_character_length", validated_config.max_character_length)
            sentence_window_size = task_props.get("sentence_window_size", validated_config.sentence_window_size)

            logger.info(
                f"Splitting documents with split_by: {split_by}, split_length: {split_length}, "
                f"split_overlap: {split_overlap}, max_character_length: {max_character_length}, "
                f"sentence_window_size: {sentence_window_size}"
            )

            split_docs = []
            for _, row in df_filtered.iterrows():
                content = row["metadata"]["content"]

                if content is None:
                    raise ValueError(
                        "DocumentSplitter only works with text documents but one or more " "'content' values are None."
                    )

                units = _split_into_units(content, split_by)
                text_splits = _concatenate_units(
                    units,
                    split_length,
                    split_overlap,
                    max_character_length=max_character_length,
                )
                split_docs.extend(_build_split_documents(row, text_splits, sentence_window_size=sentence_window_size))

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
