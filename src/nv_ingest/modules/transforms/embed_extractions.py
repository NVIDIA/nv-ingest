# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import traceback
from typing import Iterable, List

import mrc
import pandas as pd
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory, register_module
from mrc.core import operators as ops
from openai import OpenAI

import cudf

from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum, InfoMessageMetadataSchema, StatusEnum, TaskTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.tracing import traceable
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage, remove_task_by_type

logger = logging.getLogger(__name__)

MODULE_NAME = "embed_extractions"
MODULE_NAMESPACE = "nv_ingest"

EmbedExtractionsLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, EmbedExtractionsSchema)


def _make_async_request(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> list:
    """
    A function that interacts directly with the NIM embedding service to caculate embeddings for a batch of prompts.
    """
    response = {}

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=embedding_nim_endpoint,
        )

        resp = client.embeddings.create(
            input=prompts,
            model=embedding_model,
            encoding_format=encoding_format,
            extra_body={"input_type": input_type, "truncate": truncate},
        )

        response["embedding"] = resp.data
        response["info_msg"] = None

    except Exception as err:
        info_msg = {
            "task": TaskTypeEnum.EMBED.value,
            "status": StatusEnum.ERROR.value,
            "message": f"Embedding error: {err}",
            "filter": filter_errors,
        }

        validated_info_msg = validate_schema(info_msg, InfoMessageMetadataSchema).model_dump()

        response["embedding"] = [None] * len(prompts)
        response["info_msg"] = validated_info_msg

        raise RuntimeError(f"Embedding error occurred. Info message: {validated_info_msg}") from err

    return response


def _async_request_handler(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> List[dict]:
    """
    A function to gather calculated embedding results from the NIM embedding service.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _make_async_request,
                prompts=prompt_batch,
                api_key=api_key,
                embedding_nim_endpoint=embedding_nim_endpoint,
                embedding_model=embedding_model,
                encoding_format=encoding_format,
                input_type=input_type,
                truncate=truncate,
                filter_errors=filter_errors,
            )
            for prompt_batch in prompts
        ]
        results = [future.result() for future in futures]

    return results


def _async_runner(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
) -> dict:
    """
    A function that concurrently launches all NIM embedding requests.
    """
    results = _async_request_handler(
        prompts,
        api_key,
        embedding_nim_endpoint,
        embedding_model,
        encoding_format,
        input_type,
        truncate,
        filter_errors,
    )

    flat_results = {"embeddings": [], "info_msgs": []}
    for batch_dict in results:
        info_msg = batch_dict["info_msg"]
        for embedding in batch_dict["embedding"]:
            if not isinstance(embedding, list):
                if embedding is not None:
                    flat_results["embeddings"].append(embedding.embedding)
                else:
                    flat_results["embeddings"].append(embedding)
            else:
                flat_results["embeddings"].append(embedding)
            flat_results["info_msgs"].append(info_msg)

    return flat_results


def _add_embeddings(row, embeddings, info_msgs):
    """
    A pandas UDF that updates a row of extractions with an embedding, an info message for failed embeddings,
    a document type (if contains an info message), and a contains embedding flag to simplify internal pipeline
    filtering.
    """
    row["metadata"]["embedding"] = embeddings[row.name]
    if info_msgs[row.name] is not None:
        row["metadata"]["info_message_metadata"] = info_msgs[row.name]
        row["document_type"] = ContentTypeEnum.INFO_MSG
        row["_contains_embeddings"] = False
    else:
        row["_contains_embeddings"] = True

    return row


def _get_pandas_text_content(row):
    """
    A pandas UDF used to select extracted text content to be used to create embeddings.
    """
    return row["content"]


def _get_pandas_table_content(row):
    """
    A pandas UDF used to select extracted table/chart content to be used to create embeddings.
    """
    return row["table_metadata"]["table_content"]


def _get_pandas_image_content(row):
    """
    A pandas UDF used to select extracted image captions to be used to create embeddings.
    """
    return row["image_metadata"]["caption"]


def _get_cudf_text_content(df: cudf.DataFrame):
    """
    A cuDF UDF used to select extracted text content to be used to create embeddings.
    """
    return df.struct.field("content")


def _get_cudf_table_content(df: cudf.DataFrame):
    """
    A cuDF UDF used to select extracted table/chart content to be used to create embeddings.
    """
    return df.struct.field("table_metadata").struct.field("table_content")


def _get_cudf_image_content(df: cudf.DataFrame):
    """
    A cuDF UDF used to select extracted image captions to be used to create embeddings.
    """
    return df.struct.field("image_metadata").struct.field("caption")


def _batch_generator(iterable: Iterable, batch_size=10):
    """
    A generator to yield batches of size `batch_size` from an iterable.
    """
    iter_len = len(iterable)
    for idx in range(0, iter_len, batch_size):
        yield iterable[idx : min(idx + batch_size, iter_len)]


def _generate_batches(prompts: List[str], batch_size: int = 100):
    """
    A function to create a list of batches of size `batch_size` from a list of prompts.
    """
    return [x for x in _batch_generator(prompts, batch_size)]


def _generate_embeddings(
    ctrl_msg: IngestControlMessage,
    batch_size: int,
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    filter_errors: bool,
):
    """
    A function to generate text embeddings for supported content types (TEXT, STRUCTURED, IMAGE).
    """
    cudf_content_extractor = {
        ContentTypeEnum.TEXT: _get_cudf_text_content,
        ContentTypeEnum.STRUCTURED: _get_cudf_table_content,
        ContentTypeEnum.IMAGE: _get_cudf_image_content,
        ContentTypeEnum.AUDIO: lambda _: None,  # Not supported yet.
        ContentTypeEnum.VIDEO: lambda _: None,  # Not supported yet.
    }
    pandas_content_extractor = {
        ContentTypeEnum.TEXT: _get_pandas_text_content,
        ContentTypeEnum.STRUCTURED: _get_pandas_table_content,
        ContentTypeEnum.IMAGE: _get_pandas_image_content,
        ContentTypeEnum.AUDIO: lambda _: None,  # Not supported yet.
        ContentTypeEnum.VIDEO: lambda _: None,  # Not supported yet.
    }

    logger.debug("Generating text embeddings for supported content types: TEXT, STRUCTURED, IMAGE.")

    embedding_dataframes = []
    content_masks = []

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        if mdf.empty:
            return ctrl_msg

        for content_type, content_getter in pandas_content_extractor.items():
            if not content_getter:
                logger.debug(f"Skipping unsupported content type: {content_type}")
                continue

            content_mask = mdf["document_type"] == content_type.value
            if not content_mask.any():
                continue

            cudf_content_getter = cudf_content_extractor[content_type]
            content_text_mask = cudf_content_getter(mdf["metadata"]).str.strip() != ""
            content_mask = (content_mask & content_text_mask).fillna(False)
            if not content_mask.any():
                continue

            df_content = mdf.loc[content_mask].to_pandas().reset_index(drop=True)
            filtered_content = df_content["metadata"].apply(content_getter)
            # Force using a fixed batch size of 8192, ignoring the provided batch_size parameter.
            filtered_content_batches = _generate_batches(filtered_content.tolist(), batch_size=batch_size)
            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                embedding_nim_endpoint,
                embedding_model,
                encoding_format,
                input_type,
                truncate,
                filter_errors,
            )
            df_content[["metadata", "document_type", "_contains_embeddings"]] = df_content.apply(
                _add_embeddings, **content_embeddings, axis=1
            )[["metadata", "document_type", "_contains_embeddings"]]
            df_content["_content"] = filtered_content

            embedding_dataframes.append(df_content)
            content_masks.append(content_mask)

    message = _concatenate_extractions(ctrl_msg, embedding_dataframes, content_masks)

    return message


def _concatenate_extractions(ctrl_msg: IngestControlMessage, dataframes: List[pd.DataFrame], masks: List[cudf.Series]):
    """
    A function to concatenate extractions enriched with embeddings and remaining extractions into
    `IngestControlMessage`.
    """
    with ctrl_msg.payload().mutable_dataframe() as mdf:
        unified_mask = cudf.Series(False, index=mdf.index)
        for mask in masks:
            unified_mask = unified_mask | mask

        df_no_text = mdf.loc[~unified_mask].to_pandas()
        df_no_text["_contains_embeddings"] = False

    dataframes.append(df_no_text)
    df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
    ctrl_msg.payload(df)

    return ctrl_msg


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _embed_extractions(builder: mrc.Builder):
    """
    A pipeline module that receives incoming messages in IngestControlMessage format
    and calculates text embeddings for all supported content types.
    """
    validated_config = fetch_and_validate_module_config(builder, EmbedExtractionsSchema)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(validated_config.httpx_log_level.value)

    @filter_by_task(["embed"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def embed_extractions_fn(message: IngestControlMessage):
        try:
            task_props = remove_task_by_type(message, "embed")
            model_dump = task_props.model_dump()

            endpoint_url = task_props.get("endpoint_url") or validated_config.embedding_nim_endpoint
            model_name = task_props.get("model_name") or validated_config.embedding_model
            api_key = task_props.get("api_key") or validated_config.api_key
            filter_errors = model_dump.get("filter_errors", False)

            return _generate_embeddings(
                message,
                validated_config.batch_size,  # This parameter is now ignored in _generate_embeddings.
                api_key,
                endpoint_url,
                model_name,
                validated_config.encoding_format,
                validated_config.input_type,
                validated_config.truncate,
                filter_errors,
            )

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to generate embeddings: {e}")

    embedding_node = builder.make_node("embed_extractions", ops.map(embed_extractions_fn))

    builder.register_module_input("input", embedding_node)
    builder.register_module_output("output", embedding_node)
