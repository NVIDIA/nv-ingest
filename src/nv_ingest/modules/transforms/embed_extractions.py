# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import asyncio
import logging
import traceback
from typing import Iterable
from typing import List

import mrc
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.control_message_utils import cm_skip_processing_if_failed
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core import operators as ops
from openai import AsyncOpenAI

import cudf

from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import InfoMessageMetadataSchema
from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "embed_extractions"
MODULE_NAMESPACE = "nv_ingest"

EmbedExtractionsLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, EmbedExtractionsSchema)


async def _make_async_request(
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

    Parameters
    ----------
    prompts : ControlMessage
        List of all prompts that will be sent to the NIM embedding service.
    api_key : str
        The valid NGC api key to make requests to the NIM embedding service.
    embedding_nim_endpoint : str
        The url of the hosted embedding NIM.
    embedding_model : str
        Specifies the embedding model used in the embedding NIM.
    encoding_format : str
        The format to return the embeddings in, valid values are "float" or "base64"
    input_type : str
        nvidia/nv-embedqa-e5-v5 operates in `passage` or `query` mode, and thus require the `input_type` parameter.
        `passage` is used when generating embeddings during indexing. `query` is used when generating embeddings during
        querying. It is very important to use the correct `input_type`. Failure to do so will result in large drops in
        retrieval accuracy.
    truncate : str
        Specifies how inputs longer than the maximum token length of the model are handled. Passing `START` discards
        the start of the input. `END` discards the end of the input. In both cases, input is discarded until the
        remaining input is exactly the maximum input token length for the model. If `NONE` is selected, when the input
        exceeds the maximum input token length an error will be returned.
    filter_errors : bool
        A flag used set the filter criteria in an info message, allowing the pipeline drop embeddings with errors in a
        future step.

    Returns
    -------
        response : dict
            A dictionary containing embeddings and list of info messages for errors that occured during the request to
            the NIM.
    """

    response = {}

    try:
        async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=embedding_nim_endpoint,
        )

        resp = await async_client.embeddings.create(
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

        # Populate the response with the error info for logging/inspection
        response["embedding"] = [None] * len(prompts)
        response["info_msg"] = validated_info_msg

        # Raise an exception so that errors do not remain silent
        raise RuntimeError(f"Embedding error occurred. Info message: {validated_info_msg}") from err

    return response


async def _async_request_handler(
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
    A function to gather caculated embedding results from the NIM embedding service.

    Parameters
    ----------
    prompts : ControlMessage
        List of all prompts that will be sent to the NIM embedding service.
    api_key : str
        The valid NGC api key to make requests to the NIM embedding service.
    embedding_nim_endpoint : str
        The url of the hosted embedding NIM.
    embedding_model : str
        Specifies the embedding model used in the embedding NIM.
    encoding_format : str
        The format to return the embeddings in, valid values are "float" or "base64"
    input_type : str
        nvidia/nv-embedqa-e5-v5 operates in `passage` or `query` mode, and thus require the `input_type` parameter.
        `passage` is used when generating embeddings during indexing. `query` is used when generating embeddings during
        querying. It is very important to use the correct `input_type`. Failure to do so will result in large drops in
        retrieval accuracy.
    truncate : str
        Specifies how inputs longer than the maximum token length of the model are handled. Passing `START` discards
        the start of the input. `END` discards the end of the input. In both cases, input is discarded until the
        remaining input is exactly the maximum input token length for the model. If `NONE` is selected, when the input
        exceeds the maximum input token length an error will be returned.
    filter_errors : bool
        A flag used set the filter criteria in an info message, allowing the pipeline drop embeddings with errors in a
        future step.

    Returns
    -------
        res : list
            A list of dictionaries containing embeddings and info messages describing errors that occured during each
            request.
    """

    res = await asyncio.gather(
        *(
            (
                _make_async_request(
                    prompts=prompt_batch,
                    api_key=api_key,
                    embedding_nim_endpoint=embedding_nim_endpoint,
                    embedding_model=embedding_model,
                    encoding_format=encoding_format,
                    input_type=input_type,
                    truncate=truncate,
                    filter_errors=filter_errors,
                )
            )
            for prompt_batch in prompts
        )
    )

    return res


def _async_runner(
    prompts: List[str],
    api_key: str,
    embedding_nim_endpoint: str,
    embedding_model: str,
    encoding_format: str,
    input_type: str,
    truncate: str,
    event_loop: asyncio.SelectorEventLoop,
    filter_errors: bool,
) -> dict:
    """
    A function asynchronously launch all NIM embedding requests in the supplied asyncio event loop.

    Parameters
    ----------
    prompts : ControlMessage
        List of all prompts that will be sent to the NIM embedding service.
    api_key : str
        The valid NGC api key to make requests to the NIM embedding service.
    embedding_nim_endpoint : str
        The url of the hosted embedding NIM.
    embedding_model : str
        Specifies the embedding model used in the embedding NIM.
    encoding_format : str
        The format to return the embeddings in, valid values are "float" or "base64"
    input_type : str
        nvidia/nv-embedqa-e5-v5 operates in `passage` or `query` mode, and thus require the `input_type` parameter.
        `passage` is used when generating embeddings during indexing. `query` is used when generating embeddings during
        querying. It is very important to use the correct `input_type`. Failure to do so will result in large drops in
        retrieval accuracy.
    truncate : str
        Specifies how inputs longer than the maximum token length of the model are handled. Passing `START` discards
        the start of the input. `END` discards the end of the input. In both cases, input is discarded until the
        remaining input is exactly the maximum input token length for the model. If `NONE` is selected, when the input
        exceeds the maximum input token length an error will be returned.
    event_loop : asyncio.SelectorEventLoop
        The asyncio event loop used to manage asynchronous requests to embedding service.
    filter_errors : bool
        A flag used set the filter criteria in an info message, allowing the pipeline drop embeddings with errors in a
        future step.

    Returns
    -------
        flat_results : dict
            A dictionary containing a list of embeddings and list of info messages for errors that occured.
    """

    results = event_loop.run_until_complete(
        _async_request_handler(
            prompts,
            api_key,
            embedding_nim_endpoint,
            embedding_model,
            encoding_format,
            input_type,
            truncate,
            filter_errors,
        )
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
    A pandas UDF that updates a row of extractions with an embedding, info message for failed embeddings,
    document type (if contains an info message), and a contains embedding flag to simplify internal pipeline filtering.
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
    A generator to yield batches of size `batch_size` from an interable.

    Parameters
    ----------
    iterable : Iterable
        The iterable object to source data for each batch.
    batch_size : int
        Defines the size of each batch.

    Yields
    ------
    Iterable
        Yields an batch of data.

    Notes
    -----
    The length of the last batch may be less than `batch_size`.
    """

    iter_len = len(iterable)
    for idx in range(0, iter_len, batch_size):
        yield iterable[idx : min(idx + batch_size, iter_len)]  # noqa: E203


def _generate_batches(prompts: List[str], batch_size: int = 100):
    """
    A function to create a list of batches of size `batch_size` from a list of prompts.

    Parameters
    ----------
    prompts : List[str]
        A list of prompts that will be the source of data for each batch.
    batch_size : int
        Defines the size of each batch.

    Yields
    ------
    List
        Returns a list of batches of prompts.

    Notes
    -----
    The length of the last batch may be less than `batch_size`.
    """

    return [x for x in _batch_generator(prompts, batch_size)]


def _generate_embeddings(
    ctrl_msg: ControlMessage,
    event_loop: asyncio.SelectorEventLoop,
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

    This function dynamically selects the appropriate metadata field based on content type and
    calculates embeddings using the NIM embedding service. AUDIO and VIDEO types are stubbed and skipped.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The incoming control message which contains metadata to filter on and content used to create embeddings.
    content_type : ContentTypeEnum
        The content type will specify the filter criteria. Data that survives the filter is used to create embeddings.
    event_loop : asyncio.SelectorEventLoop
        The asyncio event loop used to manage asynchronous requests to embedding service.
    batch_size : int
        All elements to be embedded will be grouped into batches of size `batch_size`.
    api_key : str
        The valid NGC api key to make requests to the NIM embedding service.
    embedding_nim_endpoint : str
        The url of the hosted embedding NIM.
    embedding_model : str
        Specifies the embedding model used in the embedding NIM.
    encoding_format : str
        The format to return the embeddings in, valid values are "float" or "base64"
    input_type : str
        nvidia/nv-embedqa-e5-v5 operates in `passage` or `query` mode, and thus require the `input_type` parameter.
        `passage` is used when generating embeddings during indexing. `query` is used when generating embeddings during
        querying. It is very important to use the correct `input_type`. Failure to do so will result in large
        drops in retrieval accuracy.
    truncate : str
        Specifies how inputs longer than the maximum token length of the model are handled. Passing `START` discards
        the start of the input. `END` discards the end of the input. In both cases, input is discarded until the
        remaining input is exactly the maximum input token length for the model. If `NONE` is selected, when the input
        exceeds the maximum input token length an error will be returned.
    filter_errors : bool
        A flag used set the filter criteria in an info message, allowing the pipeline drop embeddings with errors in a
        future step.

    Returns
    -------
        df_text : pd.DataFrame
            Pandas dataframe including metadata with added embeddings and `_content` field for internal pipeline use.
        content_mask : cudf.Series
            A boolean mask representing rows filtered to calculate embeddings.
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
            # Embedding NIMs will complain if text has only whitespaces.
            content_text_mask = cudf_content_getter(mdf["metadata"]).str.strip() != ""
            content_mask = (content_mask & content_text_mask).fillna(False)
            if not content_mask.any():
                continue

            df_content = mdf.loc[content_mask].to_pandas().reset_index(drop=True)
            filtered_content = df_content["metadata"].apply(content_getter)
            # calculate embeddings
            filtered_content_batches = _generate_batches(filtered_content.tolist(), batch_size)
            content_embeddings = _async_runner(
                filtered_content_batches,
                api_key,
                embedding_nim_endpoint,
                embedding_model,
                encoding_format,
                input_type,
                truncate,
                event_loop,
                filter_errors,
            )
            # update embeddings in metadata
            df_content[["metadata", "document_type", "_contains_embeddings"]] = df_content.apply(
                _add_embeddings, **content_embeddings, axis=1
            )[["metadata", "document_type", "_contains_embeddings"]]
            df_content["_content"] = filtered_content

            embedding_dataframes.append(df_content)
            content_masks.append(content_mask)

    message = _concatenate_extractions(ctrl_msg, embedding_dataframes, content_masks)

    return message


def _concatenate_extractions(ctrl_msg: ControlMessage, dataframes: List[pd.DataFrame], masks: List[cudf.Series]):
    """
    A function to concatenate extractions enriched with embeddings and remaining extractions into `ControlMessage`.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The incoming control message which will store concatenated extractions.
    dataframes : List[pd.DataFrame]
        A list of dataframes that will be concatenated and stored in the control message payload.
    masks : List[cudf.Series]
        A list of boolean masks that will be used to identify rows without embeddings.

    Returns
    -------
    ControlMessage
        An updated control message with metadata enriched with embeddings.
    """

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        # build unified mask
        unified_mask = cudf.Series(False, index=mdf.index)
        for mask in masks:
            unified_mask = unified_mask | mask

        df_no_text = mdf.loc[~unified_mask].to_pandas()
        df_no_text["_contains_embeddings"] = False

    dataframes.append(df_no_text)
    df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)

    gdf = cudf.from_pandas(df)
    meta = MessageMeta(df=gdf)
    ctrl_msg.payload(meta)

    return ctrl_msg


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _embed_extractions(builder: mrc.Builder):
    """
    A pipeline module that receives incoming messages in ControlMessage format
    and calculates text embeddings for all supported content types.

    Parameters
    ----------
    builder : mrc.Builder
        The Morpheus builder instance to attach this module to.

    """

    validated_config = fetch_and_validate_module_config(builder, EmbedExtractionsSchema)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(validated_config.httpx_log_level.value)
    event_loop = asyncio.new_event_loop()

    @filter_by_task(["embed"])
    @traceable(MODULE_NAME)
    @cm_skip_processing_if_failed
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def embed_extractions_fn(message: ControlMessage):
        try:
            task_props = message.remove_task("embed")
            model_dump = task_props.model_dump()
            filter_errors = model_dump.get("filter_errors", False)

            return _generate_embeddings(
                message,
                event_loop,
                validated_config.batch_size,
                validated_config.api_key,
                validated_config.embedding_nim_endpoint,
                validated_config.embedding_model,
                validated_config.encoding_format,
                validated_config.input_type,
                validated_config.truncate,
                filter_errors,
            )

        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Failed to generate embeddings: {e}")

    embedding_node = builder.make_node("embed_extractions", ops.map(embed_extractions_fn))

    # Register the input and output of the module
    builder.register_module_input("input", embedding_node)
    builder.register_module_output("output", embedding_node)
