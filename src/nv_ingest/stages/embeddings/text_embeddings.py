import logging
import functools
from typing import Any, Dict, Tuple, Optional, Iterable, List

import cudf
import pandas as pd
from openai import OpenAI

from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum, TaskTypeEnum, StatusEnum, InfoMessageMetadataSchema
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


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


def _get_pandas_audio_content(row):
    """
    A pandas UDF used to select extracted audio transcription to be used to create embeddings.
    """
    return row["audio_metadata"]["audio_transcript"]


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


def _concatenate_extractions_pandas(
    base_df: pd.DataFrame, dataframes: List[pd.DataFrame], masks: List[pd.Series]
) -> pd.DataFrame:
    """
    Concatenates extractions enriched with embeddings with remaining rows from the base DataFrame,
    using only pandas operations.

    Parameters
    ----------
    base_df : pd.DataFrame
        The original DataFrame.
    dataframes : List[pd.DataFrame]
        A list of DataFrames with embeddings applied.
    masks : List[pd.Series]
        A list of pandas Series (boolean masks) indicating rows that were processed.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame.
    """
    unified_mask = pd.Series(False, index=base_df.index)
    for mask in masks:
        unified_mask = unified_mask | mask

    df_no_text = base_df.loc[~unified_mask].copy()
    df_no_text["_contains_embeddings"] = False

    dataframes.append(df_no_text)
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True).reset_index(drop=True)
    return combined_df


def _generate_text_embeddings_df(
    df: pd.DataFrame, task_props: Dict[str, Any], validated_config: Any, trace_info: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate text embeddings for supported content types (TEXT, STRUCTURED, IMAGE)
    from a pandas DataFrame. This function uses only pandas for processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the content from which embeddings are to be generated.
    task_props : Dict[str, Any]
        Dictionary containing task properties (e.g. a flag for filtering errors).
    validated_config : Any
        The validated configuration object for text embedding extraction (EmbedExtractionsSchema).
    trace_info : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the updated DataFrame with embeddings and a dictionary with trace info.
    """
    if trace_info is None:
        trace_info = {}
        logger.debug("No trace_info provided. Initialized empty trace_info dictionary.")

    if df.empty:
        return df, {"trace_info": trace_info}

    embedding_dataframes = []
    content_masks = []  # List of pandas boolean Series

    # Define pandas extractors for supported content types.
    pandas_content_extractor = {
        ContentTypeEnum.TEXT: _get_pandas_text_content,
        ContentTypeEnum.STRUCTURED: _get_pandas_table_content,
        ContentTypeEnum.IMAGE: _get_pandas_image_content,
        ContentTypeEnum.AUDIO: _get_pandas_audio_content,
        ContentTypeEnum.VIDEO: lambda x: None,  # Not supported yet.
    }

    endpoint_url = task_props.get("endpoint_url") or validated_config.embedding_nim_endpoint
    model_name = task_props.get("model_name") or validated_config.embedding_model
    api_key = task_props.get("api_key") or validated_config.api_key
    filter_errors = task_props.get("filter_errors", False)

    logger.debug("Generating text embeddings for supported content types: TEXT, STRUCTURED, IMAGE, AUDIO.")

    # Process each supported content type.
    for content_type, content_getter in pandas_content_extractor.items():
        if not content_getter:
            logger.debug(f"Skipping unsupported content type: {content_type}")
            continue

        # Create a mask for rows with the desired document type.
        content_mask = df["document_type"] == content_type.value
        if not content_mask.any():
            continue

        # Extract content from metadata and filter out rows with empty content.
        extracted_content = df.loc[content_mask, "metadata"].apply(content_getter)
        non_empty_mask = extracted_content.notna() & (extracted_content.str.strip() != "")
        final_mask = content_mask & non_empty_mask
        if not final_mask.any():
            continue

        # Select and copy the rows that pass the mask.
        df_content = df.loc[final_mask].copy().reset_index(drop=True)
        filtered_content = df_content["metadata"].apply(content_getter)
        # Create batches of content.
        filtered_content_batches = _generate_batches(filtered_content.tolist(), batch_size=validated_config.batch_size)
        # Run asynchronous embedding requests.
        content_embeddings = _async_runner(
            filtered_content_batches,
            api_key,
            endpoint_url,
            model_name,
            validated_config.encoding_format,
            validated_config.input_type,
            validated_config.truncate,
            filter_errors,
        )
        # Apply the embeddings (and any error info) to each row.
        df_content[["metadata", "document_type", "_contains_embeddings"]] = df_content.apply(
            _add_embeddings, **content_embeddings, axis=1
        )[["metadata", "document_type", "_contains_embeddings"]]
        df_content["_content"] = filtered_content

        embedding_dataframes.append(df_content)
        content_masks.append(final_mask)

    # Concatenate the processed rows with the remaining rows.
    combined_df = _concatenate_extractions_pandas(df, embedding_dataframes, content_masks)
    return combined_df, {"trace_info": trace_info}


def generate_text_embed_extractor_stage(
    c: Any,
    stage_config: Dict[str, Any],
    task: str = "embed",
    task_desc: str = "text_embed_extraction",
    pe_count: int = 1,
):
    """
    Generates a multiprocessing stage to perform text embedding extraction from a pandas DataFrame.

    Parameters
    ----------
    c : Config
        Global configuration object.
    stage_config : Dict[str, Any]
        Configuration parameters for the text embedding extractor, validated against EmbedExtractionsSchema.
    task : str, optional
        The task name for the stage worker function (default: "embed").
    task_desc : str, optional
        A descriptor used for latency tracing and logging (default: "text_embed_extraction").
    pe_count : int, optional
        Number of process engines to use concurrently (default: 1).

    Returns
    -------
    MultiProcessingBaseStage
        A configured stage with a worker function that takes a pandas DataFrame, enriches it with embeddings,
        and returns a tuple of (pandas DataFrame, trace_info dict).
    """
    # Validate the stage configuration.
    validated_config = EmbedExtractionsSchema(**stage_config)
    # Wrap the new embedding function with the validated configuration.
    _wrapped_process_fn = functools.partial(_generate_text_embeddings_df, validated_config=validated_config)
    # Return the configured stage.
    return MultiProcessingBaseStage(
        c=c, pe_count=pe_count, task=task, task_desc=task_desc, process_fn=_wrapped_process_fn
    )
