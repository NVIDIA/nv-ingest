# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import traceback
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
from morpheus.config import Config
from morpheus.utils.module_utils import ModuleLoaderFactory
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.stages.multiprocessing_stage import MultiProcessingBaseStage

logger = logging.getLogger(__name__)

MODULE_NAME = "image_caption_extraction"
MODULE_NAMESPACE = "nv_ingest"

ImageCaptionExtractionLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE)


def _extract_bboxes_and_content(data: Dict[str, Any]) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """
    Extract bounding boxes and associated content from a deeply nested data structure.

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing nested data from which bounding boxes and content are extracted.

    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], List[str]]
        A tuple containing two lists:
        - First list of tuples representing bounding boxes (x1, y1, x2, y2).
        - Second list of strings representing associated content.
    """
    nearby_objects = data["content_metadata"]["hierarchy"]["nearby_objects"]["text"]
    bboxes = nearby_objects["bbox"]
    content = nearby_objects["content"]
    return bboxes, content


def _calculate_centroids(bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[float, float]]:
    """
    Calculate centroids from bounding boxes.

    Parameters
    ----------
    bboxes : List[Tuple[int, int, int, int]]
        A list of tuples each representing a bounding box as (x1, y1, x2, y2).

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples each representing the centroid (x, y) of the corresponding bounding box.
    """
    return [(bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2) for bbox in bboxes]


def _fit_nearest_neighbors(centroids: List[Tuple[float, float]], n_neighbors: int = 5) -> Tuple[NearestNeighbors, int]:
    """
    Fit the NearestNeighbors model to the centroids, ensuring the number of neighbors does not exceed available
    centroids.

    Parameters
    ----------
    centroids : List[Tuple[float, float]]
        A list of tuples each representing the centroid coordinates (x, y) of bounding boxes.
    n_neighbors : int, optional
        The number of neighbors to use by default for kneighbors queries.

    Returns
    -------
    Tuple[NearestNeighbors, int]
        A tuple containing:
        - NearestNeighbors model fitted to the centroids.
        - The adjusted number of neighbors, which is the minimum of `n_neighbors` and the number of centroids.
    """
    centroids_array = np.array(centroids)
    adjusted_n_neighbors = min(n_neighbors, len(centroids_array))
    nbrs = NearestNeighbors(n_neighbors=adjusted_n_neighbors, algorithm="auto", metric="euclidean")
    nbrs.fit(centroids_array)

    return nbrs, adjusted_n_neighbors


def _find_nearest_neighbors(
    nbrs: NearestNeighbors, new_bbox: Tuple[int, int, int, int], content: List[str], n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Find the nearest neighbors for a new bounding box and return associated content.

    Parameters
    ----------
    nbrs : NearestNeighbors
        The trained NearestNeighbors model.
    new_bbox : Tuple[int, int, int, int]
        The bounding box for which to find the nearest neighbors, specified as (x1, y1, x2, y2).
    content : List[str]
        A list of content strings associated with each bounding box.
    n_neighbors : int
        The number of neighbors to retrieve.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        A tuple containing:
        - distances: An array of distances to the nearest neighbors.
        - indices: An array of indices for the nearest neighbors.
        - A list of content strings corresponding to the nearest neighbors.
    """
    new_centroid = np.array(
        [(new_bbox[0] + (new_bbox[2] - new_bbox[0]) / 2, new_bbox[1] + (new_bbox[3] - new_bbox[1]) / 2)]
    )
    new_centroid_reshaped = new_centroid.reshape(1, -1)  # Reshape to ensure 2D
    distances, indices = nbrs.kneighbors(new_centroid_reshaped, n_neighbors=n_neighbors)
    return distances, indices, [content[i] for i in indices.flatten()]


def _sanitize_inputs(inputs: List[List[str]]) -> List[List[str]]:
    """
    Replace non-ASCII characters with '?' in inputs.

    Parameters
    ----------
    inputs : List[List[str]]
        A list of lists where each sub-list contains strings.

    Returns
    -------
    List[List[str]]
        A list of lists where each string has been sanitized to contain only ASCII characters,
        with non-ASCII characters replaced by '?'.
    """
    cleaned_inputs = [
        [candidate.encode("ascii", "replace").decode("ascii") for candidate in candidates] for candidates in inputs
    ]
    return cleaned_inputs


TOKENIZER_NAME = "microsoft/deberta-large"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _predict_caption(triton_url: str, caption_model: str, inputs: List[List[str]], n_candidates: int = 5) -> List[str]:
    """
    Sends a request to a Triton inference server to generate captions based on provided inputs.

    Parameters
    ----------
    triton_url : str
        The URL of the Triton inference server.
    headers : Dict[str, str]
        HTTP headers to send with the request.
    inputs : List[List[str]]
        The input data for which captions are generated.
    n_candidates : int, optional
        The number of candidates per input data. Default is 5.

    Returns
    -------
    List[str]
        A list of generated captions, one for each input.
    """

    sanitized_inputs = _sanitize_inputs(inputs)
    captions = [""] * len(sanitized_inputs)
    max_batch_size = 128

    try:
        client = grpcclient.InferenceServerClient(url=triton_url)

        # Process inputs in batches
        for batch_start in range(0, len(sanitized_inputs), max_batch_size // n_candidates):
            batch_end = min(batch_start + (max_batch_size // n_candidates), len(sanitized_inputs))
            input_batch = sanitized_inputs[batch_start:batch_end]
            flattened_sentences = [sentence for batch in input_batch for sentence in batch]
            encoded_inputs = tokenizer(
                flattened_sentences, max_length=128, padding="max_length", truncation=True, return_tensors="np"
            )

            input_ids = encoded_inputs["input_ids"].astype(np.int64)
            attention_mask = encoded_inputs["attention_mask"].astype(np.float32)
            infer_inputs = [
                grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
                grpcclient.InferInput("input_mask", attention_mask.shape, "FP32"),
            ]

            # Set the data for the input tensors
            infer_inputs[0].set_data_from_numpy(input_ids)
            infer_inputs[1].set_data_from_numpy(attention_mask)

            outputs = [grpcclient.InferRequestedOutput("output")]

            # Perform inference
            response = client.infer(model_name=caption_model, inputs=infer_inputs, outputs=outputs)

            output_data = response.as_numpy("output")

            # Process the output to find the best sentence in each batch
            batch_size = n_candidates
            for i, batch in enumerate(input_batch):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_output = output_data[start_idx:end_idx]
                max_index = np.argmax(batch_output)
                best_sentence = batch[max_index]
                best_probability = batch_output[max_index]
                if best_probability > 0.5:
                    captions[batch_start + i] = best_sentence

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return captions


def _prepare_dataframes_mod(df) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if df.empty or "document_type" not in df.columns:
        return df, pd.DataFrame(), pd.Series(dtype=bool)

    bool_index = df["document_type"] == ContentTypeEnum.IMAGE
    df_filtered = df.loc[bool_index]

    return df, df_filtered, bool_index


def _process_documents(df_filtered: pd.DataFrame) -> Tuple[List[Any], List[List[str]]]:
    """
    Processes documents to extract content and bounding boxes, then finds nearest neighbors.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        The dataframe filtered to contain only relevant documents.

    Returns
    -------
    Tuple[List[Any], List[List[str]]]
        A tuple containing metadata for each document and a list of neighbor content.
    """
    neighbor_content = []
    metadata_list = []
    for _, row in df_filtered.iterrows():
        metadata = row.metadata
        metadata_list.append(metadata)
        bboxes, content = _extract_bboxes_and_content(metadata)
        _process_content(bboxes, content, metadata, neighbor_content)

    return metadata_list, neighbor_content


def _process_content(
    bboxes: List[Tuple[int, int, int, int]],
    content: List[str],
    metadata: Dict[str, Any],
    neighbor_content: List[List[str]],
    n_neighbors: int = 5,
) -> None:
    """
    Process content by finding nearest neighbors and appending the results.

    Parameters
    ----------
    bboxes : List[Tuple[int, int, int, int]]
        A list of bounding boxes.
    content : List[str]
        Content associated with each bounding box.
    metadata : Dict[str, Any]
        Metadata associated with each content piece, containing image metadata.
    neighbor_content : List[List[str]]
        A list that will be appended with the nearest neighbor content.
    n_neighbors : int, optional
        The number of nearest neighbors to find (default is 5).

    Returns
    -------
    None
    """
    if bboxes and content:
        centroids = _calculate_centroids(bboxes)
        nn_mod, adj_neighbors = _fit_nearest_neighbors(centroids)
        image_bbox = metadata["image_metadata"]["image_location"]
        distances, indices, nearest_content = _find_nearest_neighbors(nn_mod, image_bbox, content, adj_neighbors)
    else:
        nearest_content = []

    if len(nearest_content) < n_neighbors:
        nearest_content.extend([""] * (n_neighbors - len(nearest_content)))

    neighbor_content.append(nearest_content)


def _generate_captions(neighbor_content: List[List[str]], config: Any) -> List[str]:
    """
    Generate captions for provided content using a Triton inference server.

    Parameters
    ----------
    neighbor_content : List[List[str]]
        A list of content batches for which to generate captions.
    config : Any
        Configuration object containing endpoint URL, headers, and batch size.

    Returns
    -------
    List[str]
        A list of generated captions.
    """
    captions = []
    for i in range(0, len(neighbor_content), config.batch_size):
        batch = neighbor_content[i : i + config.batch_size]  # noqa: E203
        batch_captions = _predict_caption(config.endpoint_url, config.caption_classifier_model_name, batch)
        captions.extend(batch_captions)

    return captions


def _update_metadata_with_captions(
    metadata_list: List[Dict[str, Any]], captions: List[str], df_filtered: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Update metadata with captions and compile into a list of image document dictionaries.

    Parameters
    ----------
    metadata_list : List[Dict[str, Any]]
        A list of metadata dictionaries.
    captions : List[str]
        A list of captions corresponding to the metadata.
    df_filtered : pd.DataFrame
        The filtered DataFrame containing document UUIDs.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries each containing updated document metadata and type.
    """
    image_docs = []
    for metadata, caption, (_, row) in zip(metadata_list, captions, df_filtered.iterrows()):
        metadata["image_metadata"]["caption"] = caption
        image_docs.append(
            {
                "document_type": ContentTypeEnum.IMAGE.value,
                "metadata": metadata,
                "uuid": row.get("uuid"),
            }
        )

    return image_docs


def _prepare_final_dataframe_mod(df: pd.DataFrame, image_docs: List[Dict[str, Any]], filter_index: pd.Series) -> None:
    """
    Prepares the final dataframe by combining original dataframe with new image document data, converting to GPU
    dataframe, and updating the message with the new dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe.
    image_docs : List[Dict[str, Any]]
        A list of dictionaries containing image document data.
    filter_index : pd.Series
        A boolean series that filters the dataframe.

    Returns
    -------
    None
    """

    image_docs_df = pd.DataFrame(image_docs)
    docs_df = pd.concat([df[~filter_index], image_docs_df], axis=0).reset_index(drop=True)

    return docs_df


def caption_extract_stage(df, task_props, validated_config) -> pd.DataFrame:
    """
    Extracts captions from images within the provided dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing image data.
    task_props : dict
        Task properties required for processing.
    validated_config : ImageCaptionExtractionSchema
        Validated configuration for caption extraction.

    Returns
    -------
    pd.DataFrame
        The dataframe with updated image captions.

    Raises
    ------
    ValueError
        If an error occurs during caption extraction.
    """
    try:
        logger.debug("Performing caption extraction")

        # Data preparation and filtering
        df, df_filtered, filter_index = _prepare_dataframes_mod(df)
        if df_filtered.empty:
            return df

        # Process each image document
        metadata_list, neighbor_content = _process_documents(df_filtered)

        # Generate captions
        captions = _generate_captions(neighbor_content, validated_config)

        # Update metadata with captions
        image_docs = _update_metadata_with_captions(metadata_list, captions, df_filtered)

        logger.debug(f"Extracted captions from {len(image_docs)} images")

        # Final dataframe merge
        df_final = _prepare_final_dataframe_mod(df, image_docs, filter_index)

        if (df_final is None) or df_final.empty:
            logger.warning("NO IMAGE DOCUMENTS FOUND IN THE DATAFRAME")
            return df
        return df_final
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to do caption extraction: {e}")


def generate_caption_extraction_stage(
    c: Config,
    caption_config: Dict[str, Any],
    task: str = "caption",
    task_desc: str = "caption_extraction",
    pe_count: int = 8,
):
    """
    Generates a caption extraction stage with the specified configuration.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    caption_config : dict
        Configuration parameters for caption extraction.
    task : str, optional
        The task name to match for the stage worker function, by default "caption".
    task_desc : str, optional
        A descriptor to be used in latency tracing, by default "caption_extraction".
    pe_count : int, optional
        Number of processing elements to use, by default 8.

    Returns
    -------
    MultiProcessingBaseStage
        The generated caption extraction stage.

    Raises
    ------
    ValueError
        If an error occurs during stage generation.
    """

    validated_config = ImageCaptionExtractionSchema(**caption_config)
    _wrapped_caption_extract = partial(caption_extract_stage, validated_config=validated_config)

    logger.debug(
        f"Generating caption extraction stage with {pe_count} processing elements. task: {task}, document_type: *"
    )
    return MultiProcessingBaseStage(
        c=c,
        pe_count=pe_count,
        task=task,
        task_desc=task_desc,
        process_fn=_wrapped_caption_extract,
        filter_properties={"content_type": ContentTypeEnum.IMAGE.value},
    )
