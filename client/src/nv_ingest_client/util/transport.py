import requests
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import logging


logger = logging.getLogger(__name__)


def infer_microservice(
    data,
    model_name: str = None,
    embedding_endpoint: str = None,
    nvidia_api_key: str = None,
    input_type: str = "passage",
    truncate: str = "END",
    batch_size: int = 8191,
    grpc: bool = False,
):
    """
    This function takes the input data and creates a list of embeddings
    using the NVIDIA embedding microservice.
    """
    if grpc:
        return infer_with_grpc(data, model_name, embedding_endpoint, input_type, truncate, batch_size)
    else:
        return infer_with_http(data, model_name, embedding_endpoint, nvidia_api_key, input_type, truncate, batch_size)


def infer_batch_http(
    text_batch: list[str],
    model_name: str,
    embedding_endpoint: str,
    nvidia_api_key: str = None,
    input_type: str = "passage",
    truncate: str = "END",
):
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if nvidia_api_key:
        headers["Authorization"] = f"Bearer {nvidia_api_key}"
    json_data = {"input": text_batch, "model": model_name, "input_type": input_type, "truncate": truncate}
    response = requests.post(f"{embedding_endpoint}/v1/embeddings", headers=headers, json=json_data)
    if response.status_code != 200:
        raise ValueError(f"Failed retrieving embedding results: {response.status_code} - {response.text}")
    embeddings = [rec["embedding"] for rec in response.json()["data"]]
    if not embeddings:
        raise ValueError(f"Failed retrieving embedding results: {response.status_code} - {response.text}")
    return embeddings


def infer_with_http(
    payload: list[dict],
    model_name: str,
    embedding_endpoint: str,
    nvidia_api_key: str = None,
    input_type: str = "passage",
    truncate: str = "END",
    batch_size: int = 8191,
):
    """
    This function takes the input payload and creates a list of embeddings.
    The default batch size is 8191, but this can be adjusted based on the
    microservice limitations and available resources. This is set by the
    NVIDIA embedding microservice.
    """
    all_embeddings = []
    payload = [res["metadata"]["content"] for res in payload]
    for offset in range(0, len(payload), batch_size):
        text_batch = payload[offset : offset + batch_size]
        embeddings = infer_batch_http(
            text_batch=text_batch,
            embedding_endpoint=embedding_endpoint,
            model_name=model_name,
            nvidia_api_key=nvidia_api_key,
            input_type=input_type,
            truncate=truncate,
        )
        all_embeddings += embeddings
    return all_embeddings


def infer_batch(text_batch: list[str], client: InferenceServerClient, model_name: str, parameters: dict):
    text_np = np.array([[text.encode("utf-8")] for text in text_batch], dtype=np.object_)
    text_input = InferInput("text", text_np.shape, "BYTES")
    text_input.set_data_from_numpy(text_np)
    infer_input = [text_input]
    infer_output = [InferRequestedOutput(nn) for nn in ["token_count", "embeddings"]]
    result = client.infer(
        model_name=model_name,
        parameters=parameters,
        inputs=infer_input,
        outputs=infer_output,
    )
    token_count, embeddings = result.as_numpy("token_count"), result.as_numpy("embeddings")
    return token_count, embeddings


def infer_with_grpc(
    text_ls: list[dict],
    model_name: str,
    grpc_host: str = "localhost:8001",
    input_type: str = "passage",
    truncate: str = "END",
    batch_size: int = 30,
):
    grpc_client = InferenceServerClient(url=grpc_host, verbose=False)
    config = grpc_client.get_model_config(model_name=model_name).config
    max_batch_size = config.max_batch_size
    if batch_size > max_batch_size:
        logger.info(
            f"Batch size {batch_size} is larger than the maximum batch size {max_batch_size} for model {model_name},"
            f" using the maximum batch size of {max_batch_size} instead."
        )
        batch_size = max_batch_size

    embeddings_ls = []
    embed_payload = [res["metadata"]["content"] for res in text_ls]
    for offset in range(0, len(embed_payload), batch_size):
        text_batch = embed_payload[offset : offset + batch_size]
        token_count, embeddings = infer_batch(
            text_batch=text_batch,
            client=grpc_client,
            model_name=model_name,
            parameters={"input_type": input_type, "truncate": truncate},
        )
        embeddings_ls.append(embeddings)
    embeddings = np.concatenate(embeddings_ls)
    return embeddings
