import re
import logging
from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.text_embedding import EmbeddingModelInterface

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
    input_names: list = ["text"],
    output_names: list = ["embeddings"],
    dtypes: list = ["BYTES"],
):
    """
    This function takes the input data and creates a list of embeddings
    using the NVIDIA embedding microservice.
    """
    if isinstance(data[0], str):
        data = {"prompts": data}
    else:
        data = {"prompts": [res["metadata"]["content"] for res in data]}
    if grpc:
        model_name = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        client = NimClient(
            model_interface=EmbeddingModelInterface(),
            protocol="grpc",
            endpoints=(embedding_endpoint, None),
            auth_token=nvidia_api_key,
        )
        return client.infer(
            data,
            model_name,
            parameters={"input_type": input_type, "truncate": truncate},
            dtypes=dtypes,
            input_names=input_names,
            batch_size=batch_size,
            output_names=output_names,
        )
    else:
        embedding_endpoint = f"{embedding_endpoint}/embeddings"
        client = NimClient(
            model_interface=EmbeddingModelInterface(),
            protocol="http",
            endpoints=(None, embedding_endpoint),
            auth_token=nvidia_api_key,
        )
        return client.infer(data, model_name, input_type=input_type, truncate=truncate, batch_size=batch_size)
