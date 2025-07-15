import logging


from nv_ingest_client.util.vdb.adt_vdb import VDB
import opensearchpy as opensearch
from nv_ingest_client.util.util import ClientConfigSchema

logger = logging.getLogger(__name__)


class OpenSearch(VDB):
    def __init__(self, **kwargs):
        self.host = kwargs.get("host", "localhost")
        self.port = kwargs.get("port", 9200)
        self.username = kwargs.get("username", "admin")
        self.password = kwargs.get("password", "admin")
        self.use_ssl = kwargs.get("use_ssl", False)
        self.verify_certs = kwargs.get("verify_certs", False)
        self.http_compress = kwargs.get("http_compress", False)
        self.dense_dim = kwargs.get("dense_dim", 2048)
        self.index_name = kwargs.get("index_name", "nv_ingest_test")
        self.enable_text = kwargs.get("enable_text", True)
        self.enable_charts = kwargs.get("enable_charts", True)
        self.enable_tables = kwargs.get("enable_tables", True)
        self.enable_images = kwargs.get("enable_images", True)
        self.enable_infographics = kwargs.get("enable_infographics", True)
        self.enable_audio = kwargs.get("enable_audio", True)
        super().__init__(**kwargs)

        self.client = opensearch.OpenSearch(
            hosts=[{"host": self.host, "port": self.port}],
            http_compress=self.http_compress,
            # http_auth=(self.username, self.password),
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
        )

    def create_index(self, **kwargs):
        recreate = kwargs.get("recreate", False)
        exists = self.client.indices.exists(index=f"{self.index_name}_dense")
        if recreate and exists:
            self.client.indices.delete(index=f"{self.index_name}_dense")
            exists = False
        if not exists:
            index_body = {
                "settings": {
                    "index.knn": True,
                    "index.knn.algo_param.ef_search": 100,
                },
                "mappings": {
                    "properties": {
                        "dense": {
                            "type": "knn_vector",
                            "dimension": self.dense_dim,
                            "method": {
                                "name": "hnsw",
                                "engine": "faiss",
                                "space_type": "l2",
                                "parameters": {"m": 16, "ef_construction": 100},
                            },
                        },
                        "text": {"type": "text"},
                        "id": {"type": "keyword"},
                        "metadata": {"type": "object"},
                    }
                },
            }

            self.client.indices.create(index=f"{self.index_name}_dense", body=index_body)

    def write_to_index(self, records: list, **kwargs):
        count = 0
        for record_set in records:
            for record in record_set:
                transform_record = self.transform_record(record)
                if transform_record:
                    self.client.index(index=f"{self.index_name}_dense", body=transform_record, id=count)
                    count += 1

    def retrieval(self, queries: list, **kwargs):
        client_config = ClientConfigSchema()
        index_name = kwargs.get("index_name", f"{self.index_name}_dense")
        top_k = kwargs.get("top_k", 10)
        nvidia_api_key = kwargs.get("nvidia_api_key" or client_config.nvidia_api_key)
        # required for NVIDIAEmbedding call if the endpoint is Nvidia build api.
        embedding_endpoint = kwargs.get("embedding_endpoint", client_config.embedding_nim_endpoint)
        model_name = kwargs.get("model_name", client_config.embedding_nim_model_name)
        from llama_index.embeddings.nvidia import NVIDIAEmbedding

        dense_model = NVIDIAEmbedding(base_url=embedding_endpoint, model=model_name, nvidia_api_key=nvidia_api_key)
        results = []
        for query in queries:
            embedding = dense_model.get_query_embedding(query)

            query_body = {
                "query": {
                    "knn": {
                        "dense": {
                            "vector": embedding,
                            "k": top_k,
                        },
                    },
                }
            }
            response = [
                hit["_source"]
                for hit in self.client.search(index=f"{index_name}_dense", body=query_body)["hits"]["hits"]
            ]
            for res in response:
                res.pop("dense")
            results.append(response)
        return results

    def run(self, records):
        self.create_index()
        self.write_to_index(records)

    def transform_record(self, record: dict):
        text = _pull_text(
            record,
            self.enable_text,
            self.enable_charts,
            self.enable_tables,
            self.enable_images,
            self.enable_infographics,
            self.enable_audio,
        )
        if text:
            return {
                "dense": record["metadata"]["embedding"],
                "text": text,
                "metadata": record["metadata"]["content_metadata"],
            }
        else:
            return None


def verify_embedding(element):
    if element["metadata"]["embedding"] is not None:
        return True
    return False


def _pull_text(
    element,
    enable_text: bool,
    enable_charts: bool,
    enable_tables: bool,
    enable_images: bool,
    enable_infographics: bool,
    enable_audio: bool,
):
    text = None
    if element["document_type"] == "text" and enable_text:
        text = element["metadata"]["content"]
    elif element["document_type"] == "structured":
        text = element["metadata"]["table_metadata"]["table_content"]
        if element["metadata"]["content_metadata"]["subtype"] == "chart" and not enable_charts:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "table" and not enable_tables:
            text = None
        elif element["metadata"]["content_metadata"]["subtype"] == "infographic" and not enable_infographics:
            text = None
    elif element["document_type"] == "image" and enable_images:
        text = element["metadata"]["image_metadata"]["caption"]
    elif element["document_type"] == "audio" and enable_audio:
        text = element["metadata"]["audio_metadata"]["audio_transcript"]
    verify_emb = verify_embedding(element)
    if not text or not verify_emb:
        source_name = element["metadata"]["source_metadata"]["source_name"]
        pg_num = element["metadata"]["content_metadata"].get("page_number", None)
        doc_type = element["document_type"]
        if not verify_emb:
            logger.debug(f"failed to find embedding for entity: {source_name} page: {pg_num} type: {doc_type}")
        if not text:
            logger.debug(f"failed to find text for entity: {source_name} page: {pg_num} type: {doc_type}")
        # if we do find text but no embedding remove anyway
        text = None
    if text and len(text) > 65535:
        logger.warning(
            f"Text is too long, skipping. It is advised to use SplitTask, to make smaller chunk sizes."
            f"text_length: {len(text)}, file_name: {element['metadata']['source_metadata'].get('source_name', None)} "
            f"page_number: {element['metadata']['content_metadata'].get('page_number', None)}"
        )
        text = None
    return text
