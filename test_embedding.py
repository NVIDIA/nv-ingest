import os
import json
import requests
from pymilvus import Collection, connections, utility

try:
    embedding_http_endpoint = os.getenv("EMBEDDING_HTTP_ENDPOINT", "http://ipp1-3304.ipp1u1.colossus.nvidia.com:8012/v1/embeddings")
    print(f"Embedding endpoint: {embedding_http_endpoint}")
    
    connections.connect("default", host="ipp1-3304.ipp1u1.colossus.nvidia.com", port="19530")  # Update with your Milvus host/port
    COLLECTION_NAME = "nv_ingest_collection"  # Update to your Milvus collection name
    
    # embedding JSON payload
    payload = {
        "input": ["Hello World"],
        "model": "nvidia/nv-embedqa-e5-v5",
        "input_type": "query"
    }

    # Headers
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }
    
    result = requests.post(embedding_http_endpoint, data=json.dumps(payload), headers=headers)
    resp_json = json.loads(result.text)
    question_embedding = resp_json['data'][0]['embedding']
    top_k = 5
    
    collection = Collection(COLLECTION_NAME)
    results = collection.search(
        data=[question_embedding],
        anns_field="vector",  # Vector field in your collection
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["content"]  # Update based on your schema
    )
    breakpoint()
    docs = [hit.entity.get("content") for hit in results[0]]
    print(f"something")
    breakpoint()
    
except Exception as e:
    breakpoint()
    print(f"Exception embedding text: {e}")
