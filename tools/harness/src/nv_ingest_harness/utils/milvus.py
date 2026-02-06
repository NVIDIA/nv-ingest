"""Milvus-specific utility functions.

This module contains utilities that depend on pymilvus and are only needed
for ingestion benchmarking workflows.
"""

from pymilvus import MilvusClient

from nv_ingest_harness.utils.interact import kv_event_log


def unload_collection(milvus_uri: str, collection_name: str):
    """Unload/release a Milvus collection from memory.

    This function connects to a Milvus instance and releases the specified
    collection from memory, freeing up resources.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to unload.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        client.release_collection(collection_name=collection_name)
    except Exception as e:
        kv_event_log(f"{collection_name}_unload_error", str(e))
        print(f"Error unloading collection {collection_name}: {e}")


def load_collection(milvus_uri: str, collection_name: str):
    """Load a Milvus collection into memory.

    This function connects to a Milvus instance and loads the specified
    collection into memory for querying and operations.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to load.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        client.load_collection(collection_name)
    except Exception as e:
        kv_event_log(f"{collection_name}_load_error", str(e))
        print(f"Error loading collection {collection_name}: {e}")


def milvus_chunks(milvus_uri: str, collection_name: str):
    """Get and log statistics about chunks in a Milvus collection.

    This function connects to a Milvus instance, retrieves collection statistics,
    logs the chunk information using the event logging system, and closes the connection.

    Args:
      milvus_uri (str): The URI connection string for the Milvus instance.
      collection_name (str): The name of the collection to get statistics for.

    Returns:
      dict: A dictionary containing the collection statistics.
    """
    try:
        client = MilvusClient(uri=milvus_uri)
        stats = client.get_collection_stats(collection_name)
        kv_event_log(f"{collection_name}_chunks", f"{stats}")
        client.close()
    except Exception as e:
        kv_event_log(f"{collection_name}_chunks_error", str(e))
        print(f"Error getting collection stats for {collection_name}: {e}")
        stats = {}
    return stats
