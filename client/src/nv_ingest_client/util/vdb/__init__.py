from nv_ingest_client.util.vdb.adt_vdb import VDB

def available_vdb_ops():
    # Lazy import to avoid requiring pymilvus unless needed
    from nv_ingest_client.util.vdb.milvus import Milvus
    return {"milvus": Milvus}

__all__ = ["VDB", "available_vdb_ops"]
