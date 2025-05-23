from nv_ingest_client.util.vdb.adt_vdb import VDB
from nv_ingest_client.util.vdb.milvus import Milvus

available_vdb_ops = {
    "milvus": Milvus,
}
__all__ = ["VDB", "available_vdb_ops"]
