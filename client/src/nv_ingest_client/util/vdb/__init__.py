from nv_ingest_client.util.vdb.adt_vdb import VDB
from nv_ingest_client.util.vdb.milvus import Milvus
from nv_ingest_client.util.vdb.teradata import Teradata
available_vdb_ops = {
    "milvus": Milvus,
    "teradata": Teradata,
}
__all__ = ["VDB", "available_vdb_ops"]
