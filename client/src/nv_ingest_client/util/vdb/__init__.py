from nv_ingest_client.util.vdb.adt_vdb import VDB


def get_vdb_op_cls(vdb_op: str):
    """
    Lazily import and return the VDB operation class for the given op string.
    Returns the class if found, else raises ValueError.
    """

    available_vdb_ops = ["milvus", "lancedb"]

    if vdb_op == "milvus":
        from nv_ingest_client.util.vdb.milvus import Milvus

        return Milvus

    if vdb_op == "lancedb":
        from nv_ingest_client.util.vdb.lancedb import LanceDB

        return LanceDB

    raise ValueError(f"Invalid vdb_op: {vdb_op}. Available vdb_ops - {available_vdb_ops}.")


__all__ = ["VDB", "get_vdb_op_cls"]
