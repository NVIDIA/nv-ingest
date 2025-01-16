import pytest
from nv_ingest_client.util.milvus import MilvusOperator, _dict_to_params


@pytest.fixture
def milvus_test_dict():
    mil_op = MilvusOperator()
    kwargs = mil_op.milvus_kwargs
    kwargs["collection_name"] = mil_op.collection_name
    return kwargs


def test_extra_kwargs(milvus_test_dict):
    mil_op = MilvusOperator(filter_errors=True)
    milvus_test_dict.pop("collection_name")
    assert mil_op.milvus_kwargs == milvus_test_dict


@pytest.mark.parametrize("collection_name", [None, "name"])
def test_op_collection_name(collection_name):
    if collection_name:
        mo = MilvusOperator(collection_name=collection_name)
    else:
        # default
        collection_name = "nv_ingest_collection"
        mo = MilvusOperator()
    cr_collection_name, conn_params = mo.get_connection_params()
    wr_collection_name, write_params = mo.get_write_params()
    assert cr_collection_name == wr_collection_name == collection_name


def test_op_connection_params(milvus_test_dict):
    mo = MilvusOperator()
    cr_collection_name, conn_params = mo.get_connection_params()
    assert cr_collection_name == milvus_test_dict["collection_name"]
    for k, v in conn_params.items():
        assert milvus_test_dict[k] == v


def test_op_write_params(milvus_test_dict):
    mo = MilvusOperator()
    collection_name, wr_params = mo.get_write_params()
    assert collection_name == milvus_test_dict["collection_name"]
    for k, v in wr_params.items():
        assert milvus_test_dict[k] == v


@pytest.mark.parametrize(
    "collection_name, expected_results",
    [
        ({"text": ["text", "charts", "tables"]}, {"enable_text": True, "enable_charts": True, "enable_tables": True}),
        ({"text": ["text", "tables"]}, {"enable_text": True, "enable_charts": False, "enable_tables": True}),
        ({"text": ["text", "charts"]}, {"enable_text": True, "enable_charts": True, "enable_tables": False}),
        ({"text": ["text"]}, {"enable_text": True, "enable_charts": False, "enable_tables": False}),
    ],
)
def test_op_dict_to_params(collection_name, expected_results):
    mo = MilvusOperator()
    _, wr_params = mo.get_write_params()
    response = _dict_to_params(collection_name, wr_params)
    if isinstance(collection_name, str):
        collection_name = {collection_name: None}
    for res in response:
        coll_name, write_params = res
        for k, v in expected_results.items():
            assert write_params[k] == v
        coll_name in collection_name.keys()
