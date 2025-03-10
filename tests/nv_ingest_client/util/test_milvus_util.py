import pytest
from nv_ingest_client.util.milvus import (
    MilvusOperator,
    _dict_to_params,
    create_nvingest_collection,
    grab_meta_collection_info,
    reconstruct_pages,
)
from nv_ingest_client.util.util import ClientConfigSchema


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


@pytest.mark.parametrize("sparse", [True, False])
def test_milvus_meta_collection(tmp_path, sparse):
    collection_name = "collection"
    milvus_uri = f"{tmp_path}/test.db"
    create_nvingest_collection(collection_name, milvus_uri=milvus_uri, sparse=sparse)
    results = grab_meta_collection_info(collection_name, milvus_uri=milvus_uri)
    keys = list(results[0].keys())
    assert ["pk", "collection_name", "indexes", "models", "timestamp", "user_fields"] == keys
    entity = results[0]
    env_schema = ClientConfigSchema()
    assert entity["collection_name"] == collection_name
    assert entity["models"]["embedding_model"] == env_schema.embedding_nim_model_name


def test_milvus_meta_multiple_coll(tmp_path):
    collection_name = "collection"
    milvus_uri = f"{tmp_path}/test.db"
    for i in range(0, 3):
        create_nvingest_collection(f"{collection_name}{i}", milvus_uri=milvus_uri)

    results = grab_meta_collection_info(f"{collection_name}2", milvus_uri=milvus_uri)
    entity = results[0]
    assert len(results) == 1
    assert entity["collection_name"] == f"{collection_name}2"


def test_page_reconstruction():
    records = [
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "roses are red.",
                    "source_metadata": {
                        "source_name": "file_1.pdf",
                    },
                    "content_metadata": {
                        "type": "text",
                        "page_number": 0,
                    },
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "content": "violets are blue.",
                    "source_metadata": {
                        "source_name": "file_1.pdf",
                    },
                    "content_metadata": {
                        "type": "text",
                        "page_number": 0,
                    },
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "content": "sunflowers are yellow.",
                    "source_metadata": {
                        "source_name": "file_1.pdf",
                    },
                    "content_metadata": {
                        "type": "text",
                        "page_number": 0,
                    },
                },
            },
        ],
        [
            {
                "document_type": "text",
                "metadata": {
                    "content": "two time two is four.",
                    "source_metadata": {
                        "source_name": "file_2.pdf",
                    },
                    "content_metadata": {
                        "type": "text",
                        "page_number": 0,
                    },
                },
            },
            {
                "document_type": "text",
                "metadata": {
                    "content": "four times four is sixteen.",
                    "source_metadata": {
                        "source_name": "file_2.pdf",
                    },
                    "content_metadata": {
                        "type": "text",
                        "page_number": 1,
                    },
                },
            },
        ],
    ]
    candidates = [
        {
            "id": 456331433807935937,
            "distance": 0.016393441706895828,
            "entity": {
                "text": "roses are red.",
                "source": {
                    "source_name": "file_1.pdf",
                },
                "content_metadata": {
                    "type": "text",
                    "page_number": 0,
                },
            },
        },
        {
            "id": 456331433807935937,
            "distance": 0.016393441706895828,
            "entity": {
                "text": "two time two is four.",
                "source": {
                    "source_name": "file_2.pdf",
                },
                "content_metadata": {
                    "type": "text",
                    "page_number": 0,
                },
            },
        },
    ]
    pages = []
    pages.append(reconstruct_pages(candidates[0], records))
    pages.append(reconstruct_pages(candidates[1], records, page_signum=1))
    assert pages[0] == "roses are red.\nviolets are blue.\nsunflowers are yellow.\n"
    assert pages[1] == "two time two is four.\nfour times four is sixteen.\n"
