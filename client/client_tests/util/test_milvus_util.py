# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nv_ingest_client.util.milvus import (
    MilvusOperator,
    _dict_to_params,
    create_nvingest_collection,
    grab_meta_collection_info,
    reconstruct_pages,
    add_metadata,
    pandas_file_reader,
    _pull_text,
)
from nv_ingest_client.util.util import ClientConfigSchema
import pandas as pd
import logging


@pytest.fixture
def milvus_test_dict():
    mil_op = MilvusOperator()
    kwargs = mil_op.milvus_kwargs
    kwargs["collection_name"] = mil_op.collection_name
    return kwargs


def test_extra_kwargs(milvus_test_dict):
    mil_op = MilvusOperator(filter_errors=True)
    assert "filter_errors" in mil_op.milvus_kwargs
    assert mil_op.milvus_kwargs["filter_errors"] is True


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
    assert entity["indexes"]["dense_index"] == "FLAT"
    assert entity["indexes"]["sparse_index"] == "SPARSE_INVERTED_INDEX" if sparse else "None"


def test_milvus_meta_multiple_coll(tmp_path):
    collection_name = "collection"
    milvus_uri = f"{tmp_path}/test.db"
    for i in range(0, 3):
        create_nvingest_collection(f"{collection_name}{i}", milvus_uri=milvus_uri)

    results = grab_meta_collection_info(f"{collection_name}2", milvus_uri=milvus_uri)
    entity = results[0]
    assert len(results) == 1
    assert entity["collection_name"] == f"{collection_name}2"


@pytest.fixture
def records():
    return [
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


@pytest.fixture
def metadata():
    return pd.DataFrame(
        {
            "source_name": ["file_1.pdf", "file_2.pdf"],
            "meta_a": ["meta_a_1", "meta_a_2"],
            "meta_b": ["meta_b_1", "meta_b_2"],
        }
    )


def test_page_reconstruction(records):

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


def test_metadata_add(records, metadata):
    for record in records:
        for element in record:
            add_metadata(element, metadata, "source_name", ["meta_a", "meta_b"])

            assert "meta_a" in [*element["metadata"]["content_metadata"]]
            assert "meta_b" in [*element["metadata"]["content_metadata"]]
            idx = element["metadata"]["source_metadata"]["source_name"].split("_")[1].split(".")[0]
            assert element["metadata"]["content_metadata"]["meta_a"] == f"meta_a_{idx}"
            assert element["metadata"]["content_metadata"]["meta_b"] == f"meta_b_{idx}"


def test_metadata_import(metadata, tmp_path):
    file_name = f"{tmp_path}/meta.json"
    metadata.to_json(file_name)
    df = pandas_file_reader(file_name)
    pd.testing.assert_frame_equal(df, metadata)
    file_name = f"{tmp_path}/meta.csv"
    metadata.to_csv(file_name)
    df = pandas_file_reader(file_name)
    pd.testing.assert_frame_equal(df, metadata)
    file_name = f"{tmp_path}/meta.pq"
    metadata.to_parquet(file_name)
    df = pandas_file_reader(file_name)
    pd.testing.assert_frame_equal(df, metadata)
    file_name = f"{tmp_path}/meta.parquet"
    metadata.to_parquet(file_name)
    df = pandas_file_reader(file_name)
    pd.testing.assert_frame_equal(df, metadata)


def test_pull_text_length_limit(caplog):
    """Test that _pull_text handles text longer than 65535 characters correctly."""
    # Create a test element with text longer than 65535 characters
    long_text = "x" * 65536  # Create text that exceeds the limit
    test_element = {
        "document_type": "text",
        "metadata": {
            "content": long_text,
            "embedding": [0.1, 0.2, 0.3],  # Add a valid embedding
            "source_metadata": {"source_name": "test_file.txt"},
            "content_metadata": {"page_number": 1},
        },
    }

    # Set up logging capture
    caplog.set_level(logging.WARNING)

    # Call _pull_text with the test element
    result = _pull_text(
        element=test_element,
        enable_text=True,
        enable_charts=True,
        enable_tables=True,
        enable_images=True,
        enable_infographics=True,
        enable_audio=True,
    )

    # Verify that None is returned
    assert result is None

    # Verify that a warning was logged
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "Text is too long" in caplog.records[0].message
    assert "text_length: 65536" in caplog.records[0].message
    assert "file_name: test_file.txt" in caplog.records[0].message
    assert "page_number: 1" in caplog.records[0].message
