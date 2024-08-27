# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nv_ingest.util.converters.dftools import cudf_to_json
from nv_ingest.util.converters.dftools import cudf_to_pandas
from nv_ingest.util.converters.dftools import pandas_to_cudf

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if CUDA_DRIVER_OK and MORPHEUS_IMPORT_OK:
    import pandas as pd

    import cudf


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_pandas_to_cudf():
    nested_input = [[f"test{i}", {f"test{i}": {f"test{i}": "test"}}] for i in range(4)]
    df0 = pd.DataFrame(nested_input, columns=["document_type", "metadata"])
    gdf0 = pandas_to_cudf(df0)
    assert isinstance(gdf0, cudf.DataFrame)
    df1 = cudf_to_pandas(gdf0, deserialize_cols=["document_type", "metadata"])
    df2 = cudf_to_pandas(gdf0, deserialize_cols=["metadata"])
    df3 = cudf_to_pandas(gdf0, deserialize_cols=["bad_col"])
    df4 = cudf_to_pandas(gdf0)
    assert df0.equals(df1)
    assert not df0.equals(df2)
    assert not df0.equals(df3)
    assert not df0.equals(df4)
    gdf1 = pandas_to_cudf(df1)
    assert gdf0.equals(gdf1)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_cudf_to_pandas():
    nested_input = [[json.dumps(f"test{i}"), json.dumps('{"test{i}":{"test":"test"}}')] for i in range(4)]
    gdf0 = cudf.DataFrame(nested_input, columns=["document_type", "metadata"])
    df0 = cudf_to_pandas(gdf0, deserialize_cols=["document_type", "metadata"])
    assert isinstance(df0, pd.DataFrame)
    gdf1 = pandas_to_cudf(df0)
    assert gdf0.equals(gdf1)
    df1 = cudf_to_pandas(gdf1, deserialize_cols=["document_type", "metadata"])
    df2 = cudf_to_pandas(gdf1, deserialize_cols=["metadata"])
    df3 = cudf_to_pandas(gdf1, deserialize_cols=["bad_col"])
    df4 = cudf_to_pandas(gdf1)
    assert df0.equals(df1)
    assert not df0.equals(df2)
    assert not df0.equals(df3)
    assert not df0.equals(df4)


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
def test_cudf_to_json():
    nested_input = [[f"test{i}", {f"test{i}": {f"test{i}": "test"}}] for i in range(4)]
    df = pd.DataFrame(nested_input, columns=["document_type", "metadata"])
    gdf = pandas_to_cudf(df)
    expected_result = df.to_dict(orient="records")
    assert cudf_to_json(gdf, deserialize_cols=["document_type", "metadata"]) == expected_result
    assert not cudf_to_json(gdf, deserialize_cols=["bad_col"]) == expected_result
    assert not cudf_to_json(gdf) == expected_result
