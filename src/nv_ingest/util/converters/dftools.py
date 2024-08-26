# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json

import fastparquet
import pandas as pd

import cudf


class MemoryFiles:
    def __init__(self):
        self.output = {}

    def open(self, fn, mode="rb"):
        if mode != "wb":
            try:
                self.output[fn].seek(0)
            except KeyError:
                raise FileNotFoundError
            return self.output[fn]

        i = io.BytesIO()
        self.output[fn] = i
        self.output[fn].close = lambda: None

        return i


def pandas_to_cudf(
    df: pd.DataFrame,
    deserialize_cols: list = [],
    default_cols: dict = {"document_type": str, "metadata": str},
    default_type: type = str,
) -> cudf.DataFrame:
    """
    Helper function to convert from pandas to cudf until https://github.com/apache/arrow/pull/40412 is resolved.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe.
    Returns
    -------
    cudf.DataFrame
        A cuDF dataframe.
    """

    if not df.empty:
        files = MemoryFiles()
        for col in deserialize_cols:
            df[col] = df[col].apply(lambda x: json.loads(x))
        df = pd.concat([df, df.iloc[0:1]], axis=0)

        fastparquet.write("_", df, open_with=files.open, compression="UNCOMPRESSED", object_encoding="json")

        with files.output["_"] as bytes_buf:
            gdf = cudf.read_parquet(bytes_buf).iloc[:-1]
            gdf.index.name = None

            return gdf
    else:
        gdf = cudf.DataFrame({col: [] for col in default_cols})
        for col in df.columns:
            field_type = default_cols.get(col, default_type)
            gdf[col] = gdf[col].astype(field_type)

        return gdf


def cudf_to_pandas(gdf: cudf.DataFrame, deserialize_cols: list = []) -> pd.DataFrame:
    """
    Helper function to convert from cudf to pandas until https://github.com/apache/arrow/pull/40412 is resolved.

    Parameters
    ----------
    gdf : cudf.DataFrame
        A cuDF dataframe.
    nested_cols : list
        A list of columns containing nested data.
    Returns
    -------
    pd.DataFrame
        A pandas dataframe.
    """

    with io.BytesIO() as bytes_buf:
        gdf.to_parquet(bytes_buf)
        df = pd.read_parquet(bytes_buf, engine="fastparquet", index=None)

        for col in deserialize_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x))

        return df


def cudf_to_json(gdf: cudf.DataFrame, deserialize_cols: list = []) -> str:
    """
    Helper function to convert from cudf to json until https://github.com/apache/arrow/pull/40412 is resolved.

    Parameters
    ----------
    gdf : cudf.DataFrame
        A cuDF dataframe.
    nested_cols : list
        A list of columns containing nested data.
    Returns
    -------
    str
        A JSON formated string.
    """

    records = []
    dict_vals = cudf_to_pandas(gdf).to_dict(orient="records")
    for d in dict_vals:
        temp = {}
        for key, val in d.items():
            if key in deserialize_cols:
                val = json.loads(val)
            temp[key] = val
        records.append(temp)

    return records
