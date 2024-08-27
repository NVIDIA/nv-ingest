# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64

import pandas as pd
import pytest

from nv_ingest.modules.filters.image_dedup import _apply_dedup_filter
from nv_ingest.modules.filters.image_dedup import _cpu_only_apply_dedup_filter
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if CUDA_DRIVER_OK and MORPHEUS_IMPORT_OK:
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    import cudf


def valid_image_dedup_task(should_filter):
    return {
        "type": "filter",
        "task_properties": {"content_type": "image", "params": {"filter": should_filter}},
    }


def valid_image_metadata(src_content, width, height):
    image_metadata = {"image_type": ImageTypeEnum.PNG, "width": width, "height": height}

    content_metadata = {"type": ContentTypeEnum.IMAGE}

    encoding = "utf-8"
    content = base64.b64encode(bytes(src_content, encoding=encoding)).decode(encoding)

    return {"content": content, "image_metadata": image_metadata, "content_metadata": content_metadata}


def valid_image_dedup_payload(content, width=1, height=1):
    unified_metadata = {
        "source_metadata": {
            "source_name": "test",
            "source_id": "test",
            "source_type": SourceTypeEnum.PDF,
        },
    }

    metadata = valid_image_metadata(content, width, height)

    unified_metadata.update(metadata)
    validated_unified_metadata = validate_metadata(unified_metadata).dict()

    return [ContentTypeEnum.IMAGE, validated_unified_metadata]


def create_ctrl_msg(task):
    ctrl_msg = ControlMessage()
    ctrl_msg.add_task(task["type"], task["task_properties"])

    return ctrl_msg


@pytest.mark.skipif(not MORPHEUS_IMPORT_OK, reason="Morpheus modules are not available.")
@pytest.mark.skipif(
    not CUDA_DRIVER_OK,
    reason="Test environment does not have a compatible CUDA driver.",
)
@pytest.mark.parametrize(
    "should_filter, expected0, expected1, expected2",
    [
        (True, 1, 1, 0),  # filter duplicate images
        (False, 3, 1, 2),  # insert info_message
    ],
)
def test_apply_dedup(should_filter, expected0, expected1, expected2):
    img_dedup_task = valid_image_dedup_task(should_filter)
    ctrl_msg = create_ctrl_msg(img_dedup_task)
    task_props = ctrl_msg.remove_task("filter")
    task_params = task_props.get("params", {})
    filter_flag = task_params.get("filter", True)

    assert task_props.get("content_type") == ContentTypeEnum.IMAGE

    payload_list = []
    for _ in range(3):
        payload_list.append(valid_image_dedup_payload("test", 1, 1))

    extracted_df = pd.DataFrame(payload_list, columns=["document_type", "metadata"])
    extracted_gdf = cudf.from_pandas(extracted_df)
    msg_meta = MessageMeta(df=extracted_gdf)
    ctrl_msg.payload(msg_meta)

    _apply_dedup_filter(ctrl_msg, filter_flag)

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        assert mdf.shape[0] == expected0
        assert (mdf["document_type"] == ContentTypeEnum.IMAGE.value).sum() == expected1
        assert (mdf.iloc[0:3]["document_type"] == ContentTypeEnum.INFO_MSG.value).sum() == expected2


@pytest.mark.parametrize(
    "should_filter, expected0, expected1, expected2",
    [
        (True, 1, 1, 0),  # filter duplicate images
        (False, 3, 1, 2),  # insert info_message
    ],
)
def test_cpu_only_apply_dedup(should_filter, expected0, expected1, expected2):
    payload_list = []
    for _ in range(3):
        payload_list.append(valid_image_dedup_payload("test", 1, 1))

    extracted_df = pd.DataFrame(payload_list, columns=["document_type", "metadata"])
    result_df = _cpu_only_apply_dedup_filter(extracted_df, should_filter)

    assert result_df.shape[0] == expected0
    assert (result_df["document_type"] == ContentTypeEnum.IMAGE).sum() == expected1
    assert (result_df.iloc[0:3]["document_type"] == ContentTypeEnum.INFO_MSG).sum() == expected2
