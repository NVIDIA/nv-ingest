# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

from nv_ingest.modules.filters.image_filter import _apply_filter
from nv_ingest.modules.filters.image_filter import _cpu_only_apply_filter
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata

from ....import_checks import CUDA_DRIVER_OK
from ....import_checks import MORPHEUS_IMPORT_OK

if CUDA_DRIVER_OK and MORPHEUS_IMPORT_OK:
    from morpheus.messages import ControlMessage
    from morpheus.messages import MessageMeta

    import cudf


def valid_image_filter_task(should_filter):
    return {
        "type": "filter",
        "task_properties": {
            "type": "image",
            "params": {"min_size": 256, "max_aspect_ratio": 5.0, "min_aspect_ratio": 0.2, "filter": should_filter},
        },
    }


def valid_text_metadata():
    text_metadata = {
        "text_type": TextTypeEnum.PAGE,
    }

    content_metadata = {"type": ContentTypeEnum.TEXT}

    return {
        "text_metadata": text_metadata,
        "content_metadata": content_metadata,
    }


def valid_image_metadata(width, height):
    image_metadata = {"image_type": ImageTypeEnum.PNG, "width": width, "height": height}

    content_metadata = {"type": ContentTypeEnum.IMAGE}

    return {"image_metadata": image_metadata, "content_metadata": content_metadata}


def valid_image_filter_payload(content_type, width=1, height=1):
    unified_metadata = {
        "source_metadata": {
            "source_name": "test",
            "source_id": "test",
            "source_type": SourceTypeEnum.PDF,
        },
    }

    if content_type == ContentTypeEnum.IMAGE:
        metadata = valid_image_metadata(width, height)
    else:
        metadata = valid_text_metadata()

    unified_metadata.update(metadata)
    validated_unified_metadata = validate_metadata(unified_metadata).dict()

    return [content_type, validated_unified_metadata]


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
    "should_filter, width, height, expected0, expected1",
    [
        (True, 1, 1, 0, 0),  # filter small image
        (True, 1, 100, 0, 0),  # filter small aspect ratio
        (True, 100, 1, 0, 0),  # filter large aspect ratio
        (False, 1, 1, 3, 3),  # no-filter small image
        (False, 1, 100, 3, 3),  # no-filter small aspect ratio
        (False, 100, 1, 3, 3),  # no-filter large aspect ratio
    ],
)
def test_apply_filter(should_filter, width, height, expected0, expected1):
    img_filter_task = valid_image_filter_task(should_filter)
    ctrl_msg = create_ctrl_msg(img_filter_task)
    task_props = ctrl_msg.remove_task("filter")

    assert task_props.get("type") == ContentTypeEnum.IMAGE

    task_params = task_props.get("params")

    payload_list = []
    for i in range(3):
        payload_list.append(valid_image_filter_payload(ContentTypeEnum.IMAGE, width, height))

    extracted_df = pd.DataFrame(payload_list, columns=["document_type", "metadata"])
    extracted_gdf = cudf.from_pandas(extracted_df)
    msg_meta = MessageMeta(df=extracted_gdf)
    ctrl_msg.payload(msg_meta)

    _apply_filter(ctrl_msg, task_params)

    with ctrl_msg.payload().mutable_dataframe() as mdf:
        assert mdf.shape[0] == expected0
        assert (mdf.iloc[0:3]["document_type"] == ContentTypeEnum.INFO_MSG.value).sum() == expected1


@pytest.mark.parametrize(
    "should_filter, width, height, expected0, expected1",
    [
        (True, 1, 1, 0, 0),  # filter small image
        (True, 1, 100, 0, 0),  # filter small aspect ratio
        (True, 100, 1, 0, 0),  # filter large aspect ratio
        (False, 1, 1, 3, 3),  # no-filter small image
        (False, 1, 100, 3, 3),  # no-filter small aspect ratio
        (False, 100, 1, 3, 3),  # no-filter large aspect ratio
    ],
)
def test_cpu_only_apply_filter(should_filter, width, height, expected0, expected1):
    task = valid_image_filter_task(should_filter)
    task_props = task.get("task_properties")
    task_params = task_props.get("params")

    payload_list = []
    for _ in range(3):
        payload_list.append(valid_image_filter_payload(ContentTypeEnum.IMAGE, width, height))

    extracted_df = pd.DataFrame(payload_list, columns=["document_type", "metadata"])
    result_df = _cpu_only_apply_filter(extracted_df, task_params)

    assert result_df.shape[0] == expected0
    assert (result_df.iloc[0:3]["document_type"] == ContentTypeEnum.INFO_MSG).sum() == expected1
