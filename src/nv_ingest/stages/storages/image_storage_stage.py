# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import typing

import mrc
from morpheus.config import Config, ExecutionMode
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.utils.module_utils import ModuleLoader

from nv_ingest.modules.storages.image_storage import ImageStorageLoaderFactory
from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage

logger = logging.getLogger(__name__)


class ImageStorageStage(PassThruTypeMixin, SinglePortStage):
    """
    Stores images.

    Parameters
    ----------
    config : Config
        Pipeline configuration instance.

    Raises
    ------
    """

    def __init__(
        self,
        config: Config,
        module_config: typing.Dict = None,
        raise_on_failure: bool = False,
    ) -> None:
        super().__init__(config)

        if module_config is None:
            module_config = {
                "raise_on_failure": raise_on_failure,
            }

        module_name = "image_storage"

        self._module_loader: ModuleLoader = ImageStorageLoaderFactory.get_instance(module_name, module_config)

    @property
    def name(self) -> str:
        return "image-storage"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(IngestControlMessage, MultiResponseMessage, MultiMessage)
            Accepted input types.

        """
        return (IngestControlMessage,)

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def supported_execution_modes(self) -> tuple[ExecutionMode]:
        # Provide your own logic here; for example:
        return (ExecutionMode.CPU,)

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        module = self._module_loader.load(builder)

        # Input and Output port names should be same as input and output port names of write_to_vector_db module.
        mod_in_node = module.input_port("input")
        mod_out_node = module.output_port("output")

        builder.make_edge(input_node, mod_in_node)

        return mod_out_node
