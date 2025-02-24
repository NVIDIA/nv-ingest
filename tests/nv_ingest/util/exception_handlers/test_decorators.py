# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest.util.exception_handlers.decorators import (
    nv_ingest_node_failure_context_manager,
    nv_ingest_source_failure_context_manager,
    CMNVIngestFailureContextManager,
)

import unittest
from unittest.mock import patch

from nv_ingest_api.primitives.ingest_control_message import IngestControlMessage

MODULE_UNDER_TEST = "nv_ingest.util.exception_handlers.decorators"


# A minimal dummy IngestControlMessage for testing purposes.
class DummyIngestControlMessage(IngestControlMessage):
    def __init__(self, payload="default", metadata=None):
        self.payload = payload
        self._metadata = metadata or {}

    def get_metadata(self, key, default=None):
        return self._metadata.get(key, default)

    def set_metadata(self, key, value):
        self._metadata[key] = value


##############################################
# Tests for nv_ingest_node_failure_context_manager
##############################################
class TestNVIngestNodeFailureContextManager(unittest.TestCase):

    @patch(f"{MODULE_UNDER_TEST}.cm_ensure_payload_not_null")
    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    def test_normal_execution(self, mock_annotate, mock_ensure):
        # Create a control message that is not failed and has non-null payload.
        cm = DummyIngestControlMessage(payload="data", metadata={"cm_failed": False})

        @nv_ingest_node_failure_context_manager("annotation1", payload_can_be_empty=False)
        def dummy_node_func(control_message):
            # Mark the message as processed.
            control_message.set_metadata("processed", True)
            return control_message

        result = dummy_node_func(cm)
        self.assertTrue(result.get_metadata("processed"))
        # Verify that payload-check was called.
        mock_ensure.assert_called_once_with(control_message=cm)
        # On successful exit, annotate_task_result should have been called.
        mock_annotate.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    def test_skip_processing_with_forward_func(self, mock_annotate):
        # Simulate a control message that is already marked as failed.
        cm = DummyIngestControlMessage(payload="data", metadata={"cm_failed": True})

        def forward_func(control_message):
            control_message.set_metadata("forwarded", True)
            return control_message

        @nv_ingest_node_failure_context_manager("annotation2", forward_func=forward_func)
        def dummy_node_func(control_message):
            control_message.set_metadata("processed", True)
            return control_message

        result = dummy_node_func(cm)
        self.assertTrue(result.get_metadata("forwarded"))
        self.assertFalse("processed" in result._metadata)

    @patch(
        f"{MODULE_UNDER_TEST}.cm_ensure_payload_not_null",
        side_effect=lambda control_message: (_ for _ in ()).throw(ValueError("payload is null")),
    )
    @pytest.mark.xfail(reason="Fix after IngestCM is merged")
    def test_payload_null_raises_error(self, mock_ensure):
        # When payload is None and payload_can_be_empty is False, an error should be raised.
        cm = DummyIngestControlMessage(payload=None, metadata={"cm_failed": False})

        @nv_ingest_node_failure_context_manager("annotation3", payload_can_be_empty=False)
        def dummy_node_func(control_message):
            control_message.set_metadata("processed", True)
            return control_message

        with self.assertRaises(ValueError):
            dummy_node_func(cm)

    def test_raise_on_failure_propagates_exception(self):
        cm = DummyIngestControlMessage(payload="data", metadata={"cm_failed": False})

        @nv_ingest_node_failure_context_manager("annotation4", payload_can_be_empty=True, raise_on_failure=True)
        def dummy_node_func(control_message):
            raise ValueError("dummy error")

        with self.assertRaises(ValueError):
            dummy_node_func(cm)


##############################################
# Tests for nv_ingest_source_failure_context_manager
##############################################
class TestNVIngestSourceFailureContextManager(unittest.TestCase):

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    def test_normal_execution(self, mock_annotate):
        # Function returns a valid IngestControlMessage with non-null payload.
        def dummy_source_func():
            return DummyIngestControlMessage(payload="data")

        decorated = nv_ingest_source_failure_context_manager("annotation_source")(dummy_source_func)
        result = decorated()
        self.assertIsInstance(result, IngestControlMessage)
        self.assertIsNotNone(result.payload)
        # Expect a success annotation.
        mock_annotate.assert_called_once()

    @pytest.mark.xfail(reason="Fix after IngestCM is merged")
    def test_non_control_message_output(self):
        # Function returns a non-IngestControlMessage.
        def dummy_source_func():
            return 123

        decorated = nv_ingest_source_failure_context_manager("annotation_source")(dummy_source_func)
        with self.assertRaises(TypeError):
            decorated()

    @pytest.mark.xfail(reason="Fix after IngestCM is merged")
    def test_null_payload_raises_value_error(self):
        # Function returns a IngestControlMessage with a null payload.
        def dummy_source_func():
            return DummyIngestControlMessage(payload=None)

        decorated = nv_ingest_source_failure_context_manager("annotation_source", payload_can_be_empty=False)(
            dummy_source_func
        )
        with self.assertRaises(ValueError):
            decorated()

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    @patch(f"{MODULE_UNDER_TEST}.cm_set_failure")
    def test_exception_in_function_sets_failure(self, mock_set_failure, mock_annotate):
        def dummy_source_func():
            raise ValueError("dummy error")

        decorated = nv_ingest_source_failure_context_manager("annotation_source", raise_on_failure=False)(
            dummy_source_func
        )
        result = decorated()
        self.assertIsInstance(result, IngestControlMessage)
        # Expect that both cm_set_failure and annotate_task_result were called.
        mock_set_failure.assert_called_once()
        mock_annotate.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    @patch(f"{MODULE_UNDER_TEST}.cm_set_failure")
    def test_exception_propagates_when_raise_on_failure(self, mock_set_failure, mock_annotate):
        def dummy_source_func():
            raise ValueError("dummy error")

        decorated = nv_ingest_source_failure_context_manager("annotation_source", raise_on_failure=True)(
            dummy_source_func
        )
        with self.assertRaises(ValueError):
            decorated()


##############################################
# Tests for CMNVIngestFailureContextManager
##############################################
class TestCMNVIngestFailureContextManager(unittest.TestCase):

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    def test_context_manager_success(self, mock_annotate):
        cm = DummyIngestControlMessage(payload="data")
        # In a context that does not raise, success should be annotated.
        with CMNVIngestFailureContextManager(cm, "annotation_cm", raise_on_failure=False, func_name="test_func"):
            pass
        mock_annotate.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    @patch(f"{MODULE_UNDER_TEST}.cm_set_failure")
    def test_context_manager_failure_suppresses_exception(self, mock_set_failure, mock_annotate):
        cm = DummyIngestControlMessage(payload="data")
        # When an exception is raised in the block, it should be annotated but suppressed.
        try:
            with CMNVIngestFailureContextManager(cm, "annotation_cm", raise_on_failure=False, func_name="test_func"):
                raise ValueError("test error")
        except Exception:
            self.fail("Exception should have been suppressed")
        mock_set_failure.assert_called_once()
        mock_annotate.assert_called_once()

    @patch(f"{MODULE_UNDER_TEST}.annotate_task_result")
    @patch(f"{MODULE_UNDER_TEST}.cm_set_failure")
    def test_context_manager_failure_raises_exception(self, mock_set_failure, mock_annotate):
        cm = DummyIngestControlMessage(payload="data")
        # When raise_on_failure is True, the exception should propagate.
        with self.assertRaises(ValueError):
            with CMNVIngestFailureContextManager(cm, "annotation_cm", raise_on_failure=True, func_name="test_func"):
                raise ValueError("test error")
        mock_set_failure.assert_called_once()
        mock_annotate.assert_called_once()
