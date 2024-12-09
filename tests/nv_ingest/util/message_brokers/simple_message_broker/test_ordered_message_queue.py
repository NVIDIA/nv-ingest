# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import threading
from uuid import uuid4

from nv_ingest.util.message_brokers.simple_message_broker.ordered_message_queue import OrderedMessageQueue


@pytest.fixture
def queue():
    """Fixture to create a queue with a fixed max size for tests."""
    return OrderedMessageQueue(maxsize=3)


def test_can_push_empty_queue(queue):
    """Test can_push on an empty queue."""
    assert queue.can_push() is True


def test_can_push_full_queue(queue):
    """Test can_push when the queue is full."""
    queue.push("Message 1")
    queue.push("Message 2")
    queue.push("Message 3")
    assert queue.can_push() is False


def test_can_push_in_flight_not_counted(queue):
    """Test can_push when in-flight messages don't count towards maxsize."""
    queue.push("Message 1")
    transaction_id = str(uuid4())
    queue.pop(transaction_id)
    assert queue.can_push() is True


def test_push(queue):
    """Test push adds messages and maintains order."""
    queue.push("Message 1")
    queue.push("Message 2")
    assert queue.qsize() == 2


def test_push_order(queue):
    """Test that pushed messages maintain insertion order."""
    index1 = queue.push("Message 1")
    index2 = queue.push("Message 2")
    assert index1 < index2


def test_pop(queue):
    """Test pop retrieves messages in order."""
    queue.push("Message 1")
    queue.push("Message 2")
    transaction_id = str(uuid4())
    message = queue.pop(transaction_id)
    assert message == "Message 1"
    assert queue.qsize() == 1


def test_pop_blocks_until_message_available():
    """Test pop blocks until a message is available."""
    queue = OrderedMessageQueue()
    transaction_id = str(uuid4())

    def delayed_push():
        threading.Event().wait(1)  # Wait 1 second before pushing
        queue.push("Delayed Message")

    threading.Thread(target=delayed_push).start()
    message = queue.pop(transaction_id)
    assert message == "Delayed Message"


def test_acknowledge_removes_in_flight(queue):
    """Test acknowledge removes in-flight message."""
    queue.push("Message 1")
    transaction_id = str(uuid4())
    queue.pop(transaction_id)
    assert transaction_id in queue.in_flight
    queue.acknowledge(transaction_id)
    assert transaction_id not in queue.in_flight


def test_return_message(queue):
    """Test return_message re-enqueues a message."""
    queue.push("Message 1")
    transaction_id = str(uuid4())
    message = queue.pop(transaction_id)
    assert message == "Message 1"
    queue.return_message(transaction_id)
    assert queue.qsize() == 1
    assert queue.pop(str(uuid4())) == "Message 1"


def test_qsize(queue):
    """Test qsize returns the correct number of messages in the queue."""
    queue.push("Message 1")
    queue.push("Message 2")
    assert queue.qsize() == 2


def test_empty(queue):
    """Test empty returns True for an empty queue."""
    assert queue.empty() is True
    queue.push("Message 1")
    assert queue.empty() is False


def test_full(queue):
    """Test full returns True when the queue is full."""
    queue.push("Message 1")
    queue.push("Message 2")
    queue.push("Message 3")
    assert queue.full() is True


def test_full_with_in_flight(queue):
    """Test full considers both queued and in-flight messages."""
    queue.push("Message 1")
    transaction_id = str(uuid4())
    queue.pop(transaction_id)
    queue.push("Message 2")
    queue.push("Message 3")
    queue.push("Message 4")
    assert queue.full() is True


def test_return_message_order(queue):
    """Test that return_message retains original message order."""
    queue.push("Message 1")
    queue.push("Message 2")
    transaction_id = str(uuid4())
    queue.pop(transaction_id)
    queue.push("Message 3")
    queue.return_message(transaction_id)

    assert queue.pop(str(uuid4())) == "Message 1"
    assert queue.pop(str(uuid4())) == "Message 2"
    assert queue.pop(str(uuid4())) == "Message 3"
