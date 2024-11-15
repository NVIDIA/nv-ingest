import pytest
import threading
import json
import time
from uuid import uuid4

from nv_ingest.util.message_brokers.simple_message_broker import SimpleClient, SimpleMessageBroker

HOST = '127.0.0.1'
PORT = 9999  # Use an available port
MAX_QUEUE_SIZE = 10


@pytest.fixture(scope='module')
def broker_server():
    """Fixture to start and stop the SimpleMessageBroker server."""
    server = SimpleMessageBroker(HOST, PORT, MAX_QUEUE_SIZE)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True  # Allows program to exit even if thread is running
    server_thread.start()
    time.sleep(1)  # Give the server a moment to start
    yield
    server.shutdown()
    server.server_close()
    server_thread.join()


@pytest.fixture
def client():
    """Fixture to provide a SimpleClient instance."""
    return SimpleClient(HOST, PORT)


@pytest.mark.usefixtures("broker_server")
def test_push_success(client):
    """Test successful PUSH operation."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    response = client.submit_message(queue_name, message)
    assert response.response_code == 0
    assert response.response == "Data stored."
    assert response.transaction_id is not None


@pytest.mark.usefixtures("broker_server")
def test_pop_success(client):
    """Test successful POP operation."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Push a message first
    push_response = client.submit_message(queue_name, message)
    assert push_response.response_code == 0

    # Pop the message
    pop_response = client.fetch_message(queue_name)
    assert pop_response.response_code == 0
    assert pop_response.response == message
    assert pop_response.transaction_id is not None


@pytest.mark.usefixtures("broker_server")
def test_size_command(client):
    """Test SIZE command."""
    queue_name = f"test_queue_{uuid4()}"
    messages = ["Message 1", "Message 2", "Message 3"]

    # Push messages
    for msg in messages:
        response = client.submit_message(queue_name, msg)
        assert response.response_code == 0

    # Check size
    size_response = client.size(queue_name)
    assert size_response.response_code == 0
    assert size_response.response == str(len(messages))


@pytest.mark.usefixtures("broker_server")
def test_push_large_message(client):
    """Test pushing a large message."""
    queue_name = f"test_queue_{uuid4()}"
    message = "A" * (5 * 1024 * 1024)  # 5MB message

    response = client.submit_message(queue_name, message)
    assert response.response_code == 0
    assert response.response == "Data stored."
    assert response.transaction_id is not None


@pytest.mark.usefixtures("broker_server")
def test_pop_empty_queue(client):
    """Test popping from an empty queue."""
    queue_name = f"test_queue_{uuid4()}"

    response = client.fetch_message(queue_name)
    assert response.response_code == 1
    assert response.response_reason == "Queue is empty"


@pytest.mark.usefixtures("broker_server")
def test_push_with_timeout(client):
    """Test PUSH operation with a timeout."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    response = client.submit_message(queue_name, message, timeout=5)
    assert response.response_code == 0
    assert response.response == "Data stored."
    assert response.transaction_id is not None


@pytest.mark.usefixtures("broker_server")
def test_pop_with_timeout(client):
    """Test POP operation with a timeout."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Push a message first
    push_response = client.submit_message(queue_name, message)
    assert push_response.response_code == 0

    # Pop the message with a timeout
    pop_response = client.fetch_message(queue_name, timeout=5)
    assert pop_response.response_code == 0
    assert pop_response.response == message
    assert pop_response.transaction_id is not None


@pytest.mark.usefixtures("broker_server")
def test_push_error_handling(client):
    """Test error handling during PUSH."""
    queue_name = f"test_queue_{uuid4()}"
    message = "Test Message"

    # Simulate connection error by closing the client's socket
    client._socket = None

    response = client.submit_message(queue_name, message)
    # The client should handle the error and retry
    assert response.response_code == 0
    assert response.response == "Data stored."


@pytest.mark.usefixtures("broker_server")
def test_pop_error_handling(client):
    """Test error handling during POP."""
    queue_name = f"test_queue_{uuid4()}"

    # Simulate connection error by closing the client's socket
    client._socket = None

    response = client.fetch_message(queue_name)
    # The client should handle the error and return an appropriate response
    assert response.response_code == 1
    assert response.response_reason == "Queue is empty"


@pytest.mark.usefixtures("broker_server")
def test_invalid_command(client):
    """Test sending an invalid command."""
    data = json.dumps({"command": "INVALID", "queue_name": "test_queue"}).encode('utf-8')
    response = client._execute_command(data)
    assert response.response_code == 1
    assert response.response_reason == "Unknown command"


@pytest.mark.usefixtures("broker_server")
def test_multiple_clients(client):
    """Test multiple clients interacting with the server."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(5)]
    clients = [SimpleClient(HOST, PORT) for _ in range(5)]

    # Each client pushes a message
    for i, c in enumerate(clients):
        response = c.submit_message(queue_name, messages[i])
        assert response.response_code == 0

    # Each client pops a message
    popped_messages = []
    for c in clients:
        response = c.fetch_message(queue_name)
        assert response.response_code == 0
        popped_messages.append(response.response)

    # Verify that all messages were popped
    assert sorted(popped_messages) == sorted(messages)


@pytest.mark.usefixtures("broker_server")
def test_concurrent_push_pop(client):
    """Test concurrent PUSH and POP operations."""
    queue_name = f"test_queue_{uuid4()}"
    messages = [f"Message {i}" for i in range(10)]
    results = []
    lock = threading.Lock()

    def push(msg):
        response = client.submit_message(queue_name, msg)
        with lock:
            results.append(response)

    def pop():
        response = client.fetch_message(queue_name)
        with lock:
            results.append(response)

    threads = []
    for msg in messages:
        t_push = threading.Thread(target=push, args=(msg,))
        t_pop = threading.Thread(target=pop)
        threads.extend([t_push, t_pop])

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify that all messages were pushed and popped
    pushed = [r for r in results if r.response == "Data stored."]
    popped = [r for r in results if r.response_code == 0 and r.response.startswith("Message")]
    assert len(pushed) == len(messages)
    assert len(popped) == len(messages)
