import json
import logging
import threading
import time
import uuid
from typing import Dict

from nv_ingest.util.message_brokers.simple_message_broker import ResponseSchema
from nv_ingest.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest.util.message_brokers.simple_message_broker import SimpleMessageBroker

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Error collection list to log all errors at the end of the sweep
error_log = []


def truncate_response(response_schema: Dict):
    truncated_response = response_schema.get("response", '')
    response_code = response_schema.get("response_code", 0)
    response_reason = response_schema.get("response_reason", "OK")

    if isinstance(truncated_response, str) and len(truncated_response) > 80:
        truncated_response = truncated_response[:80] + "..."

    return f"ResponseSchema(response_code={response_code}, " \
           f"response_reason='{response_reason}', " \
           f"response='{truncated_response}')"


def start_broker_server(host='127.0.0.1', port=63790, max_queue_size=5):
    """Start the SimpleMessageBroker server."""
    broker = SimpleMessageBroker(host, port, max_queue_size)
    broker.broker = broker  # Attach the broker instance to the server to access _process_command
    logger.info(f"SimpleMessageBroker server started on {host}:{port}")

    # Start the server in a separate thread to allow it to run in the background
    server_thread = threading.Thread(target=broker.serve_forever, daemon=True)
    server_thread.start()
    return broker, server_thread


def stop_broker_server(broker: SimpleMessageBroker):
    """Stops the SimpleMessageBroker server."""
    logger.info("Stopping SimpleMessageBroker server...")
    broker.shutdown()
    broker.server_close()
    logger.info("SimpleMessageBroker server stopped.")


# Start SimpleMessageBroker and SimpleClient
broker_instance, broker_thread = start_broker_server()
client = SimpleClient(host="127.0.0.1", port=63790)

# Give the broker some time to start
time.sleep(3)


# Test client function to push messages to a queue
def push_message(queue_name, message, timeout=None):
    try:
        response = client.submit_message(queue_name, message, timeout=timeout)
        logger.debug(f"Pushed message of size {len(message)} | Response: {truncate_response(response.dict())}")
        return response
    except Exception as e:
        error_log.append(f"Error in push_message: {e}")
        return ResponseSchema(response_code=1, response_reason=str(e)).dict()


# Test client function to pop messages from a queue with an optional timeout
def pop_message(queue_name, timeout=None):
    try:
        response = client.fetch_message(queue_name, timeout=timeout)
        logger.debug(f"Popped from {queue_name} | Response: {truncate_response(response.dict())}")

        return response
    except Exception as e:
        error_log.append(f"Error in pop_message: {e}")

        return ResponseSchema(response_code=1, response_reason=str(e)).dict()

    # Helper function to check queue size


def check_queue_size(queue_name, expected_size):
    size_response = client.size(queue_name)

    # Check if the response was successful
    if size_response.response_code != 0:
        error_log.append(f"Error retrieving size for queue {queue_name}: {size_response}")
        return False  # Indicate that the size check failed due to an error

    # Parse the actual size from the response and compare it to the expected size
    try:
        actual_size = int(size_response.response)
    except ValueError:
        error_log.append(f"Invalid response for queue size from {queue_name}: {size_response.response}")
        return False  # Indicate failure due to invalid response format

    # Check if the actual size matches the expected size
    if actual_size != expected_size:
        error_log.append(f"Queue {queue_name} expected size {expected_size}, but got {actual_size}.")
        return False

    return True  # Return True if the actual size matches the expected size

    # JSON Push/Pop Tests for Small, Medium, and Large Payloads


def json_push_pop(queue_name, json_data):
    """Helper function to push and pop JSON data, verifying correctness."""
    push_response = push_message(queue_name, json.dumps(json_data)).dict()
    print(str(push_response))
    if push_response["response_code"] != 0:
        error_log.append(f"Failed to push JSON data to {queue_name}: {push_response}")
        return False
    check_queue_size(queue_name, 1)

    pop_response = pop_message(queue_name).dict()
    print(str(pop_response))
    if pop_response["response_code"] != 0 or json.loads(pop_response["response"]) != json_data:
        error_log.append(f"Failed to pop JSON data from {queue_name}: {pop_response}")
        return False
    check_queue_size(queue_name, 0)
    return True


try:
    # Initialize the SimpleClient
    client = SimpleClient(host="127.0.0.1", port=63790,
                          max_retries=0)  # Setting max_retries=0 for direct error handling

    # Test pushing small messages to the queue
    # logger.debug("Starting push tests")
    # push_message("test_queue", "Hello, world!")
    # check_queue_size("test_queue", 1)
    # push_message("test_queue", "Another message")
    # check_queue_size("test_queue", 2)
    # push_message("test_queue", "Message 3")
    # check_queue_size("test_queue", 3)

    # # Test popping messages from the queue
    # logger.debug("Starting pop tests")
    # pop_message("test_queue")
    # check_queue_size("test_queue", 2)
    # pop_message("test_queue")
    # check_queue_size("test_queue", 1)
    # pop_message("test_queue")
    # check_queue_size("test_queue", 0)

    # # Test blocking push with timeout when queue is full
    # logger.debug("Testing blocking push with timeout on a full queue")
    # for i in range(5):
    #     push_message("test_queue", f"Message {i}")
    # check_queue_size("test_queue", 5)

    # # Queue is now full, so the next push should fail with "Queue is full"
    # timeout_push_response = push_message("test_queue", "Overflow message", timeout=2)
    # if timeout_push_response.response_code == 0:
    #     error_log.append(f"Expected 'Queue is full' error for blocking push, but got success: {timeout_push_response}")
    # elif timeout_push_response.response_reason != "Submit message failed after retries":
    #     error_log.append(f"Unexpected response for timeout push: {timeout_push_response}")

    # # Test blocking pop with timeout when queue is empty
    # logger.debug("Testing blocking pop with timeout on an empty queue")
    # for _ in range(5):  # Empty the queue
    #     pop_message("test_queue")
    # check_queue_size("test_queue", 0)

    # # Queue is now empty, so the next pop should fail with "Queue is empty"
    # timeout_pop_response = pop_message("test_queue", timeout=2)
    # if timeout_pop_response.response_code == 0:
    #     error_log.append("Expected 'Queue is empty' error for blocking pop, but got success.")
    # elif timeout_pop_response.response_reason != "Fetch message failed after retries":
    #     error_log.append(f"Unexpected response for timeout pop: {timeout_pop_response}")

    # Small JSON payload
    test_queue_name = f"test_{uuid.uuid4()}"
    small_json = {"key": "value"}
    logger.debug(f"Testing JSON push/pop with small payload on queue {test_queue_name}")
    if not json_push_pop(test_queue_name, small_json):
        error_log.append(f"Small JSON payload test failed for queue {test_queue_name}")

    # Medium JSON payload (approx 1MB)
    # test_queue_name = f"test_{uuid.uuid4()}"
    # medium_json = {"key": "value" * 100000}  # Repeat 'value' to increase size
    # logger.debug(f"Testing JSON push/pop with medium payload on queue {test_queue_name}")
    # if not json_push_pop(test_queue_name, medium_json):
    #     error_log.append(f"Medium JSON payload test failed for queue {test_queue_name}")

    # Large JSON payload (approx 150MB)
    # test_queue_name = f"test_{uuid.uuid4()}"
    # large_json = {"key": "A" * (150 * 1024 * 1024 // 10)}  # Adjust size to 150MB as JSON
    # logger.debug(f"Testing JSON push/pop with large payload on queue {test_queue_name}")
    # if not json_push_pop(test_queue_name, large_json):
    #     error_log.append(f"Large JSON payload test failed for queue {test_queue_name}")

    #logger.debug("Test script completed.")

finally:
    # Ensure cleanup and error reporting
    if error_log:
        logger.error("Summary of API validation errors:")
        for error in error_log:
            logger.error(error)
    else:
        logger.info("All API validation tests passed successfully.")
