import threading
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from nv_ingest.util.message_brokers.simple_message_broker import SimpleMessageBroker

max_queue_size = 100  # Define the maximum queue size
server_host = '0.0.0.0'  # Bind to all network interfaces
server_port = 63790  # Port to listen on

# Obtain the singleton instance
server = SimpleMessageBroker(server_host, server_port, max_queue_size)

try:
    # Start the server if not already running
    if not hasattr(server, 'server_thread') or not server.server_thread.is_alive():
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True  # Allows program to exit even if thread is running
        server.server_thread = server_thread  # Attach the thread to the server instance
        server_thread.start()
        logger.info(f"Started SimpleMessageBroker server on {server_host}:{server_port}")
    else:
        logger.info(f"SimpleMessageBroker server already running on {server_host}:{server_port}")

    # Keep the script running until interrupted
    logger.info("Press Ctrl+C to stop the server.")
    while True:
        threading.Event().wait(1)  # Sleep to keep the main thread alive

except KeyboardInterrupt:
    logger.info("Shutting down the SimpleMessageBroker server.")
    server.shutdown()
    server.server_close()
    server.server_thread.join()
    logger.info("Server successfully stopped.")