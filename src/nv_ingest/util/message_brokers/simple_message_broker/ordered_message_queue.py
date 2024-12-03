# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import threading
import heapq


class OrderedMessageQueue:
    def __init__(self, maxsize=0):
        self.queue = []  # List of (index, message) tuples
        self.maxsize = maxsize
        self.next_index = 0  # Monotonically increasing message index
        self.in_flight = {}  # Mapping of transaction_id to (index, message)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def can_push(self):
        """Check if the queue can accept more messages."""
        with self.lock:
            return self.maxsize == 0 or (len(self.queue) + len(self.in_flight)) < self.maxsize

    def push(self, message):
        """Add a message to the queue after it has been acknowledged."""
        with self.lock:
            index = self.next_index
            self.next_index += 1
            heapq.heappush(self.queue, (index, message))
            self.not_empty.notify()
            return index

    def pop(self, transaction_id):
        """Pop a message from the queue and mark it as in-flight."""
        with self.lock:
            while not self.queue:
                self.not_empty.wait()
            index, message = heapq.heappop(self.queue)
            self.in_flight[transaction_id] = (index, message)
            self.not_full.notify()
            return message

    def acknowledge(self, transaction_id):
        """Acknowledge that a message has been processed."""
        with self.lock:
            if transaction_id in self.in_flight:
                del self.in_flight[transaction_id]

    def return_message(self, transaction_id):
        """Return an unacknowledged message back to the queue."""
        with self.lock:
            if transaction_id in self.in_flight:
                index, message = self.in_flight.pop(transaction_id)
                heapq.heappush(self.queue, (index, message))
                self.not_empty.notify()

    def qsize(self):
        """Get the number of messages currently in the queue."""
        with self.lock:
            return len(self.queue)

    def empty(self):
        """Check if the queue is empty."""
        with self.lock:
            return not self.queue

    def full(self):
        """Check if the queue is full."""
        with self.lock:
            return self.maxsize > 0 and (len(self.queue) + len(self.in_flight)) >= self.maxsize
