# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
HTTP writer strategy for the NV-Ingest data writer.
"""

import json
from typing import List

# Type imports (will be resolved at runtime)
try:
    from nv_ingest_api.data_handlers.data_writer import HttpDestinationConfig
except ImportError:
    # Handle circular import during development
    HttpDestinationConfig = None


class HttpWriterStrategy:
    """Strategy for writing to HTTP destinations with connection pooling and status-aware error handling."""

    def __init__(self):
        """Initialize HTTP strategy with connection pooling."""
        self._session = None

    def is_available(self) -> bool:
        """Check if requests is available."""
        try:
            import requests

            return requests is not None
        except ImportError:
            return False

    def _get_session(self):
        """Get or create a requests session for connection pooling."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _classify_http_error(self, response):
        """Classify HTTP response errors for appropriate retry behavior."""
        from nv_ingest_api.data_handlers.errors import TransientError, PermanentError

        status_code = response.status_code

        # 4xx Client Errors - generally permanent (don't retry)
        if 400 <= status_code < 500:
            # Special cases that might be retryable
            if status_code in [408, 429]:  # Request Timeout, Too Many Requests
                # Check for Retry-After header
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = int(retry_after)
                        return TransientError(f"HTTP {status_code} with Retry-After: {delay}s")
                    except ValueError:
                        pass
                # Without Retry-After, treat as permanent per policy
                return PermanentError(f"HTTP {status_code} client error")
            else:
                # Other 4xx errors are permanent
                return PermanentError(f"HTTP {status_code} client error")

        # 5xx Server Errors - transient (retry)
        elif 500 <= status_code < 600:
            return TransientError(f"HTTP {status_code} server error")

        # Should not reach here if raise_for_status() was called
        return TransientError(f"HTTP {status_code} unexpected error")

    def write(self, data_payload: List[str], config: HttpDestinationConfig) -> None:
        """Write payloads via HTTP request with connection pooling and smart error handling."""
        if not self.is_available():
            from nv_ingest_api.data_handlers.errors import DependencyError

            raise DependencyError(
                "requests library is not available. Install requests for HTTP destination support: "
                "pip install requests"
            )

        session = self._get_session()

        # Combine payloads
        combined_data = [json.loads(payload) for payload in data_payload]

        headers = config.headers.copy()
        if config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"

        response = session.request(
            method=config.method, url=config.url, json=combined_data, headers=headers, timeout=30
        )

        # Check for HTTP errors and classify them appropriately
        if not response.ok:
            classified_error = self._classify_http_error(response)
            raise classified_error

        # Success - response is valid
        return
