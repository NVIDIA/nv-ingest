# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nv_ingest_api.util.string_processing as module_under_test


def test_remove_url_endpoints_removes_v1_path():
    assert module_under_test.remove_url_endpoints("http://deplot:8000/v1/chat/completions") == "http://deplot:8000"
    assert module_under_test.remove_url_endpoints("https://myapi.com/v1") == "https://myapi.com"
    assert module_under_test.remove_url_endpoints("http://localhost:5000/v1") == "http://localhost:5000"


def test_remove_url_endpoints_no_change_if_no_v1():
    url = "http://deplot:8000/chat"
    assert module_under_test.remove_url_endpoints(url) == url


def test_generate_url_prepends_http_if_missing():
    assert module_under_test.generate_url("deplot:8000") == "http://deplot:8000"
    assert module_under_test.generate_url("localhost:8080") == "http://localhost:8080"


def test_generate_url_keeps_existing_scheme():
    assert module_under_test.generate_url("http://deplot:8000") == "http://deplot:8000"
    assert module_under_test.generate_url("https://secureapi.com") == "https://secureapi.com"
