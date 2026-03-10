# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Black box tests for function_inspect.py utilities.
"""

from nv_ingest_api.util.introspection.function_inspect import infer_udf_function_name


class TestInferUdfFunctionName:
    """Test cases for infer_udf_function_name function."""

    def test_inline_function_simple(self):
        """Test simple inline function definition."""
        udf_function = "def my_func(control_message): pass"
        result = infer_udf_function_name(udf_function)
        assert result == "my_func"

    def test_inline_function_with_whitespace(self):
        """Test inline function with various whitespace patterns."""
        test_cases = [
            ("def  my_func  (control_message): pass", "my_func"),
            ("def my_func(control_message): pass", "my_func"),
            ("def my_func (control_message): pass", "my_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_inline_function_with_non_space_whitespace(self):
        """Test inline function with non-space whitespace (should return None)."""
        # The function only accepts "def " with a space, not other whitespace
        test_cases = [
            "def\tmy_func(control_message): pass",
            "def\nmy_func(control_message): pass",
            "def\rmy_func(control_message): pass",
        ]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result is None, f"Failed for: {udf_function}"

    def test_inline_function_complex_names(self):
        """Test inline function with complex but valid function names."""
        test_cases = [
            ("def process_data(control_message): pass", "process_data"),
            ("def _private_func(control_message): pass", "_private_func"),
            ("def func123(control_message): pass", "func123"),
            ("def MyClass_method(control_message): pass", "MyClass_method"),
            ("def func_with_underscores(control_message): pass", "func_with_underscores"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_inline_function_multiline(self):
        """Test inline function with multiline definition."""
        udf_function = """def complex_func(control_message):
    # This is a complex function
    return control_message"""
        result = infer_udf_function_name(udf_function)
        assert result == "complex_func"

    def test_inline_function_with_leading_whitespace(self):
        """Test inline function with leading whitespace."""
        test_cases = [
            ("  def my_func(control_message): pass", "my_func"),
            ("\tdef my_func(control_message): pass", "my_func"),
            ("   \t def my_func(control_message): pass", "my_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_import_path_simple(self):
        """Test simple import path."""
        udf_function = "my_module.my_function"
        result = infer_udf_function_name(udf_function)
        assert result == "my_function"

    def test_import_path_nested(self):
        """Test nested import paths."""
        test_cases = [
            ("my_module.submodule.process_data", "process_data"),
            ("package.subpackage.module.function", "function"),
            ("a.b.c.d.e.final_func", "final_func"),
            ("utils.data_processing.transform", "transform"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_import_path_with_underscores(self):
        """Test import paths with underscores."""
        test_cases = [
            ("my_module.my_function", "my_function"),
            ("data_utils.process_data", "process_data"),
            ("_private_module._private_func", "_private_func"),
            ("module_123.func_456", "func_456"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_import_path_with_whitespace(self):
        """Test import paths with leading/trailing whitespace."""
        test_cases = [
            ("  my_module.my_function  ", "my_function"),
            ("\tmy_module.my_function\t", "my_function"),
            ("   my_module.my_function   ", "my_function"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_unix_style(self):
        """Test Unix-style file paths with function names."""
        test_cases = [
            ("/path/to/file.py:function_name", "function_name"),
            ("/home/user/scripts/my_script.py:process_data", "process_data"),
            ("/absolute/path/to/module.py:my_func", "my_func"),
            ("./relative/path/script.py:local_func", "local_func"),
            ("../parent/dir/file.py:parent_func", "parent_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_windows_style(self):
        """Test Windows-style file paths with function names."""
        test_cases = [
            ("C:\\path\\to\\file.py:function_name", "function_name"),
            ("D:\\Users\\user\\scripts\\my_script.py:process_data", "process_data"),
            ("\\\\server\\share\\module.py:network_func", "network_func"),
            (".\\relative\\path\\script.py:local_func", "local_func"),
            ("..\\parent\\dir\\file.py:parent_func", "parent_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_mixed_separators(self):
        """Test file paths with mixed path separators."""
        test_cases = [
            ("/path\\to/file.py:function_name", "function_name"),
            ("C:/Windows\\System32/script.py:mixed_func", "mixed_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_with_whitespace(self):
        """Test file paths with whitespace."""
        test_cases = [
            ("  /path/to/file.py:function_name  ", "function_name"),
            ("\t/path/to/file.py:function_name\t", "function_name"),
            ("/path/to/file.py: function_name ", "function_name"),
            ("/path/to/file.py:  function_name", "function_name"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_complex_function_names(self):
        """Test file paths with complex function names."""
        test_cases = [
            ("/path/to/file.py:_private_func", "_private_func"),
            ("/path/to/file.py:func123", "func123"),
            ("/path/to/file.py:MyClass_method", "MyClass_method"),
            ("/path/to/file.py:func_with_underscores", "func_with_underscores"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_file_path_multiple_colons(self):
        """Test file paths with multiple colons (should use last one)."""
        test_cases = [
            ("C:\\path:with:colons\\file.py:function_name", "function_name"),
            ("/path:with:colons/file.py:my_func", "my_func"),
            # Note: this case has no path separators, so it's treated as import path
            ("file:with:many:colons.py:final_func", "py:final_func"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_edge_case_empty_string(self):
        """Test empty string input."""
        result = infer_udf_function_name("")
        assert result is None

    def test_edge_case_whitespace_only(self):
        """Test whitespace-only input."""
        test_cases = ["   ", "\t\t", "\n\n", "  \t  \n  "]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result is None, f"Failed for: {repr(udf_function)}"

    def test_edge_case_file_path_without_function(self):
        """Test file paths without explicit function names."""
        test_cases = [
            # These are treated as import paths due to dots, so they return 'py'
            ("/path/to/file.py", "py"),
            ("./relative/path.py", "py"),
            ("../parent/file.py", "py"),
            ("simple_file.py", "py"),
            # Windows path with colon is treated as file path, returns everything after last colon
            ("C:\\path\\to\\file.py", "\\path\\to\\file.py"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_edge_case_malformed_inline_function(self):
        """Test malformed inline function definitions."""
        test_cases = [
            "def",
            "def ",
            "def (",
            "def (control_message):",
            "def-invalid(control_message):",
            "define my_func(control_message):",
            "def my_func",
        ]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result is None, f"Failed for: {udf_function}"

    def test_edge_case_invalid_python_identifiers_but_regex_matches(self):
        """Test cases where regex matches but result is invalid Python identifier."""
        # The regex \w+ matches these, even though they're invalid Python identifiers
        test_cases = [
            ("def 123invalid(control_message):", "123invalid"),
            ("def my_func(", "my_func"),  # Incomplete but matches pattern
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_edge_case_single_word(self):
        """Test single word inputs that don't match any format."""
        test_cases = [
            "function_name",
            "my_func",
            "process",
            "123",
            "_private",
        ]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result is None, f"Failed for: {udf_function}"

    def test_edge_case_colon_without_path(self):
        """Test colon usage without file path context."""
        test_cases = [
            "module:function",
            "simple:name",
            ":function_name",
            "function_name:",
            ":",
        ]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            # These should be treated as import paths since they don't have path separators
            if "." not in udf_function and not udf_function.startswith("def "):
                assert result is None, f"Failed for: {udf_function}"

    def test_edge_case_dot_without_import_context(self):
        """Test dot usage that starts with 'def' (should not be treated as import)."""
        test_cases = [
            "def module.function(control_message): pass",
        ]
        for udf_function in test_cases:
            result = infer_udf_function_name(udf_function)
            # This should be treated as inline function, but the regex won't match due to dot
            assert result is None, f"Failed for: {udf_function}"

    def test_edge_case_mixed_formats(self):
        """Test inputs that could match multiple formats (priority testing)."""
        # File path should take precedence over import path
        result = infer_udf_function_name("/path/to/module.submodule.py:function_name")
        assert result == "function_name"

        # Import path should take precedence over malformed inline function
        result = infer_udf_function_name("def.module.function")
        assert result == "function"

    def test_boundary_conditions(self):
        """Test boundary conditions and special characters."""
        test_cases = [
            # Very long function names
            ("def " + "a" * 100 + "(control_message): pass", "a" * 100),
            ("module." + "b" * 100, "b" * 100),
            ("/path/file.py:" + "c" * 100, "c" * 100),
            # Function names with numbers
            ("def func123(control_message): pass", "func123"),
            ("module.func456", "func456"),
            ("/path/file.py:func789", "func789"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        test_cases = [
            # These cases show the function doesn't validate Python identifiers
            ("def func-name(control_message): pass", None),  # Regex \w+ won't match "func-name"
            ("def func@name(control_message): pass", None),  # Regex \w+ won't match "func@name"
            ("module.func-name", "func-name"),  # Import path: takes last part after dot
            ("/path/file.py:func-name", "func-name"),  # File path: takes last part after colon
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_case_sensitivity(self):
        """Test case sensitivity in function names."""
        test_cases = [
            ("def MyFunc(control_message): pass", "MyFunc"),
            ("def UPPERCASE_FUNC(control_message): pass", "UPPERCASE_FUNC"),
            ("module.CamelCase", "CamelCase"),
            ("/path/file.py:MixedCase", "MixedCase"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"

    def test_realistic_examples(self):
        """Test realistic usage examples."""
        test_cases = [
            # Realistic inline functions
            ("def add_metadata(control_message):\n    return control_message", "add_metadata"),
            ("def process_document(control_message): pass", "process_document"),
            # Realistic import paths
            ("data.processors.text_processor", "text_processor"),
            ("utils.document_utils.extract_metadata", "extract_metadata"),
            ("my_project.custom_udfs.sentiment_analysis", "sentiment_analysis"),
            # Realistic file paths
            ("/home/user/udfs/custom_processor.py:process_data", "process_data"),
            ("./scripts/data_enrichment.py:enrich_document", "enrich_document"),
            ("C:\\Projects\\MyProject\\udfs\\analyzer.py:analyze_content", "analyze_content"),
        ]
        for udf_function, expected in test_cases:
            result = infer_udf_function_name(udf_function)
            assert result == expected, f"Failed for: {udf_function}"
