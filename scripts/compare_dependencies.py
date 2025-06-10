#!/usr/bin/env python3

import argparse
import yaml
import toml
import re
from pathlib import Path

from conda_build.api import render


def normalize_dependency(dep):
    """Normalize a dependency to a name=version or name format for comparison"""
    match = re.match(r"^([a-zA-Z0-9_.\-]+)([><=!\s\d\.\*]*)$", dep.strip())
    if match:
        name = match.group(1).lower()
        version = match.group(2).strip()
        return f"{name} {version}".strip()
    return dep.strip().lower()


def normalize_version_syntax(dep_string):
    """
    Convert Python (PEP 508) version specifiers to Conda-compatible ones.

    Examples:
        "foo >=1.2.3" => "foo >=1.2.3"
        "bar ==1.0.0" => "bar =1.0.0"
        "baz ~=1.4"   => "baz >=1.4,<2.0"
    """
    # Match: name + version specifier
    match = re.match(r"^([a-zA-Z0-9_.\-]+)\s*(.*)$", dep_string.strip())
    if not match:
        return dep_string.strip()

    name, version_spec = match.groups()
    version_spec = version_spec.strip()

    # Convert known operators
    if version_spec.startswith("=="):
        version_spec = "=" + version_spec[2:]
    elif version_spec.startswith("==="):
        version_spec = "=" + version_spec[3:]
    elif version_spec.startswith("~="):
        # Convert ~=1.4 -> >=1.4,<2.0 or ~=1.4.5 -> >=1.4.5,<1.5.0
        base_version = version_spec[2:]
        parts = base_version.split(".")
        if len(parts) >= 2:
            upper = str(int(parts[0])) + "." + str(int(parts[1]) + 1) + ".0"
            version_spec = f">={base_version},<{upper}"
        else:
            version_spec = f">={base_version},<{int(parts[0]) + 1}"
    # Else: keep >=, <=, !=, >, < unchanged

    return f"{name} {version_spec}".strip()


def parse_meta_yaml(meta_path):
    """Uses conda-build to render meta.yaml and extract dependencies."""
    try:
        # render returns a list of (MetaData, other_stuff)
        rendered, *_ = render(meta_path, finalize=False, bypass_env_check=True)
        meta_str = rendered.metadata.extract_requirements_text()
    except Exception as e:
        print(f"Failed to render meta.yaml: {e}")
        return set()

    deps = set()
    try:
        meta = yaml.safe_load(meta_str)
        run_deps = meta["requirements"]["run"]
        deps.update(normalize_dependency(d) for d in run_deps)
    except Exception as e:
        print(f"Error extracting dependencies from rendered meta.yaml: {e}")
    return deps


def parse_pyproject_toml(pyproject_path):
    with open(pyproject_path, "r") as f:
        pyproject = toml.load(f)

    deps = set()
    try:
        if "project" in pyproject:
            deps.update(normalize_dependency(d) for d in pyproject["project"].get("dependencies", []))
        elif "tool" in pyproject and "poetry" in pyproject["tool"]:
            poetry_deps = pyproject["tool"]["poetry"].get("dependencies", {})
            for name, version in poetry_deps.items():
                if name.lower() == "python":
                    continue  # Skip base Python version
                if isinstance(version, str):
                    deps.add(normalize_dependency(f"{name} {version}"))
                elif isinstance(version, dict) and "version" in version:
                    deps.add(normalize_dependency(f"{name} {version['version']}"))
    except Exception as e:
        print(f"Error parsing pyproject.toml dependencies: {e}")
    return deps


def main():
    parser = argparse.ArgumentParser(description="Compare dependencies between meta.yaml and pyproject.toml")
    parser.add_argument("--meta", required=True, help="Path to Conda meta.yaml")
    parser.add_argument("--pyproject", required=True, help="Path to pyproject.toml")

    args = parser.parse_args()

    meta_deps = parse_meta_yaml(Path(args.meta))
    pyproject_deps = parse_pyproject_toml(Path(args.pyproject))
    pyproject_deps = set(normalize_version_syntax(dep) for dep in pyproject_deps)

    only_in_meta = meta_deps - pyproject_deps
    only_in_pyproject = pyproject_deps - meta_deps

    print("\nDependencies only in meta.yaml:")
    for dep in sorted(only_in_meta):
        print(f"  {dep}")

    print("\nDependencies only in pyproject.toml:")
    for dep in sorted(only_in_pyproject):
        print(f"  {dep}")


if __name__ == "__main__":
    main()
