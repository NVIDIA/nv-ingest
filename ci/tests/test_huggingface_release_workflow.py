# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_nightly_build_publish_module() -> ModuleType:
    script_path = REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py"
    spec = importlib.util.spec_from_file_location("nightly_build_publish", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nightly_builder_can_patch_exact_release_version_in_pyproject(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0.dev20260520010101"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_version(project_dir, release_version="2.0.0")

    assert 'version = "2.0.0"' in pyproject.read_text(encoding="utf-8")


def test_nightly_builder_can_patch_exact_release_version_in_setup_cfg(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    setup_cfg = project_dir / "setup.cfg"
    setup_cfg.write_text(
        """
[metadata]
name = example
version = 2.0.0.dev20260520010101
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_setup_cfg_version(project_dir, release_version="2.0.0")

    assert "version = 2.0.0" in setup_cfg.read_text(encoding="utf-8")


def test_nightly_builder_relaxes_existing_requires_python(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0"
requires-python = ">=3.12,<3.13"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")

    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in text
    assert ">=3.12,<3.13" not in text


def test_nightly_builder_adds_requires_python_when_missing(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["hatchling"]

[project]
name = "example"
version = "2.0.0"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")

    text = pyproject.read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.14"' in text
    # A no-op re-run must not duplicate the field.
    assert not nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")
    assert text.count("requires-python") == 1


@pytest.mark.parametrize(
    "version",
    ["", "2.0.0a1", "2.0.0rc1", "2.0.0+local", "2.0.0.dev1"],
)
def test_nightly_builder_rejects_non_stable_release_versions(version: str) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()

    with pytest.raises(ValueError, match="--release-version must be a stable public version"):
        nightly_build_publish._pep440_stable_release(version)


def test_nightly_builder_relaxes_single_quoted_requires_python(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "example"
version = "2.0.0"
requires-python = '>=3.12,<3.13'
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_requires_python(project_dir, ">=3.11,<3.14")

    text = pyproject.read_text(encoding="utf-8")
    assert "requires-python = '>=3.11,<3.14'" in text
    assert text.count("requires-python") == 1


def test_nightly_builder_rejects_empty_release_version_with_nightly_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nightly_build_publish.py",
            "--repo-id",
            "example",
            "--repo-url",
            "https://huggingface.co/nvidia/example",
            "--nightly-base-version",
            "2.0.0",
            "--release-version",
            "",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        nightly_build_publish.main()

    assert exc_info.value.code == 2


def test_huggingface_workflow_has_manual_stable_ocr_release_controls() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert "package:" in workflow
    assert "release_type:" in workflow
    assert "release_version:" in workflow
    assert "Stable releases must select a single package" in workflow
    assert "--release-version" in workflow
    assert 'expected_version="${INPUT_RELEASE_VERSION}"' in workflow
    assert "Built wheel metadata does not declare expected version" in workflow


def test_huggingface_non_ocr_nightlies_are_versioned_after_current_stable() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert '--nightly-base-version "${{ matrix.repo.nightly_base_version }}"' in workflow
    assert "id: nemotron-page-elements-v3" in workflow
    assert 'nightly_base_version: "3.0.2"' in workflow
    assert "id: nemotron-table-structure-v1" in workflow
    assert workflow.count('nightly_base_version: "1.0.1"') == 1


def test_huggingface_ocr_builds_and_publishes_wheels_for_all_supported_pythons() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    ocr_job = workflow.split("build_ocr_cuda:", 1)[1]

    # Python matrix drives setup-python so each leg builds an ABI-specific wheel.
    for version, tag in (("3.11", "cp311"), ("3.12", "cp312"), ("3.13", "cp313")):
        assert f'version: "{version}"' in ocr_job
        assert f"tag: {tag}" in ocr_job
    assert 'python-version: "${{ matrix.python.version }}"' in ocr_job

    # Published wheels stay pip-installable on every targeted interpreter.
    assert '--set-requires-python ">=3.11,<3.14"' in ocr_job

    # The identical sdist is emitted (and uploaded) exactly once across the matrix.
    assert "sdist_arg=" in ocr_job
    assert '"${{ matrix.platform.arch }}" == "x86_64" && "${{ matrix.python.version }}" == "3.12"' in ocr_job
    assert "${sdist_arg}" in ocr_job

    # Artifact names stay unique per architecture and Python version.
    assert "dist-${{ matrix.ocr.id }}-${{ matrix.platform.arch }}-${{ matrix.python.tag }}" in ocr_job


def test_nightly_builder_adds_license_metadata_for_ocr_like_pyproject(tmp_path: Path) -> None:
    project_dir = tmp_path / "nemotron-ocr"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "nemotron-ocr"
version = "1.0.0"
description = "Nemoton OCR"
authors = [{ name = "NVIDIA Nemotron" }]
requires-python = ">=3.12,<3.13"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_license(
        project_dir,
        license_text=nightly_build_publish._DEFAULT_LICENSE_TEXT,
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )

    text = pyproject.read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in text
    assert '"License :: OSI Approved :: Apache Software License",' in text
    assert not nightly_build_publish._patch_pyproject_license(
        project_dir,
        license_text=nightly_build_publish._DEFAULT_LICENSE_TEXT,
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )


def test_nightly_builder_writes_orchestrator_apache_license_into_project_subdir(tmp_path: Path) -> None:
    repo_dir = tmp_path / "nemotron-ocr-v2"
    project_dir = repo_dir / "nemotron-ocr"
    project_dir.mkdir(parents=True)
    (repo_dir / "LICENSE").write_text("NVIDIA Open Model License Agreement\n", encoding="utf-8")
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._ensure_license_file(
        project_dir,
        search_roots=[project_dir, project_dir.parent, repo_dir],
    )
    bundled = (project_dir / "LICENSE").read_text(encoding="utf-8")
    assert bundled.startswith("                                 Apache License")
    assert "NVIDIA Open Model License Agreement" not in bundled
    assert not nightly_build_publish._ensure_license_file(
        project_dir,
        search_roots=[project_dir, project_dir.parent, repo_dir],
    )


def test_nightly_builder_replaces_existing_nvidia_open_model_license_metadata(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "nemotron-page-elements-v3"
version = "3.0.1"
license = {text = "NVIDIA Open Model License"}
classifiers = [
    "License :: Other/Proprietary License",
]
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_license(
        project_dir,
        license_text=nightly_build_publish._DEFAULT_LICENSE_TEXT,
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )

    text = pyproject.read_text(encoding="utf-8")
    assert 'license = "Apache-2.0"' in text
    assert "NVIDIA Open Model License" not in text
    assert "License :: OSI Approved :: Apache Software License" in text
    assert "License :: Other/Proprietary License" not in text


def test_nightly_builder_uses_legacy_license_with_setuptools_classifier(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        """
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "example"
version = "1.0.0"
""".lstrip(),
        encoding="utf-8",
    )
    nightly_build_publish = _load_nightly_build_publish_module()

    assert nightly_build_publish._patch_pyproject_license(
        project_dir,
        license_text=nightly_build_publish._DEFAULT_LICENSE_TEXT,
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )

    text = pyproject.read_text(encoding="utf-8")
    assert 'license = {text = "Apache-2.0"}' in text
    assert 'license = "Apache-2.0"' not in text
    assert '"License :: OSI Approved :: Apache Software License"' in text


def test_nightly_builder_leaves_existing_apache_license_metadata_unchanged(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    original = """
[project]
name = "nemotron-page-elements-v3"
version = "3.0.1"
license = "Apache-2.0"
classifiers = [
    "License :: OSI Approved :: Apache Software License",
]
""".lstrip()
    pyproject.write_text(original, encoding="utf-8")
    nightly_build_publish = _load_nightly_build_publish_module()

    assert not nightly_build_publish._patch_pyproject_license(
        project_dir,
        license_text=nightly_build_publish._DEFAULT_LICENSE_TEXT,
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )
    assert pyproject.read_text(encoding="utf-8") == original


def test_huggingface_workflow_leaves_license_validation_to_shared_builder() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    ocr_validation = workflow.split("export EXPECTED_OCR_VERSION", 1)[1]
    assert 'expected_license = "Apache-2.0"' not in ocr_validation
    assert "Built wheel metadata does not declare expected license" not in ocr_validation
    assert "Built wheel metadata does not declare expected version" in ocr_validation
    assert "Built wheel metadata does not declare runtime dependency" in ocr_validation
    assert "Built wheel is missing nemotron_ocr_cpp extension artifact" in ocr_validation


def test_huggingface_ocr_job_prefers_https_apt_mirrors_on_arm() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    ocr_job = workflow.split("build_ocr_cuda:", 1)[1]
    assert "http://ports.ubuntu.com|https://ports.ubuntu.com" in ocr_job
    assert "apt_retry update" in ocr_job


def test_huggingface_nightly_builder_defaults_to_public_pypi() -> None:
    script = (REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py").read_text(encoding="utf-8")

    assert 'default="https://upload.pypi.org/legacy/"' in script
    assert 'default="PYPI_API_TOKEN"' in script


def _write_wheel(path: Path, members: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for member in members:
            zf.writestr(member, "")


def _write_wheel_metadata(
    path: Path,
    *,
    license_expression: str | None = None,
    license_text: str | None = None,
    classifiers: list[str] | None = None,
) -> None:
    headers = ["Metadata-Version: 2.4", "Name: example", "Version: 1.0.0"]
    if license_expression is not None:
        headers.append(f"License-Expression: {license_expression}")
    if license_text is not None:
        headers.append(f"License: {license_text}")
    headers.extend(f"Classifier: {classifier}" for classifier in classifiers or [])
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("example-1.0.0.dist-info/METADATA", "\n".join(headers) + "\n\n")


@pytest.mark.parametrize(
    ("license_expression", "license_text"),
    [("Apache-2.0", None), (None, "Apache-2.0")],
)
def test_nightly_builder_accepts_modern_or_legacy_wheel_license_metadata(
    tmp_path: Path,
    license_expression: str | None,
    license_text: str | None,
) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel_metadata(
        tmp_path / "example.whl",
        license_expression=license_expression,
        license_text=license_text,
        classifiers=[nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER],
    )

    nightly_build_publish._validate_wheel_license_metadata(
        tmp_path,
        license_text="Apache-2.0",
        license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
    )


def test_nightly_builder_prefers_modern_license_expression_over_legacy_field(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel_metadata(
        tmp_path / "example.whl",
        license_expression="MIT",
        license_text="Apache-2.0",
        classifiers=[nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"example\.whl: expected license 'Apache-2\.0', got 'MIT'.*"
            r"License-Expression='MIT'.*License='Apache-2\.0'"
        ),
    ):
        nightly_build_publish._validate_wheel_license_metadata(
            tmp_path,
            license_text="Apache-2.0",
            license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
        )


@pytest.mark.parametrize(("license_expression", "license_text"), [(None, None), ("MIT", None)])
def test_nightly_builder_rejects_missing_or_incorrect_wheel_license_metadata(
    tmp_path: Path,
    license_expression: str | None,
    license_text: str | None,
) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel_metadata(
        tmp_path / "example.whl",
        license_expression=license_expression,
        license_text=license_text,
        classifiers=[nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER],
    )

    with pytest.raises(RuntimeError, match=r"example\.whl: expected license 'Apache-2\.0'"):
        nightly_build_publish._validate_wheel_license_metadata(
            tmp_path,
            license_text="Apache-2.0",
            license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
        )


def test_nightly_builder_rejects_missing_wheel_license_classifier(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel_metadata(tmp_path / "example.whl", license_expression="Apache-2.0")

    with pytest.raises(RuntimeError, match="missing classifier"):
        nightly_build_publish._validate_wheel_license_metadata(
            tmp_path,
            license_text="Apache-2.0",
            license_classifier=nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER,
        )


def test_nightly_builder_validates_license_metadata_in_every_wheel(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    classifier = nightly_build_publish._DEFAULT_LICENSE_CLASSIFIER
    _write_wheel_metadata(
        tmp_path / "passing.whl",
        license_expression="Apache-2.0",
        classifiers=[classifier],
    )
    _write_wheel_metadata(tmp_path / "failing.whl", license_expression="MIT", classifiers=[classifier])

    with pytest.raises(RuntimeError, match=r"failing\.whl: expected license") as exc_info:
        nightly_build_publish._validate_wheel_license_metadata(
            tmp_path,
            license_text="Apache-2.0",
            license_classifier=classifier,
        )
    assert "passing.whl" not in str(exc_info.value)


def test_nightly_builder_validates_required_members_in_every_wheel(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    required_members = [
        "nemotron_ocr/inference/pipeline.py",
        "nemotron_ocr/inference/pipeline_v2.py",
    ]
    _write_wheel(tmp_path / "x86.whl", required_members)
    _write_wheel(tmp_path / "arm.whl", required_members + ["nemotron_ocr_cpp/extension.so"])

    nightly_build_publish._validate_required_wheel_members(tmp_path, required_members)


def test_nightly_builder_reports_wheel_and_missing_required_member(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()
    _write_wheel(tmp_path / "nemotron_ocr.whl", ["nemotron_ocr/inference/pipeline_v2.py"])

    with pytest.raises(RuntimeError, match=r"nemotron_ocr\.whl: missing nemotron_ocr/inference/pipeline\.py"):
        nightly_build_publish._validate_required_wheel_members(
            tmp_path,
            ["nemotron_ocr/inference/pipeline.py", "nemotron_ocr/inference/pipeline_v2.py"],
        )


def test_nightly_builder_skips_wheel_validation_without_requirements(tmp_path: Path) -> None:
    nightly_build_publish = _load_nightly_build_publish_module()

    nightly_build_publish._validate_required_wheel_members(tmp_path, [])


def test_nightly_builder_validates_wheels_before_upload() -> None:
    script = (REPO_ROOT / "ci" / "scripts" / "nightly_build_publish.py").read_text(encoding="utf-8")
    main_source = script.split("def main() -> int:", 1)[1]

    assert main_source.index("_validate_required_wheel_members(") < main_source.index("if args.upload:")
    assert main_source.index("_validate_wheel_license_metadata(") < main_source.index("if args.upload:")


def test_huggingface_ocr_wheels_require_base_and_v2_pipeline_modules() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "huggingface-nightly.yml").read_text(encoding="utf-8")

    assert '--require-wheel-member "nemotron_ocr/inference/pipeline.py"' in workflow
    assert '--require-wheel-member "nemotron_ocr/inference/pipeline_v2.py"' in workflow
