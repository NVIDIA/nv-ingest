#!/usr/bin/env python3
"""
Nightly builder/publisher for Hugging Face-hosted NVIDIA OSS repos.

Behavior:
- Clones a HF git repo with Git LFS smudge disabled (so large weights are not downloaded).
- Patches a PEP 440 dev version into pyproject.toml or setup.cfg (default suffix: UTC
  YYYYMMDD; override with NIGHTLY_DATE_SUFFIX or NIGHTLY_DATE_YYYYMMDD, e.g. from CI).
- Builds sdist + wheel via `python -m build` (``--skip-sdist`` builds the wheel only,
  so a platform/Python matrix can emit the identical sdist exactly once).
- Optional ``--set-requires-python`` relaxes ``[project].requires-python`` so wheels
  built on a Python matrix stay pip-installable on every targeted interpreter.
- Optional ``--auditwheel-repair`` rewrites ``linux_*`` wheels to ``manylinux_*`` for PyPI.
- Optionally uploads to (Test)PyPI via twine.

This is intentionally best-effort across heterogeneous upstream projects.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


_PATCHABLE_TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".pyi",
    ".rst",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_PYTHON_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _canonical_dependency_name(name: str) -> str:
    return re.sub(r"[-_.]", "-", name.strip()).lower()


def _dependency_name(requirement: str) -> str | None:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", requirement)
    if not m:
        return None
    return _canonical_dependency_name(m.group(1))


def _public_distribution_version(version: str) -> str:
    return version.strip().split("+", 1)[0]


def _runtime_dependency_specifier(version: str) -> str:
    return f"~={_public_distribution_version(version)}"


def _nightly_suffix() -> str:
    """Return the numeric part after ``.dev`` in the nightly PEP 440 version.

    Default is UTC ``YYYYMMDD`` so parallel matrix legs (e.g. x86_64 + aarch64) publish
    under the same version. Override with ``NIGHTLY_DATE_SUFFIX`` or
    ``NIGHTLY_DATE_YYYYMMDD`` (e.g. set once per workflow run in CI).
    """
    forced = os.environ.get("NIGHTLY_DATE_SUFFIX") or os.environ.get("NIGHTLY_DATE_YYYYMMDD")
    if forced:
        return forced
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_is_usable(venv_dir: Path) -> bool:
    py = _venv_python(venv_dir)
    if not py.is_file() and not py.is_symlink():
        return False
    try:
        subprocess.run(
            [str(py), "-c", "pass"],
            check=True,
            capture_output=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _ensure_venv(venv_dir: Path, *, system_site_packages: bool) -> Path:
    """
    Ensure a local venv exists and return its python path.

    We intentionally do all pip/build/twine operations inside this venv to avoid
    system-Python restrictions (e.g. PEP 668 externally-managed environments).
    """
    marker = venv_dir / ".orch-system-site-packages"
    py = _venv_python(venv_dir)
    if _venv_is_usable(venv_dir):
        # If caller changes system_site_packages setting, recreate the venv to ensure
        # the correct site-packages visibility.
        has_marker = marker.exists()
        if has_marker == system_site_packages:
            return py
        shutil.rmtree(venv_dir)
    elif venv_dir.exists():
        shutil.rmtree(venv_dir)

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "venv"]
    if system_site_packages:
        cmd.append("--system-site-packages")
    cmd.append(str(venv_dir))
    _run(cmd)
    py = _venv_python(venv_dir)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if system_site_packages:
        marker.write_text("1", encoding="utf-8")
    return py


def _pep440_nightly(base_version: str, suffix: str) -> str:
    """
    Convert a base version to a nightly dev version.
    Examples:
      1.2.3 -> 1.2.3.dev20260127
      1.2.3+local -> 1.2.3.dev20260127
    """
    base = base_version.split("+", 1)[0].strip()
    base = re.sub(r"\.dev\d+$", "", base)
    return f"{base}.dev{suffix}"


def _pep440_stable_release(version: str) -> str:
    version = version.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)*(?:\.post\d+)?", version):
        raise ValueError(
            "--release-version must be a stable public version like '2.0.0' "
            "or '2.0.0.post1' (no dev, pre-release, local, or whitespace suffixes)"
        )
    return version


def _target_version(
    old_version: str,
    *,
    nightly_base_version: str | None = None,
    release_version: str | None = None,
) -> str:
    if release_version is not None:
        return _pep440_stable_release(release_version)
    return _pep440_nightly(
        nightly_base_version or old_version,
        _nightly_suffix(),
    )


def _patch_pyproject_version(
    repo_dir: Path,
    *,
    nightly_base_version: str | None = None,
    release_version: str | None = None,
) -> bool:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    # Match: version = "x.y.z" under [project] (best-effort, not a full TOML parser)
    m = re.search(r'(?ms)^\[project\]\s.*?^\s*version\s*=\s*"([^"]+)"\s*$', text)
    if not m:
        return False

    old_version = m.group(1)
    new_version = _target_version(
        old_version,
        nightly_base_version=nightly_base_version,
        release_version=release_version,
    )
    if new_version == old_version:
        return False

    text2 = text[: m.start(1)] + new_version + text[m.end(1) :]
    _write_text(pyproject, text2)
    print(f"Patched pyproject.toml version: {old_version} -> {new_version}")
    return True


def _patch_pyproject_project_name(repo_dir: Path, project_name: str) -> bool:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    bounds = _project_table_bounds(text)
    if bounds is None:
        return False

    project_start, project_end = bounds
    project_text = text[project_start:project_end]
    m = re.search(r'(?m)^(\s*name\s*=\s*")([^"]+)(")\s*$', project_text)
    if not m:
        return False

    old_name = m.group(2)
    if old_name == project_name:
        return False

    patched_project_text = project_text[: m.start(2)] + project_name + project_text[m.end(2) :]
    patched_text = text[:project_start] + patched_project_text + text[project_end:]
    _write_text(pyproject, patched_text)
    print(f"Patched pyproject.toml project name: {old_name} -> {project_name}")
    return True


def _patch_setup_cfg_project_name(repo_dir: Path, project_name: str) -> bool:
    setup_cfg = repo_dir / "setup.cfg"
    if not setup_cfg.exists():
        return False

    text = _read_text(setup_cfg)
    m = re.search(r"(?ms)^\[metadata\]\s.*?^(\s*name\s*=\s*)([^\s#]+)(\s*(?:#.*)?)$", text)
    if not m:
        return False

    old_name = m.group(2).strip().strip('"').strip("'")
    if old_name == project_name:
        return False

    text2 = text[: m.start(2)] + project_name + text[m.end(2) :]
    _write_text(setup_cfg, text2)
    print(f"Patched setup.cfg project name: {old_name} -> {project_name}")
    return True


def _parse_package_rename(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--rename-python-package must be OLD=NEW, got: {spec!r}")
    old_name, new_name = (part.strip() for part in spec.split("=", 1))
    if not old_name or not new_name:
        raise ValueError(f"--rename-python-package must be OLD=NEW, got: {spec!r}")
    if not _PYTHON_IDENTIFIER_RE.fullmatch(old_name) or not _PYTHON_IDENTIFIER_RE.fullmatch(new_name):
        raise ValueError(f"--rename-python-package values must be Python package names, got: {spec!r}")
    return old_name, new_name


def _iter_patchable_text_files(project_dir: Path) -> list[Path]:
    skip_parts = {".git", ".venv", "__pycache__", "build", "dist"}
    files: list[Path] = []
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        if skip_parts.intersection(path.relative_to(project_dir).parts):
            continue
        if path.suffix in _PATCHABLE_TEXT_SUFFIXES:
            files.append(path)
    return files


def _rename_python_package(project_dir: Path, old_name: str, new_name: str) -> bool:
    if old_name == new_name:
        return False

    old_rel = Path(*old_name.split("."))
    new_rel = Path(*new_name.split("."))
    renamed = False
    for root in (project_dir / "src", project_dir):
        old_path = root / old_rel
        if not old_path.is_dir():
            continue
        new_path = root / new_rel
        if new_path.exists():
            raise RuntimeError(f"Cannot rename Python package: destination already exists: {new_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))
        print(
            "Renamed Python package directory: "
            f"{old_path.relative_to(project_dir)} -> {new_path.relative_to(project_dir)}"
        )
        renamed = True

    if not renamed:
        raise RuntimeError(f"Cannot rename Python package: {old_name!r} was not found under {project_dir}")

    token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(old_name)}(?![A-Za-z0-9_])")
    patched_files = 0
    for path in _iter_patchable_text_files(project_dir):
        try:
            text = _read_text(path)
        except UnicodeDecodeError:
            continue
        patched, count = token.subn(new_name, text)
        if count == 0:
            continue
        _write_text(path, patched)
        patched_files += 1

    print(f"Patched Python package references: {old_name} -> {new_name} in {patched_files} file(s)")
    return True


def _patch_hatch_build_force_platform_wheel(project_dir: Path) -> bool:
    """
    Hatchling may emit py3-none-any for extension builds unless the hook sets
    build_data[\"pure_python\"] = False and build_data[\"infer_tag\"] = True so the
    wheel tag matches the current interpreter/platform. Patch upstream hatch_build.py
    when we recognize the Nemotron OCR layout.
    """
    path = project_dir / "hatch_build.py"
    if not path.is_file():
        return False
    text = _read_text(path)
    has_pp = 'build_data["pure_python"]' in text or "build_data['pure_python']" in text
    has_it = 'build_data["infer_tag"]' in text or "build_data['infer_tag']" in text
    if has_pp and has_it:
        return False
    if "CustomBuildHook" not in text or "def initialize" not in text:
        return False
    needle = "def initialize(self, version: str, build_data: dict) -> None:"
    idx = text.find(needle)
    if idx < 0:
        return False
    body_start = idx + len(needle)
    insert_at = body_start
    while insert_at < len(text) and text[insert_at] in " \t":
        insert_at += 1
    if insert_at < len(text) and text[insert_at] == "\n":
        insert_at += 1

    block = '        build_data["pure_python"] = False\n' '        build_data["infer_tag"] = True\n'

    if not has_pp and not has_it:
        patched = text[:insert_at] + block + text[insert_at:]
        _write_text(path, patched)
        print("Patched hatch_build.py: pure_python=False, infer_tag=True")
        return True

    if has_pp and not has_it:
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if inserted:
                continue
            if ('build_data["pure_python"]' in line or "build_data['pure_python']" in line) and "False" in line:
                out.append('        build_data["infer_tag"] = True\n')
                inserted = True
        if not inserted:
            return False
        _write_text(path, "".join(out))
        print('Patched hatch_build.py: build_data["infer_tag"] = True')
        return True

    # has_it and not has_pp: insert pure_python line before first infer_tag line
    lines = text.splitlines(keepends=True)
    out = []
    inserted = False
    for line in lines:
        if not inserted and ('build_data["infer_tag"]' in line or "build_data['infer_tag']" in line):
            out.append('        build_data["pure_python"] = False\n')
            inserted = True
        out.append(line)
    if not inserted:
        return False
    _write_text(path, "".join(out))
    print('Patched hatch_build.py: build_data["pure_python"] = False')
    return True


def _project_table_bounds(text: str) -> tuple[int, int] | None:
    m = re.search(r"(?m)^\[project\]\s*(?:#.*)?$", text)
    if not m:
        return None

    for next_table in re.finditer(r"(?m)^\[([^\]]+)\]\s*(?:#.*)?$", text[m.end() :]):
        table_name = next_table.group(1).strip()
        if table_name != "project" and not table_name.startswith("project."):
            return m.end(), m.end() + next_table.start()
    return m.end(), len(text)


def _project_core_bounds(text: str) -> tuple[int, int] | None:
    """Bounds of inline ``[project]`` keys only (excludes ``[project.*]`` subtables)."""
    m = re.search(r"(?m)^\[project\]\s*(?:#.*)?$", text)
    if not m:
        return None

    for next_table in re.finditer(r"(?m)^\[([^\]]+)\]\s*(?:#.*)?$", text[m.end() :]):
        table_name = next_table.group(1).strip()
        if table_name != "project":
            return m.end(), m.end() + next_table.start()
    return m.end(), len(text)


def _set_requirement_specifier(requirement: str, specifier: str) -> str:
    marker = ""
    base = requirement
    if ";" in requirement:
        base, marker_part = requirement.split(";", 1)
        marker = ";" + marker_part

    m = re.match(r"(\s*[A-Za-z0-9][A-Za-z0-9_.-]*)(\s*\[[^\]]+\])?", base)
    if not m:
        return requirement
    return f"{m.group(1)}{m.group(2) or ''}{specifier}{marker}"


def _patch_pyproject_runtime_dependency_pins(project_dir: Path, pins: dict[str, str]) -> bool:
    """
    Pin selected [project].dependencies entries to the installed version's minor family.

    This intentionally patches runtime metadata only. Build-system or hatch hook
    dependencies can remain broad because they do not describe the installed wheel's
    runtime compatibility.
    """
    if not pins:
        return False

    pyproject = project_dir / "pyproject.toml"
    if not pyproject.exists():
        raise RuntimeError(f"Cannot pin runtime dependencies: {pyproject} not found")

    normalized_specifiers = {
        _canonical_dependency_name(package): _runtime_dependency_specifier(version) for package, version in pins.items()
    }
    text = _read_text(pyproject)
    bounds = _project_table_bounds(text)
    if bounds is None:
        raise RuntimeError("Cannot pin runtime dependencies: [project] table not found in pyproject.toml")

    project_start, project_end = bounds
    project_text = text[project_start:project_end]
    deps_match = re.search(r"(?ms)^(\s*dependencies\s*=\s*\[)(.*?)(^\s*\])", project_text)
    if not deps_match:
        raise RuntimeError("Cannot pin runtime dependencies: [project].dependencies not found in pyproject.toml")

    found: set[str] = set()

    def replace_dependency(m: re.Match[str]) -> str:
        requirement = m.group("requirement")
        dep_name = _dependency_name(requirement)
        if dep_name not in normalized_specifiers:
            return m.group(0)
        found.add(dep_name)
        pinned = _set_requirement_specifier(requirement, normalized_specifiers[dep_name])
        return f"{m.group('quote')}{pinned}{m.group('quote')}"

    deps_body = deps_match.group(2)
    patched_body = re.sub(
        r"(?P<quote>['\"])(?P<requirement>[^'\"]+)(?P=quote)",
        replace_dependency,
        deps_body,
    )

    missing = sorted(set(normalized_specifiers) - found)
    if missing:
        raise RuntimeError("No matching [project].dependencies entries found for runtime pin(s): " + ", ".join(missing))

    if patched_body == deps_body:
        return False

    patched_project_text = project_text[: deps_match.start(2)] + patched_body + project_text[deps_match.end(2) :]
    patched_text = text[:project_start] + patched_project_text + text[project_end:]
    _write_text(pyproject, patched_text)
    for package in sorted(found):
        print(f"Patched pyproject.toml runtime dependency: {package}{normalized_specifiers[package]}")
    return True


_DEFAULT_LICENSE_TEXT = "Apache-2.0"
_DEFAULT_LICENSE_CLASSIFIER = "License :: OSI Approved :: Apache Software License"
_PROPRIETARY_LICENSE_CLASSIFIER = "License :: Other/Proprietary License"
_ORCHESTRATOR_APACHE_LICENSE = Path(__file__).resolve().parents[2] / "nemo_retriever" / "LICENSE"


def _pyproject_license_line(license_text: str, *, legacy: bool = False) -> str:
    if license_text == "Apache-2.0" and not legacy:
        return 'license = "Apache-2.0"\n'
    return f'license = {{text = "{license_text}"}}\n'


def _patch_pyproject_license(
    project_dir: Path,
    *,
    license_text: str,
    license_classifier: str | None,
) -> bool:
    """
    Ensure PyPI-visible license metadata is present in ``[project]``.

    Published Nemotron Python packages ship source code governed by Apache-2.0.
    Upstream HF repos may omit license metadata or declare the NVIDIA Open Model
    License for the overall model card even though the wheel/sdist is source code.
    """
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    build_backend = tomllib.loads(text).get("build-system", {}).get("build-backend", "")
    use_legacy_license = isinstance(build_backend, str) and build_backend.startswith("setuptools.")
    core_bounds = _project_core_bounds(text)
    if core_bounds is None:
        return False

    core_start, core_end = core_bounds
    project_text = text[core_start:core_end]
    changed = False

    if not re.search(r"(?m)^\s*license\s*=", project_text):
        line = _pyproject_license_line(license_text, legacy=use_legacy_license)
        version_m = re.search(r"(?m)^(\s*version\s*=\s*[^\n]+\n)", project_text)
        if version_m:
            insert_at = version_m.end()
            project_text = project_text[:insert_at] + line + project_text[insert_at:]
        else:
            project_text = "\n" + line + project_text
        changed = True
        print(f"Added pyproject.toml license: {license_text!r}")
    else:
        license_m = re.search(r"(?m)^(\s*license\s*=\s*)(.+)$", project_text)
        if license_m:
            desired = _pyproject_license_line(license_text, legacy=use_legacy_license).strip()
            if license_m.group(0).strip() != desired:
                project_text = project_text[: license_m.start()] + desired + "\n" + project_text[license_m.end() :]
                changed = True
                print(f"Patched pyproject.toml license -> {license_text!r}")

    if license_classifier and f'"{license_classifier}"' not in project_text:
        classifiers_m = re.search(r"(?ms)^(\s*classifiers\s*=\s*\[)(.*?)(^\s*\])", project_text)
        if classifiers_m:
            patched_body = classifiers_m.group(2) + f'    "{license_classifier}",\n'
            project_text = project_text[: classifiers_m.start(2)] + patched_body + project_text[classifiers_m.end(2) :]
            changed = True
            print(f"Added pyproject.toml classifier: {license_classifier!r}")
        else:
            project_text += "\nclassifiers = [\n" f'    "{license_classifier}",\n' "]\n"
            changed = True
            print(f"Added pyproject.toml classifiers with: {license_classifier!r}")

    if license_classifier == _DEFAULT_LICENSE_CLASSIFIER and f'"{_PROPRIETARY_LICENSE_CLASSIFIER}"' in project_text:
        project_text = project_text.replace(f'    "{_PROPRIETARY_LICENSE_CLASSIFIER}",\n', "")
        project_text = project_text.replace(f'"{_PROPRIETARY_LICENSE_CLASSIFIER}",\n', "")
        changed = True
        print(f"Removed pyproject.toml classifier: {_PROPRIETARY_LICENSE_CLASSIFIER!r}")

    if not changed:
        return False

    _write_text(pyproject, text[:core_start] + project_text + text[core_end:])
    return True


def _patch_setup_cfg_license(
    project_dir: Path,
    *,
    license_text: str,
    license_classifier: str | None,
) -> bool:
    setup_cfg = project_dir / "setup.cfg"
    if not setup_cfg.exists():
        return False

    text = _read_text(setup_cfg)
    changed = False

    if not re.search(r"(?ms)^\[metadata\]\s.*?^\s*license\s*=", text):
        metadata_m = re.search(r"(?ms)^(\[metadata\]\s*(?:#.*)?\n)", text)
        if not metadata_m:
            return False
        insert_at = metadata_m.end()
        text = text[:insert_at] + f"license = {license_text}\n" + text[insert_at:]
        changed = True
        print(f"Added setup.cfg license: {license_text!r}")

    if license_classifier and license_classifier not in text:
        classifiers_m = re.search(r"(?ms)^(\s*classifiers\s*=\s*\n)(.*?)(^\S|\Z)", text)
        if classifiers_m:
            text = text[: classifiers_m.end(2)] + f"    {license_classifier}\n" + text[classifiers_m.end(2) :]
        else:
            metadata_m = re.search(r"(?ms)^(\[metadata\]\s*(?:#.*)?\n)", text)
            if not metadata_m:
                return changed
            insert_at = metadata_m.end()
            block = f"classifiers =\n    {license_classifier}\n"
            text = text[:insert_at] + block + text[insert_at:]
        changed = True
        print(f"Added setup.cfg classifier: {license_classifier!r}")

    if not changed:
        return False

    _write_text(setup_cfg, text)
    return True


def _ensure_license_file(project_dir: Path, *, search_roots: list[Path]) -> bool:
    """Bundle the Apache-2.0 LICENSE that governs published Python package source."""
    _ = search_roots  # kept for call-site compatibility; source is orchestrator-local.
    source = _ORCHESTRATOR_APACHE_LICENSE
    if not source.is_file():
        print(f"No Apache-2.0 license source found at {source}; continuing without bundled LICENSE.")
        return False

    dest = project_dir / "LICENSE"
    apache_text = _read_text(source)
    if dest.is_file() and _read_text(dest) == apache_text:
        return False

    _write_text(dest, apache_text)
    print(f"Wrote Apache-2.0 license file into project dir: {source} -> {dest}")
    return True


def _patch_pyproject_requires_python(repo_dir: Path, requires_python: str) -> bool:
    """
    Set or relax the ``[project].requires-python`` constraint before building.

    Upstream packages (e.g. Nemotron OCR) may pin ``requires-python`` to a single
    interpreter (``>=3.12,<3.13``), which makes pip refuse the wheel on other
    interpreters even though the compiled extension is ABI-tagged per interpreter.
    Widening the metadata to the full supported range lets a matrix build publish
    installable wheels for every targeted Python version.
    """
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    bounds = _project_table_bounds(text)
    if bounds is None:
        return False

    project_start, project_end = bounds
    project_text = text[project_start:project_end]
    m = re.search(r"""(?m)^(\s*requires-python\s*=\s*["'])([^"']*)(['"])\s*$""", project_text)
    if m:
        old_value = m.group(2)
        if old_value == requires_python:
            return False
        patched_project_text = project_text[: m.start(2)] + requires_python + project_text[m.end(2) :]
        patched_text = text[:project_start] + patched_project_text + text[project_end:]
        _write_text(pyproject, patched_text)
        print(f"Patched pyproject.toml requires-python: {old_value} -> {requires_python}")
        return True

    # No requires-python field yet: insert one at the top of the [project] table body.
    insertion = f'\nrequires-python = "{requires_python}"'
    patched_text = text[:project_start] + insertion + text[project_start:]
    _write_text(pyproject, patched_text)
    print(f"Added pyproject.toml requires-python: {requires_python}")
    return True


def _patch_setup_cfg_version(
    repo_dir: Path,
    *,
    nightly_base_version: str | None = None,
    release_version: str | None = None,
) -> bool:
    setup_cfg = repo_dir / "setup.cfg"
    if not setup_cfg.exists():
        return False

    text = _read_text(setup_cfg)
    # Match in [metadata]: version = x.y.z
    m = re.search(r"(?ms)^\[metadata\]\s.*?^\s*version\s*=\s*([^\s#]+)\s*$", text)
    if not m:
        return False

    old_version = m.group(1).strip().strip('"').strip("'")
    new_version = _target_version(
        old_version,
        nightly_base_version=nightly_base_version,
        release_version=release_version,
    )
    if new_version == old_version:
        return False

    text2 = text[: m.start(1)] + new_version + text[m.end(1) :]
    _write_text(setup_cfg, text2)
    print(f"Patched setup.cfg version: {old_version} -> {new_version}")
    return True


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clone_repo(repo_url: str, dest: Path) -> None:
    env = os.environ.copy()
    # Do not download large LFS objects.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    _run(["git", "clone", "--depth", "1", repo_url, str(dest)], env=env)


def _looks_like_python_project(project_dir: Path) -> bool:
    return (project_dir / "pyproject.toml").exists() or (project_dir / "setup.cfg").exists()


def _auto_project_subdir(repo_dir: Path, repo_id: str) -> str:
    """
    Best-effort detection for repos that keep the Python project in a nested
    same-named directory (e.g. repo 'llama-nemotron-embed-1b-v2' containing
    'llama-nemotron-embed-1b-v2/pyproject.toml').
    """
    if _looks_like_python_project(repo_dir):
        return ""
    candidate = repo_dir / repo_id
    if candidate.is_dir() and _looks_like_python_project(candidate):
        return repo_id
    return ""


def _apply_build_env_overrides(env: dict[str, str], build_env: list[str]) -> dict[str, str]:
    """
    Apply KEY=VALUE overrides to the environment dict.
    """
    for item in build_env:
        if "=" not in item:
            raise ValueError(f"--build-env must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--build-env must have a key, got: {item!r}")
        env[k] = v
    return env


def _ensure_tmpdir(env: dict[str, str]) -> dict[str, str]:
    """
    Ensure temp files (including `python -m build` isolated envs) land on a
    writable filesystem with enough space.

    In some CI/container setups, `/tmp` can be a small tmpfs, which breaks builds
    for projects that need large build dependencies (e.g. PyTorch wheels).
    """
    tmpdir = env.get("TMPDIR") or os.environ.get("TMPDIR")
    if not tmpdir:
        tmpdir = str(Path.cwd() / ".orch-tmp")
        env["TMPDIR"] = tmpdir
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    return env


def _pip_install(py: Path, packages: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    if not packages:
        return
    _run([str(py), "-m", "pip", "install", "--upgrade", *packages], cwd=cwd, env=env)


def _pyproject_build_system_requires(project_dir: Path) -> list[str]:
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.exists():
        return []

    data = tomllib.loads(_read_text(pyproject))
    requires = data.get("build-system", {}).get("requires", [])
    if not isinstance(requires, list) or not all(isinstance(req, str) for req in requires):
        raise RuntimeError("Invalid pyproject.toml [build-system].requires; expected a list of strings")
    return requires


def _installed_distribution_public_version(
    py: Path,
    package: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    script = "import importlib.metadata as md, sys; print(md.version(sys.argv[1]))"
    cmd = [str(py), "-c", script, package]
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise RuntimeError(f"Unable to read installed distribution version for {package!r}")
    return _public_distribution_version(result.stdout.strip())


def _build(
    project_dir: Path,
    out_dir: Path,
    *,
    build_env: list[str],
    no_isolation: bool,
    venv_system_site_packages: bool,
    venv_pip_install: list[str],
    pin_runtime_dependencies: list[str],
    build_sdist: bool = True,
) -> None:
    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    py = _ensure_venv(venv_dir, system_site_packages=venv_system_site_packages)

    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env = _apply_build_env_overrides(env, build_env)
    env = _ensure_tmpdir(env)

    # Clean build outputs
    for p in (project_dir / "dist", project_dir / "build"):
        if p.exists():
            shutil.rmtree(p)

    _pip_install(py, ["build"], cwd=project_dir, env=env)
    if no_isolation:
        build_system_requires = _pyproject_build_system_requires(project_dir)
        if build_system_requires:
            _pip_install(py, build_system_requires, cwd=project_dir, env=env)
    _pip_install(py, venv_pip_install, cwd=project_dir, env=env)

    if pin_runtime_dependencies:
        pins = {
            package: _installed_distribution_public_version(py, package, cwd=project_dir, env=env)
            for package in pin_runtime_dependencies
        }
        _patch_pyproject_runtime_dependency_pins(project_dir, pins)

    cmd = [str(py), "-m", "build", "--wheel"]
    if build_sdist:
        cmd.insert(3, "--sdist")
    if no_isolation:
        cmd.append("--no-isolation")
    _run(cmd, cwd=project_dir, env=env)

    dist_dir = project_dir / "dist"
    if not dist_dir.exists():
        raise RuntimeError("Build succeeded but dist/ not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    for artifact in dist_dir.iterdir():
        shutil.copy2(artifact, out_dir / artifact.name)


def _auditwheel_repair_dist_dir(dist_dir: Path, *, exclude_libs: list[str] | None = None) -> None:
    """
    Rewrite linux_* wheels to manylinux_* so PyPI accepts the upload.
    Requires ``patchelf`` on PATH (e.g. apt install patchelf).

    *exclude_libs* is a list of shared library basenames (e.g. ``libtorch_cpu.so``)
    that auditwheel should NOT bundle.  This is needed for wheels that link against
    PyTorch: the torch libs are a runtime dependency, not something to vendor.
    """
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        return

    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    py = _ensure_venv(venv_dir, system_site_packages=False)
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env = _ensure_tmpdir(env)
    _pip_install(py, ["auditwheel"], cwd=dist_dir.parent, env=env)

    repair_out = dist_dir / ".auditwheel-out"
    _ensure_clean_dir(repair_out)
    exclude_flags: list[str] = []
    for lib in exclude_libs or []:
        exclude_flags += ["--exclude", lib]
    cmd = [
        str(py),
        "-m",
        "auditwheel",
        "repair",
        *[str(w) for w in wheels],
        "-w",
        str(repair_out),
        *exclude_flags,
    ]
    _run(cmd, env=env)

    repaired = sorted(repair_out.glob("*.whl"))
    if not repaired:
        raise RuntimeError("auditwheel repair produced no wheels")

    for w in wheels:
        w.unlink()

    for rw in repaired:
        dest = dist_dir / rw.name
        shutil.move(str(rw), dest)
        print(f"auditwheel: {dest.name}")

    shutil.rmtree(repair_out)


def _validate_required_wheel_members(dist_dir: Path, required_members: list[str]) -> None:
    """Require exact archive members in every wheel before publication."""
    if not required_members:
        return

    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheels found in {dist_dir} to validate required members")

    invalid_members = [member for member in required_members if not member or member.startswith("/")]
    if invalid_members:
        raise ValueError(f"Required wheel members must be non-empty relative paths: {invalid_members!r}")

    failures: list[str] = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            wheel_members = set(zf.namelist())
        missing = [member for member in required_members if member not in wheel_members]
        if missing:
            failures.append(f"{wheel.name}: missing {', '.join(missing)}")

    if failures:
        raise RuntimeError("Required wheel member validation failed:\n" + "\n".join(failures))


def _validate_wheel_license_metadata(
    dist_dir: Path,
    *,
    license_text: str,
    license_classifier: str | None,
) -> None:
    """Require the configured license metadata in every wheel before publication."""
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheels found in {dist_dir} to validate license metadata")

    failures: list[str] = []
    for wheel in wheels:
        expression: str | None = None
        legacy: str | None = None
        classifiers: list[str] = []
        errors: list[str] = []

        with zipfile.ZipFile(wheel) as zf:
            metadata_names = [name for name in zf.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                errors.append(f"expected exactly one .dist-info/METADATA member, found {len(metadata_names)}")
            else:
                metadata = BytesParser().parsebytes(zf.read(metadata_names[0]))
                expression = metadata.get("License-Expression")
                legacy = metadata.get("License")
                classifiers = metadata.get_all("Classifier", [])

                # PEP 639 metadata is authoritative when present. Fall back to
                # the legacy License field only for older metadata producers.
                observed_license = expression if expression is not None else legacy
                if observed_license != license_text:
                    errors.append(f"expected license {license_text!r}, got {observed_license!r}")
                if license_classifier and license_classifier not in classifiers:
                    errors.append(f"missing classifier {license_classifier!r}")

        if errors:
            failures.append(
                f"{wheel.name}: {'; '.join(errors)} "
                f"(License-Expression={expression!r}, License={legacy!r}, Classifier={classifiers!r})"
            )

    if failures:
        raise RuntimeError("Wheel license metadata validation failed:\n" + "\n".join(failures))


def _twine_upload(
    dist_dir: Path,
    repository_url: str,
    token: str,
    *,
    skip_existing: bool,
    verbose: bool,
) -> None:
    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    # Twine doesn't need system site packages; keep it off by default.
    py = _ensure_venv(venv_dir, system_site_packages=False)

    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    env = _ensure_tmpdir(env)
    _run([str(py), "-m", "pip", "install", "--upgrade", "twine"], env=env)
    cmd = [
        str(py),
        "-m",
        "twine",
        "upload",
        "--non-interactive",
        "--repository-url",
        repository_url,
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if verbose:
        cmd.append("--verbose")
    cmd.append(str(dist_dir / "*"))
    _run(cmd, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="Short identifier for logs/paths")
    ap.add_argument("--repo-url", required=True, help="HF git repo URL, e.g. https://huggingface.co/nvidia/xxx")
    ap.add_argument("--work-dir", default=".work", help="Workspace temp dir")
    ap.add_argument("--dist-dir", default="dist-out", help="Where to collect artifacts")
    ap.add_argument(
        "--project-subdir",
        default="",
        help="If the Python project lives in a subdirectory, build from there (e.g. 'nemotron-ocr')",
    )
    ap.add_argument(
        "--nightly-base-version",
        default=None,
        help="Override the source project version used before appending .dev<suffix> "
        "(e.g. build 1.0.2.devYYYYMMDD from a source tree still declaring 1.0.0).",
    )
    ap.add_argument(
        "--release-version",
        default=None,
        help="Patch the source project to this exact stable public version before building "
        "(e.g. '2.0.0'). Mutually exclusive with --nightly-base-version.",
    )
    ap.add_argument(
        "--project-name",
        default=None,
        help="Patch the source Python project/distribution name before building "
        "(e.g. publish a source tree under an alternate distribution name).",
    )
    ap.add_argument(
        "--set-requires-python",
        default=None,
        help="Set or relax the [project].requires-python constraint before building "
        "(e.g. '>=3.11,<3.14') so wheels built on a Python matrix stay pip-installable.",
    )
    ap.add_argument(
        "--license-text",
        default=_DEFAULT_LICENSE_TEXT,
        help="Ensure [project].license metadata is set to this text when missing (PyPI display).",
    )
    ap.add_argument(
        "--license-classifier",
        default=_DEFAULT_LICENSE_CLASSIFIER,
        help="Ensure this Trove classifier is present when missing.",
    )
    ap.add_argument(
        "--rename-python-package",
        action="append",
        default=[],
        help="Rename a Python import package before building, as OLD=NEW " "(repeatable).",
    )
    ap.add_argument(
        "--build-env",
        action="append",
        default=[],
        help="Extra env vars for the build backend, as KEY=VALUE (repeatable)",
    )
    ap.add_argument(
        "--build-no-isolation",
        action="store_true",
        help="Pass --no-isolation to `python -m build` (useful to reuse preinstalled deps in CI images)",
    )
    ap.add_argument(
        "--skip-sdist",
        action="store_true",
        help="Only build the wheel, not the sdist. Use on all but one leg of a platform/Python "
        "matrix so the identical sdist is produced (and uploaded) exactly once.",
    )
    ap.add_argument(
        "--venv-dir",
        default=".venv-build",
        help="Local venv used for build/twine (default: .venv-build)",
    )
    ap.add_argument(
        "--venv-system-site-packages",
        action="store_true",
        help="Create the build venv with --system-site-packages (to reuse system-installed packages like torch)",
    )
    ap.add_argument(
        "--venv-pip-install",
        action="append",
        default=[],
        help="Extra packages to pip install into the build venv before building (repeatable)",
    )
    ap.add_argument(
        "--pin-runtime-dependency",
        action="append",
        default=[],
        help="Constrain this [project].dependencies package to the installed build-venv version's minor family "
        "before building (repeatable; useful for ABI-coupled deps like torch)",
    )
    ap.add_argument("--upload", action="store_true", help="Upload built dists via twine")
    ap.add_argument("--repository-url", default="https://upload.pypi.org/legacy/", help="Twine repository URL")
    ap.add_argument("--token-env", default="PYPI_API_TOKEN", help="Env var containing API token")
    ap.add_argument("--skip-existing", action="store_true", help="Pass --skip-existing to twine")
    ap.add_argument(
        "--twine-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --verbose to twine upload (default: true; use --no-twine-verbose to silence)",
    )
    ap.add_argument(
        "--hatch-force-platform-wheel",
        action="store_true",
        help="Patch hatch_build.py so hatchling emits a platform-specific wheel (not py3-none-any)",
    )
    ap.add_argument(
        "--auditwheel-repair",
        action="store_true",
        help="Run auditwheel repair on built wheels (manylinux tag; needed for PyPI)",
    )
    ap.add_argument(
        "--auditwheel-exclude",
        action="append",
        default=[],
        help="Shared library to exclude from auditwheel bundling (repeatable). "
        "Use for runtime deps like libtorch_cpu.so that should not be vendored.",
    )
    ap.add_argument(
        "--require-wheel-member",
        action="append",
        default=[],
        help="Exact archive member required in every built wheel before upload (repeatable).",
    )
    args = ap.parse_args()
    if args.release_version is not None and args.nightly_base_version:
        ap.error("--release-version cannot be used with --nightly-base-version")

    root = Path.cwd()
    work_root = root / args.work_dir
    dist_root = root / args.dist_dir
    os.environ["ORCH_VENV_DIR"] = str(root / args.venv_dir)

    repo_dir = work_root / args.repo_id
    _ensure_clean_dir(work_root)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print(f"=== Cloning {args.repo_url} -> {repo_dir} ===")
    _clone_repo(args.repo_url, repo_dir)

    version_mode = "release" if args.release_version is not None else "nightly"
    print(f"=== Attempting {version_mode} version patch ===")
    if not args.project_subdir:
        detected = _auto_project_subdir(repo_dir, args.repo_id)
        if detected:
            args.project_subdir = detected
            print(f"Auto-detected project subdir: {args.project_subdir}")
    project_dir = repo_dir / args.project_subdir if args.project_subdir else repo_dir
    patched = _patch_pyproject_version(
        project_dir,
        nightly_base_version=args.nightly_base_version,
        release_version=args.release_version,
    ) or _patch_setup_cfg_version(
        project_dir,
        nightly_base_version=args.nightly_base_version,
        release_version=args.release_version,
    )
    if not patched:
        print("No static version field found to patch (continuing).")

    if args.set_requires_python:
        if not _patch_pyproject_requires_python(project_dir, args.set_requires_python):
            print(f"requires-python already set to {args.set_requires_python!r} or no [project] table found.")

    if args.license_text:
        print("=== Ensuring PyPI license metadata ===")
        if not _patch_pyproject_license(
            project_dir,
            license_text=args.license_text,
            license_classifier=args.license_classifier or None,
        ):
            _patch_setup_cfg_license(
                project_dir,
                license_text=args.license_text,
                license_classifier=args.license_classifier or None,
            )
        _ensure_license_file(
            project_dir,
            search_roots=[project_dir, project_dir.parent, repo_dir],
        )

    if args.project_name:
        patched_name = _patch_pyproject_project_name(
            project_dir,
            args.project_name,
        ) or _patch_setup_cfg_project_name(
            project_dir,
            args.project_name,
        )
        if not patched_name:
            print(f"Project name already set to {args.project_name!r} or no static project name found.")

    for rename_spec in args.rename_python_package:
        old_name, new_name = _parse_package_rename(rename_spec)
        _rename_python_package(project_dir, old_name, new_name)

    if args.hatch_force_platform_wheel:
        if not _patch_hatch_build_force_platform_wheel(project_dir):
            print("hatch-force-platform-wheel: no applicable hatch_build.py patch applied")

    print("=== Building sdist + wheel ===")
    out_dir = dist_root / args.repo_id
    _ensure_clean_dir(out_dir)
    _build(
        project_dir,
        out_dir,
        build_env=args.build_env,
        no_isolation=args.build_no_isolation,
        venv_system_site_packages=args.venv_system_site_packages,
        venv_pip_install=args.venv_pip_install,
        pin_runtime_dependencies=args.pin_runtime_dependency,
        build_sdist=not args.skip_sdist,
    )
    print(f"Artifacts in: {out_dir}")

    if args.auditwheel_repair:
        print("=== auditwheel repair ===")
        _auditwheel_repair_dist_dir(out_dir, exclude_libs=args.auditwheel_exclude)

    _validate_required_wheel_members(out_dir, args.require_wheel_member)

    if args.license_text:
        _validate_wheel_license_metadata(
            out_dir,
            license_text=args.license_text,
            license_classifier=args.license_classifier or None,
        )

    if args.upload:
        token = os.environ.get(args.token_env)
        if not token:
            raise RuntimeError(f"Missing required env var: {args.token_env}")
        print(f"=== Uploading to {args.repository_url} ===")
        _twine_upload(
            out_dir,
            args.repository_url,
            token,
            skip_existing=args.skip_existing,
            verbose=args.twine_verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
