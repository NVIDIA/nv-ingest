#!/usr/bin/env python3
"""
Nightly builder/publisher for Hugging Face-hosted NVIDIA OSS repos.

Behavior:
- Clones a HF git repo with Git LFS smudge disabled (so large weights are not downloaded).
- Attempts to append a PEP 440 dev version suffix (YYYYMMDD) to pyproject.toml or setup.cfg.
- Builds sdist + wheel via `python -m build`.
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
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _nightly_suffix_yyyymmdd() -> str:
    # Allow overriding for reproducibility.
    forced = os.environ.get("NIGHTLY_DATE_YYYYMMDD")
    if forced:
        return forced
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def _git_rev(repo_path: Path, short: int = 7) -> str:
    """Return short git commit hash for repo_path, or 'unknown' if not a git repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", f"--short={short}", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path, *, system_site_packages: bool) -> Path:
    """
    Ensure a local venv exists and return its python path.

    We intentionally do all pip/build/twine operations inside this venv to avoid
    system-Python restrictions (e.g. PEP 668 externally-managed environments).
    """
    marker = venv_dir / ".orch-system-site-packages"
    py = _venv_python(venv_dir)
    if py.exists():
        # If caller changes system_site_packages setting, recreate the venv to ensure
        # the correct site-packages visibility.
        has_marker = marker.exists()
        if has_marker == system_site_packages:
            return py
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


def _pep440_nightly(base_version: str, yyyymmdd: str) -> str:
    """
    Convert a base version to a nightly dev version.
    Examples:
      1.2.3 -> 1.2.3.dev20260127
      1.2.3+local -> 1.2.3.dev20260127
    """
    base = base_version.split("+", 1)[0].strip()
    # If already has a .dev segment, replace it to keep it monotonic daily.
    base = re.sub(r"\.dev\d+$", "", base)
    return f"{base}.dev{yyyymmdd}"


def _patch_pyproject_version(repo_dir: Path, *, local_version_suffix: str = "") -> bool:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    # Match: version = "x.y.z" under [project] (best-effort, not a full TOML parser)
    m = re.search(r'(?ms)^\[project\]\s.*?^\s*version\s*=\s*"([^"]+)"\s*$', text)
    if not m:
        return False

    old_version = m.group(1)
    new_version = _pep440_nightly(old_version, _nightly_suffix_yyyymmdd())
    if local_version_suffix:
        new_version = f"{new_version}+{local_version_suffix}"
    if new_version == old_version:
        return False

    text2 = text[: m.start(1)] + new_version + text[m.end(1) :]
    _write_text(pyproject, text2)
    print(f"Patched pyproject.toml version: {old_version} -> {new_version}")
    return True


def _patch_setup_cfg_version(repo_dir: Path, *, local_version_suffix: str = "") -> bool:
    setup_cfg = repo_dir / "setup.cfg"
    if not setup_cfg.exists():
        return False

    text = _read_text(setup_cfg)
    # Match in [metadata]: version = x.y.z
    m = re.search(r"(?ms)^\[metadata\]\s.*?^\s*version\s*=\s*([^\s#]+)\s*$", text)
    if not m:
        return False

    old_version = m.group(1).strip().strip('"').strip("'")
    new_version = _pep440_nightly(old_version, _nightly_suffix_yyyymmdd())
    if local_version_suffix:
        new_version = f"{new_version}+{local_version_suffix}"
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


def _build(
    project_dir: Path,
    out_dir: Path,
    *,
    build_env: list[str],
    no_isolation: bool,
    venv_system_site_packages: bool,
    venv_pip_install: list[str],
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
    _pip_install(py, venv_pip_install, cwd=project_dir, env=env)

    cmd = [str(py), "-m", "build", "--sdist", "--wheel"]
    if no_isolation:
        cmd.append("--no-isolation")
    _run(cmd, cwd=project_dir, env=env)

    dist_dir = project_dir / "dist"
    if not dist_dir.exists():
        raise RuntimeError("Build succeeded but dist/ not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    for artifact in dist_dir.iterdir():
        shutil.copy2(artifact, out_dir / artifact.name)


def _twine_upload(dist_dir: Path, repository_url: str, token: str, *, skip_existing: bool) -> None:
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
    ap.add_argument("--upload", action="store_true", help="Upload built dists via twine")
    ap.add_argument("--repository-url", default="https://test.pypi.org/legacy/", help="Twine repository URL")
    ap.add_argument("--token-env", default="TEST_PYPI_API_TOKEN", help="Env var containing API token")
    ap.add_argument("--skip-existing", action="store_true", help="Pass --skip-existing to twine")
    args = ap.parse_args()

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

    # PEP 440 local version: include git hashes so changes to orchestrator or source produce distinct wheels
    orch_hash = _git_rev(root)
    source_hash = _git_rev(repo_dir)
    local_version_suffix = f"orch.{orch_hash}.src.{source_hash}"
    print(f"Version local suffix: +{local_version_suffix}")

    print("=== Attempting nightly version patch ===")
    if not args.project_subdir:
        detected = _auto_project_subdir(repo_dir, args.repo_id)
        if detected:
            args.project_subdir = detected
            print(f"Auto-detected project subdir: {args.project_subdir}")
    project_dir = repo_dir / args.project_subdir if args.project_subdir else repo_dir
    patched = _patch_pyproject_version(
        project_dir, local_version_suffix=local_version_suffix
    ) or _patch_setup_cfg_version(project_dir, local_version_suffix=local_version_suffix)
    if not patched:
        print("No static version field found to patch (continuing).")

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
    )
    print(f"Artifacts in: {out_dir}")

    if args.upload:
        token = os.environ.get(args.token_env)
        if not token:
            raise RuntimeError(f"Missing required env var: {args.token_env}")
        print(f"=== Uploading to {args.repository_url} ===")
        _twine_upload(out_dir, args.repository_url, token, skip_existing=args.skip_existing)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
