"""Version helpers for build-time and runtime metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
import os
import subprocess

try:
    from ._build_info import BUILD_DATE as _PACKAGE_BUILD_DATE
    from ._build_info import BUILD_GIT_SHA as _PACKAGE_BUILD_GIT_SHA
except ImportError:
    # During setuptools build isolation the package may not be importable
    _PACKAGE_BUILD_DATE = "unknown"
    _PACKAGE_BUILD_GIT_SHA = "unknown"

_PKG_NAME = "retriever"
_UNKNOWN = "unknown"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def _build_datetime() -> datetime:
    if _PACKAGE_BUILD_DATE and _PACKAGE_BUILD_DATE != _UNKNOWN:
        normalized = _PACKAGE_BUILD_DATE.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
    if source_date_epoch and source_date_epoch.isdigit():
        return datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)

    build_date = os.getenv("RETRIEVER_BUILD_DATE")
    if build_date:
        # Accept ISO-8601-like strings from CI and normalize if timezone missing.
        normalized = build_date.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    return _utc_now()


@lru_cache(maxsize=1)
def _run_git_short_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    return out or None


def _is_source_checkout() -> bool:
    package_root = Path(__file__).resolve().parents[2]
    return (package_root / "pyproject.toml").exists()


@lru_cache(maxsize=1)
def _git_sha_short() -> str:
    candidate = (
        os.getenv("RETRIEVER_GIT_SHA")
        or _PACKAGE_BUILD_GIT_SHA
        or os.getenv("GITHUB_SHA")
        or _run_git_short_sha()
        or _UNKNOWN
    )
    return candidate[:7]


@lru_cache(maxsize=1)
def _build_number() -> str:
    # Prefer explicit CI build numbers; fall back to timestamp so local builds
    # naturally produce monotonically increasing dev versions.
    candidate = (
        os.getenv("RETRIEVER_BUILD_NUMBER")
        or os.getenv("GITHUB_RUN_NUMBER")
        or os.getenv("NV_INGEST_REV")
    )
    if candidate and candidate.isdigit():
        return candidate
    return _build_datetime().strftime("%Y%m%d%H%M%S")


@lru_cache(maxsize=1)
def _base_version() -> str:
    return (
        os.getenv("RETRIEVER_VERSION")
        or os.getenv("NV_INGEST_VERSION")
        or _build_datetime().strftime("%Y.%m.%d")
    )


def get_build_version() -> str:
    """Return a PEP 440 compliant version string for packaging."""
    release_type = (
        os.getenv("RETRIEVER_RELEASE_TYPE")
        or os.getenv("NV_INGEST_RELEASE_TYPE")
        or "dev"
    ).lower()

    base_version = _base_version()
    build_number = _build_number()

    if release_type == "release":
        return f"{base_version}.post{build_number}" if int(build_number) > 0 else base_version
    if release_type == "dev":
        return f"{base_version}.dev{build_number}"

    raise ValueError(
        "Invalid release type. Expected one of: dev, release. "
        f"Received: {release_type!r}"
    )


def _installed_version() -> str | None:
    if _is_source_checkout():
        return None
    try:
        return dist_version(_PKG_NAME)
    except PackageNotFoundError:
        return None


def get_version_info() -> dict[str, str]:
    """Return version metadata for runtime diagnostics."""
    version = _installed_version() or get_build_version()
    build_dt = _build_datetime()
    build_date = build_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    build_date_tag = build_dt.strftime("%Y%m%d")
    git_sha = _git_sha_short()

    if git_sha == _UNKNOWN:
        full_version = version
    else:
        # Keep package version PEP 440 clean; expose traceability metadata here.
        full_version = f"{version}+g{git_sha}.d{build_date_tag}"

    return {
        "version": version,
        "git_sha": git_sha,
        "build_date": build_date,
        "full_version": full_version,
    }


def get_version() -> str:
    """Return installed package version string."""
    return get_version_info()["version"]


__version__ = _installed_version() or get_build_version()
