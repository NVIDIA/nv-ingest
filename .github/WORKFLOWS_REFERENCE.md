# GitHub Actions Workflows Reference

Complete reference documentation for nv-ingest GitHub Actions workflows.

## Table of Contents
1. [Workflow Overview](#workflow-overview)
2. [Continuous Integration](#continuous-integration)
3. [Nightly Builds](#nightly-builds)
4. [Release Workflows](#release-workflows)
5. [Reusable Workflows](#reusable-workflows)
6. [Composite Actions](#composite-actions)

---

## Workflow Overview

### Trigger Summary

| Workflow | PR | Main | Schedule | Manual | Branch Create |
|----------|-----|------|----------|--------|---------------|
| `ci-pull-request.yml` | ✓ | | | | |
| `ci-main.yml` | | ✓ | | | |
| `scheduled-nightly.yml` | | ✓ | 23:30 UTC | ✓ | |
| `release-docker.yml` | | | | ✓ | release/* |
| `release-conda.yml` | | | | ✓ | release/* |
| `release-pypi.yml` | | | | ✓ | release/* |
| `docs-deploy.yml` | | ✓ | | ✓ | |

---

## Continuous Integration

### `ci-pull-request.yml`

**Purpose**: Validates pull requests before merge

**Triggers**:
- Pull request (opened, synchronize, reopened)
- Pull request target (for external contributors)

**Jobs**:

#### 1. `pre-commit`
- **Runs**: Pre-commit hooks (linting, formatting)
- **Runner**: `ubuntu-latest`
- **Fast fail**: Yes (runs first)

#### 2. `docker-build-test`
- **Runs**: Docker build for x86_64
- **Platform**: `linux/amd64`
- **Image tag**: `nv-ingest:pr-{number}`
- **Push**: No (local only)
- **Base**: Ubuntu Jammy (public)

#### 3. `docker-test`
- **Runs**: Full pytest suite
- **Coverage**: Enabled
- **Markers**: Excludes integration tests
- **Artifacts**: Coverage reports

#### 4. `library-mode-build` + `library-mode-test`
- **Runs**: Integration tests (conda-based)
- **Requires**: Approval for external contributors
- **Access Control**: 
  - Auto-runs for MEMBER/COLLABORATOR/OWNER
  - Requires `ok-to-test` label for others
- **Timeout**: 60 minutes
- **Dependencies**: Multiple NVIDIA NIMs

**Status Checks**: 
- Required: `pre-commit`, `docker-test`
- Optional: `library-mode-test`

---

### `ci-main.yml`

**Purpose**: Validates main branch commits and tests multi-platform builds

**Triggers**:
- Push to `main` branch

**Jobs**:

#### 1. `pre-commit`
- Same as PR workflow

#### 2. `docker-build` + `docker-test`
- **Platform**: `linux/amd64`
- **Full coverage**: Yes
- **Image tag**: `nv-ingest:main-{sha}`

#### 3. `docker-build-arm` + `docker-test-arm`
- **Platform**: `linux/arm64`
- **Emulation**: QEMU
- **Testing**: Random 100 tests (faster)
- **Non-blocking**: Runs in parallel

#### 4. `library-mode-build` + `library-mode-test`
- **Always runs**: No approval needed (trusted branch)
- Full integration test suite

**Parallelization**:
- ARM and x86 builds run in parallel
- Tests run after respective builds complete

---

## Nightly Builds

### `scheduled-nightly.yml`

**Purpose**: Automated nightly builds and publishing

**Triggers**:
- Schedule: Daily at 23:30 UTC
- Push to `main` (optional)
- Manual dispatch with skip options

**Manual Inputs**:
- `skip-docker`: Skip Docker build/publish
- `skip-conda`: Skip Conda build/publish
- `skip-pypi`: Skip PyPI build/publish

**Jobs**:

#### 1. `determine-version`
- Generates version from date: `YYYY.MM.DD`
- Used by all downstream jobs

#### 2. `docker-build-publish`
- **Platforms**: `linux/amd64,linux/arm64`
- **Registry**: NGC (`nvcr.io`)
- **Tag**: `nv-ingest:YYYY.MM.DD`
- **Push**: Yes
- **Multi-platform**: Yes (buildx)

#### 3. `conda-build` + `conda-publish`
- **Channel**: `dev`
- **Version**: Date-based (YYYY.MM.DD)
- **Packages**: All nv-ingest conda packages
- **Force upload**: Yes

#### 4. `pypi-build` + `pypi-publish`
- **Release type**: `dev`
- **Version**: Date-based
- **Packages**: 
  - `nv-ingest-api`
  - `nv-ingest-client`
  - `nv-ingest` (service)
- **Repository**: Artifactory

**Dependencies**: All jobs independent (run in parallel after version determination)

---

## Release Workflows

### `release-docker.yml`

**Purpose**: Publish official Docker release images

**Triggers**:
- Automatic: Branch creation matching `release/*`
- Manual: workflow_dispatch

**Manual Inputs**:
- `version`: Version string (e.g., `25.4.0`)
- `source-ref`: Git ref to build from (default: `main`)

**Version Determination**:
- **Automatic**: Extracted from branch name (`release/25.4.0` → `25.4.0`)
- **Manual**: Uses input version

**Build Details**:
- **Platforms**: `linux/amd64,linux/arm64`
- **Registry**: NGC
- **Tag**: `nv-ingest:{version}`
- **Example**: `nv-ingest:25.4.0`

**Usage Examples**:
```bash
# Automatic trigger
git checkout -b release/25.4.0
git push origin release/25.4.0

# Manual trigger (GitHub UI)
Actions → Release - Docker → Run workflow
  Version: 25.4.0
  Source ref: main
```

---

### `release-conda.yml`

**Purpose**: Publish conda packages to RapidsAI channels

**Triggers**:
- Automatic: Branch creation matching `release/*`
- Manual: workflow_dispatch

**Optional Inputs** (manual dispatch):
- `version`: Version string (default: extracted from branch name)
- `channel`: Target channel (default: `main`, options: `dev` or `main`)
- `source-ref`: Git ref to build from (default: release branch or `main`)

**Version Determination**:
- **Automatic**: Extracted from branch name (`release/25.4.0` → `25.4.0`)
- **Manual**: Uses input version or falls back to branch extraction

**Build Details**:
- **Container**: `rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.12`
- **Packages**: All nv-ingest conda packages
- **Force upload**: Yes (overwrites existing)
- **Default channel**: `main` (for release branches)

**Usage Examples**:
```bash
# Automatic trigger (recommended)
git checkout -b release/25.4.0
git push origin release/25.4.0
# → Publishes to main channel automatically

# Manual trigger (for custom options)
Actions → Release - Conda → Run workflow
  Version: 25.4.0
  Channel: dev (for testing) or main
  Source ref: main
```

**Channels**:
- `dev`: Development/testing releases
- `main`: Production releases (default for release branches)

---

### `release-pypi.yml`

**Purpose**: Publish Python wheels to PyPI/Artifactory

**Triggers**:
- Automatic: Branch creation matching `release/*`
- Manual: workflow_dispatch

**Optional Inputs** (manual dispatch):
- `version`: Version string (default: extracted from branch name)
- `release-type`: Type (default: `release`, options: `dev` or `release`)
- `source-ref`: Git ref to build from (default: release branch or `main`)

**Version Determination**:
- **Automatic**: Extracted from branch name (`release/25.4.0` → `25.4.0`)
- **Manual**: Uses input version or falls back to branch extraction

**Build Details**:
- **Container**: `rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.12`
- **Packages built**:
  - `nv-ingest-api` (from `api/`)
  - `nv-ingest-client` (from `client/`)
  - `nv-ingest` (from `src/`)
- **Artifacts**: Wheels (.whl) and source distributions (.tar.gz)
- **Default release type**: `release` (for release branches)

**Usage Examples**:
```bash
# Automatic trigger (recommended)
git checkout -b release/25.4.0
git push origin release/25.4.0
# → Publishes as release type automatically

# Manual trigger (for custom options)
Actions → Release - PyPI → Run workflow
  Version: 25.4.0
  Release type: dev (for testing) or release
  Source ref: main
```

**Release Types**:
- `dev`: Development releases (with dev suffix)
- `release`: Production releases (default for release branches)

---

### `docs-deploy.yml`

**Purpose**: Build and deploy documentation to GitHub Pages

**Triggers**:
- Push to `main`
- Manual dispatch

**Process**:
1. Build docs Docker image (target: `docs`)
2. Run container to generate static site
3. Extract generated site from container
4. Deploy to GitHub Pages

**Output**: https://{org}.github.io/{repo}/

**Permissions**:
- `contents: read`
- `pages: write`
- `id-token: write`

**Concurrency**: Single deployment (no cancellation)

---

## Reusable Workflows

### `reusable-docker-build.yml`

**Purpose**: Reusable Docker image build logic

**Inputs**:
- `platform`: Target platform(s) (default: `linux/amd64`)
- `target`: Docker build stage (default: `runtime`)
- `push`: Push to registry (default: `false`)
- `tags`: Image tags, comma-separated
- `base-image`: Base image name (default: `ubuntu`)
- `base-image-tag`: Base image tag
- `runner`: GitHub runner (default: `linux-large-disk`)
- `use-qemu`: Enable QEMU for cross-platform
- `registry`: Docker registry URL (optional)

**Secrets**:
- `HF_ACCESS_TOKEN`: Hugging Face token
- `DOCKER_PASSWORD`: Registry password

**Outputs**:
- `image-digest`: Built image digest

**Features**:
- Automatic buildx setup for multi-platform
- Conditional QEMU setup
- Flexible tag support
- Registry login (if push enabled)

---

### `reusable-docker-test.yml`

**Purpose**: Run tests in Docker containers

**Inputs**:
- `image-tag`: Docker image to test
- `platform`: Platform to test on
- `test-selection`: `full`, `random`, or marker-based
- `random-count`: Number of random tests
- `pytest-markers`: Pytest marker expression
- `coverage`: Enable coverage report
- `runner`: GitHub runner

**Artifacts**:
- Coverage reports (if enabled)
- Test reports (always)

**Usage Example**:
```yaml
test-arm:
  uses: ./.github/workflows/reusable-docker-test.yml
  with:
    image-tag: 'nv-ingest:test'
    platform: 'linux/arm64'
    test-selection: 'random'
    random-count: '100'
```

---

### `reusable-conda-build.yml`

**Purpose**: Build conda packages

**Inputs**:
- `version`: Explicit version (optional)
- `source-ref`: Git ref to build from
- `runner`: GitHub runner
- `upload-artifacts`: Upload build artifacts

**Container**: `rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.12`

**Outputs**:
- `package-path`: Path to built packages

**Artifacts**: Conda packages (if upload enabled)

---

### `reusable-conda-publish.yml`

**Purpose**: Publish conda packages

**Inputs**:
- `channel`: Target channel (`dev` or `main`)
- `package-path`: Path to packages
- `force-upload`: Overwrite existing packages

**Secrets**:
- `NVIDIA_CONDA_TOKEN`: Anaconda authentication

**Validation**: Ensures channel is `dev` or `main`

---

### `reusable-pypi-build.yml`

**Purpose**: Build Python wheels

**Inputs**:
- `version`: Explicit version (optional, date if omitted)
- `release-type`: `dev` or `release`
- `source-ref`: Git ref to build from
- `runner`: GitHub runner

**Outputs**:
- `version`: Version that was built

**Artifacts**: Python wheels and source distributions

**Process**:
1. Installs build dependencies
2. Builds all three packages (api, client, service)
3. Uploads artifacts for publishing

---

### `reusable-pypi-publish.yml`

**Purpose**: Publish Python wheels to Artifactory

**Inputs**:
- `repository-url`: PyPI repository URL

**Secrets**:
- `ARTIFACTORY_URL`: Repository URL
- `ARTIFACTORY_USERNAME`: Auth username
- `ARTIFACTORY_PASSWORD`: Auth password

**Process**:
1. Downloads wheel artifacts
2. Installs twine
3. Publishes all packages

---

### `reusable-integration-test.yml`

**Purpose**: Run integration tests with conda environment

**Inputs**:
- `runner`: GitHub runner
- `python-version`: Python version (default: `3.12.11`)
- `timeout-minutes`: Job timeout (default: 60)

**Secrets**: Multiple NVIDIA NIM and service endpoints

**Process**:
1. Download conda packages (from artifacts)
2. Setup Miniconda
3. Install packages and dependencies
4. Run integration tests

**Dependencies**:
- NVIDIA NIMs (audio, VLM, OCR, YOLOX)
- Milvus
- Various Python packages

---

## Composite Actions

### `setup-docker-buildx`

**Purpose**: Setup Docker Buildx with optional QEMU

**Inputs**:
- `use-qemu`: Enable QEMU emulation (default: `false`)
- `platforms`: Supported platforms (default: `linux/amd64`)

**Steps**:
1. Setup QEMU (if enabled)
2. Setup Docker Buildx

**Usage**:
```yaml
- uses: ./.github/actions/setup-docker-buildx
  with:
    use-qemu: 'true'
    platforms: 'linux/amd64,linux/arm64'
```

---

### `docker-login-ngc`

**Purpose**: Authenticate with NGC registry

**Inputs**:
- `registry`: Registry URL (default: `nvcr.io`)
- `password`: NGC API token (required)

**Usage**:
```yaml
- uses: ./.github/actions/docker-login-ngc
  with:
    password: ${{ secrets.DOCKER_PASSWORD }}
```

---

### `determine-version`

**Purpose**: Determine version from various sources

**Inputs**:
- `version`: Explicit version (optional)
- `date-format`: Date format for auto-generation
- `branch-name`: Branch name to extract from

**Outputs**:
- `version`: Determined version string

**Priority**:
1. Explicit version input
2. Extract from branch name (release/*)
3. Generate from date

**Usage**:
```yaml
- id: version
  uses: ./.github/actions/determine-version
  with:
    date-format: '%Y.%m.%d'

- run: echo "Version is ${{ steps.version.outputs.version }}"
```

---

## Quick Reference

### Common Tasks

#### Run PR validation locally
```bash
# Pre-commit checks
pre-commit run --all-files

# Docker build
docker build --target runtime -t nv-ingest:test .

# Run tests
docker run nv-ingest:test pytest -m "not integration"
```

#### Trigger nightly build manually
```bash
# GitHub UI
Actions → Nightly Builds & Publishing → Run workflow
  Branch: main
```

#### Create a release
```bash
# Automatic - All three artifact types (recommended)
git checkout -b release/25.4.0
git push origin release/25.4.0
# → Automatically triggers:
#    - Docker (multi-platform)
#    - Conda (main channel)
#    - PyPI (release type)

# Manual - For custom options
Actions → Release - Docker/Conda/PyPI → Run workflow
```

#### Debug workflow issues
```bash
# View workflow runs
Actions → Select workflow → View runs

# Download artifacts
Actions → Workflow run → Artifacts section

# Re-run failed jobs
Actions → Workflow run → Re-run failed jobs
```

---

## Best Practices

1. **Always test workflows locally** when possible
2. **Use manual dispatch** for testing workflow changes
3. **Check artifacts** for build outputs and logs
4. **Monitor first run** after workflow changes
5. **Use skip flags** in nightly builds during maintenance
6. **Label external PRs** with `ok-to-test` after review
7. **Create release branches** from tested commits
8. **Verify secrets** are available before running workflows

---

## Maintenance

### Updating Docker Base Images

Edit base image references in workflows or create a workflow variable.

Current: `ubuntu:jammy-20250415.1`

### Updating Runner Types

Change `runner:` inputs in workflow calls:
- `ubuntu-latest`: Small jobs, public images
- `linux-large-disk`: Large Docker builds

### Updating Python/Conda Versions

Edit container images in reusable workflows:
- Current: `rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.12`

### Adding New Secrets

1. Add to repository secrets (Settings → Secrets → Actions)
2. Add to workflow secrets declarations
3. Pass through reusable workflow calls

---

## Support

For issues or questions:
- Check workflow logs in Actions tab
- Review this documentation
- Check migration guide: `WORKFLOWS_MIGRATION.md`
- Contact DevOps team
