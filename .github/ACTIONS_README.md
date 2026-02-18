# GitHub Actions Structure

This directory contains all GitHub Actions workflows, reusable components, and documentation for the nv-ingest CI/CD pipeline.

## 📁 Directory Structure

```
.github/
├── workflows/              # Workflows (including reusable workflows)
├── actions/                # Composite actions (3 actions)
├── ISSUE_TEMPLATE/         # Issue templates
├── CODEOWNERS             # Code ownership
├── PULL_REQUEST_TEMPLATE.md
├── copy-pr-bot.yaml
│
└── Documentation:
    ├── README.md (this file)   # Overview and quick reference
    ├── WORKFLOWS_REFERENCE.md  # Complete technical reference
    ├── WORKFLOWS_QUICKSTART.md # Quick start guide
    └── ARCHITECTURE.md         # System architecture
```

## 🚀 Quick Start

### For Developers
Read: [`WORKFLOWS_QUICKSTART.md`](./WORKFLOWS_QUICKSTART.md)

### For Complete Reference
Read: [`WORKFLOWS_REFERENCE.md`](./WORKFLOWS_REFERENCE.md)

### For Architecture Details
Read: [`ARCHITECTURE.md`](./ARCHITECTURE.md)

## 🎯 Workflow Overview

### Continuous Integration

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| PR Validation | `ci-pull-request.yml` | Pull requests | Pre-commit, build, test |
| Main CI | `ci-main.yml` | Push to main | Full validation + multi-platform |

### Nightly & Scheduled

| Workflow | File | Schedule | Purpose |
|----------|------|----------|---------|
| Nightly Builds | `scheduled-nightly.yml` | Daily 23:30 UTC | Docker + Conda + PyPI |

### Release Management

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| Docker Release | `release-docker.yml` | release/* branch OR manual | Publish Docker images |
| Conda Release | `release-conda.yml` | release/* branch OR manual | Publish Conda packages |
| PyPI Release | `release-pypi.yml` | release/* branch OR manual | Publish Python wheels |

### Documentation

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| Docs Deploy | `docs-deploy.yml` | Push to main OR manual | Deploy to GitHub Pages |

## 🔧 Reusable Components

### Workflows (in `workflows/`, prefixed with `reusable-`)

- `reusable-docker-build.yml` - Flexible Docker image building
- `reusable-docker-test.yml` - Container-based testing
- `reusable-conda-build.yml` - Conda package building
- `reusable-conda-publish.yml` - Conda package publishing
- `reusable-pypi-build.yml` - Python wheel building
- `reusable-pypi-publish.yml` - PyPI publishing
- `reusable-integration-test.yml` - Library mode testing

### Actions (in `actions/`)

- `setup-docker-buildx/` - Docker Buildx + QEMU setup
- `docker-login-ngc/` - NGC registry authentication
- `determine-version/` - Smart version determination

## 📊 Workflow Architecture

```
┌─────────────────────┐
│  Main Workflows     │  (Triggered by events)
│  - ci-pull-request │
│  - ci-main         │
│  - scheduled-*     │
│  - release-*       │
└──────┬──────────────┘
       │ calls
       ▼
┌─────────────────────┐
│ Reusable Workflows  │  (Business logic)
│  - docker-build    │
│  - conda-publish   │
│  - pypi-build      │
└──────┬──────────────┘
       │ uses
       ▼
┌─────────────────────┐
│ Composite Actions   │  (Common operations)
│  - setup-buildx    │
│  - docker-login    │
│  - determine-ver   │
└─────────────────────┘
```

## 🎯 Key Features

### Reusable Workflows
- Docker build logic defined once, used everywhere
- Consistent patterns across all workflows
- Type-safe interfaces with validation

### Flexible Configuration
- Reusable workflows accept inputs
- Composite actions are parameterized
- Easy to customize per use case

### Clear Separation
- Main workflows = triggers + orchestration
- Reusable workflows = business logic
- Composite actions = common operations

### ✅ Type Safety
- Inputs/outputs explicitly defined
- Required vs optional parameters
- Validation built-in

### ✅ Better Testing
- Reusable components can be tested independently
- workflow_dispatch for manual testing
- Clear job dependencies

## 🔐 Required Secrets

### Docker/NGC
- `DOCKER_PASSWORD` - NGC API token
- `DOCKER_REGISTRY` - Registry URL (e.g., nvcr.io)
- `HF_ACCESS_TOKEN` - Hugging Face token

### Conda
- `NVIDIA_CONDA_TOKEN` - Anaconda.org token

### PyPI
- `ARTIFACTORY_URL` - PyPI repository URL
- `ARTIFACTORY_USERNAME` - Username
- `ARTIFACTORY_PASSWORD` - Password

### Integration Tests
- `NGC_API_KEY` / `NVIDIA_API_KEY`
- `AUDIO_FUNCTION_ID`
- `EMBEDDING_NIM_MODEL_NAME`
- `NEMOTRON_PARSE_MODEL_NAME`
- `PADDLE_HTTP_ENDPOINT`
- `VLM_CAPTION_ENDPOINT`
- `VLM_CAPTION_MODEL_NAME`
- `YOLOX_*_HTTP_ENDPOINT` (multiple)

## 📝 Common Tasks

### Run PR checks locally
```bash
pre-commit run --all-files
docker build --target runtime -t nv-ingest:test .
docker run nv-ingest:test pytest -m "not integration"
```

### Trigger nightly build manually
```
Actions → "Nightly Builds & Publishing" → Run workflow
```

### Create a release
```bash
# Automatic - All three artifact types (recommended)
git checkout -b release/25.4.0
git push origin release/25.4.0
# → Triggers Docker, Conda, AND PyPI releases automatically

# Manual (for custom options)
Actions → Release - Docker/Conda/PyPI → Run workflow
```

### Debug workflows
```
Actions → Select workflow → View logs → Expand steps
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow not found | Check path: `.github/workflows/` (reusable workflows are `reusable-*.yml`) |
| Secret not available | Verify in Settings → Secrets → Actions |
| Build timeout | Use `linux-large-disk` runner |
| Integration tests fail | Check NIM endpoints and credentials |

## 📚 Documentation

- **Quick Start**: [`WORKFLOWS_QUICKSTART.md`](./WORKFLOWS_QUICKSTART.md)
- **Complete Reference**: [`WORKFLOWS_REFERENCE.md`](./WORKFLOWS_REFERENCE.md)
- **Architecture**: [`ARCHITECTURE.md`](./ARCHITECTURE.md)

## 🆘 Getting Help

1. Check workflow logs in Actions tab
2. Review documentation in this folder
3. Search for similar issues
4. Contact DevOps team
5. Open an issue with details

## 📞 Maintainers

See [`CODEOWNERS`](./CODEOWNERS) for ownership information.

---

**Architecture**: Reusable workflows + Composite actions  
**Documentation**: 4 comprehensive guides  
**Total Components**: Workflows + reusable workflows + composite actions
