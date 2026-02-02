# GitHub Actions Documentation Index

**Complete CI/CD pipeline documentation for nv-ingest**

---

## 📦 What's Included

This documentation package contains **17 components** organized into a clean, maintainable structure:

### 🎯 Main Workflows (7 files)
Located in: `.github/workflows/`

1. **`ci-pull-request.yml`** - PR validation with pre-commit, Docker build/test, library mode
2. **`ci-main.yml`** - Main branch CI with multi-platform testing
3. **`scheduled-nightly.yml`** - Unified nightly builds (Docker + Conda + PyPI)
4. **`release-docker.yml`** - Docker release publishing
5. **`release-conda.yml`** - Conda package releases
6. **`release-pypi.yml`** - PyPI wheel releases
7. **`docs-deploy.yml`** - Documentation deployment

### ♻️ Reusable Workflows (7 files)
Located in: `.github/workflows-reusable/`

1. **`docker-build.yml`** - Flexible Docker image building
2. **`docker-test.yml`** - Container-based testing
3. **`conda-build.yml`** - Conda package building
4. **`conda-publish.yml`** - Conda publishing to channels
5. **`pypi-build.yml`** - Python wheel building
6. **`pypi-publish.yml`** - PyPI publishing
7. **`integration-test.yml`** - Library mode integration tests

### 🔧 Composite Actions (3 directories)
Located in: `.github/actions/`

1. **`setup-docker-buildx/`** - Docker Buildx + QEMU setup
2. **`docker-login-ngc/`** - NGC registry authentication
3. **`determine-version/`** - Smart version determination

### 📚 Documentation (4 files)
Located in: `.github/`

1. **`README.md`** - Main entry point with directory overview
2. **`WORKFLOWS_QUICKSTART.md`** - Quick start guide for developers
3. **`WORKFLOWS_REFERENCE.md`** - Complete technical reference
4. **`ARCHITECTURE.md`** - Visual architecture diagrams
5. **`INDEX.md`** - This file

---

## 🎓 How to Use This Documentation

### For Different Audiences:

#### 👨‍💻 **Developers** (Contributing code)
1. Read: **`WORKFLOWS_QUICKSTART.md`**
2. Reference: **`README.md`** for quick lookups

#### 🔧 **Maintainers** (Managing releases)
1. Read: **`WORKFLOWS_QUICKSTART.md`** (Common tasks)
2. Reference: **`WORKFLOWS_REFERENCE.md`** (Complete details)

#### 🏗️ **DevOps/SRE** (System maintenance)
1. Read: **`ARCHITECTURE.md`** (System design)
2. Read: **`WORKFLOWS_REFERENCE.md`** (Technical specs)
3. Reference: **`README.md`** (Overview)

#### 📊 **Management** (Understanding scope)
1. Read: **`README.md`** (Executive summary)
2. Review: **`ARCHITECTURE.md`** (Visual diagrams)

---

## 📊 System Overview

### Workflow Components

- **7** main trigger workflows
- **7** reusable workflow components  
- **3** composite actions
- **4** documentation files

### Key Capabilities

- ✅ Automated PR validation
- ✅ Multi-platform Docker builds (amd64, arm64)
- ✅ Daily nightly builds (Docker, Conda, PyPI)
- ✅ Unified release process (one branch = all artifacts)
- ✅ Integration testing with conda environment
- ✅ Automatic documentation deployment

---

## 🚀 Getting Started

### Step 1: Understand the Structure
```bash
# Read the main README
cat .github/README.md

# Review the architecture
cat .github/ARCHITECTURE.md
```

### Step 2: Choose Your Path

**Quick Start (Developers):**
```bash
cat .github/WORKFLOWS_QUICKSTART.md
# Start contributing immediately
```

**Complete Reference (Advanced):**
```bash
cat .github/WORKFLOWS_REFERENCE.md
# Deep dive into every workflow
```

### Step 3: Start Using

**For PRs:**
1. Create a pull request
2. Watch automated checks run
3. Address any failures

**For Releases:**
```bash
git checkout -b release/25.4.0
git push origin release/25.4.0
# Automatically releases Docker, Conda, and PyPI
```

---

## 🔐 Security & Access

### Required Secrets

All secrets must be configured in repository settings:

**Docker/NGC:**
- `DOCKER_PASSWORD`
- `DOCKER_REGISTRY`
- `HF_ACCESS_TOKEN`

**Conda:**
- `NVIDIA_CONDA_TOKEN`

**PyPI:**
- `ARTIFACTORY_URL`
- `ARTIFACTORY_USERNAME`
- `ARTIFACTORY_PASSWORD`

**Integration Tests:**
- `NGC_API_KEY` / `NVIDIA_API_KEY`
- Multiple NIM endpoint secrets

### Access Control
- External contributors require `ok-to-test` label
- `pull_request_target` used safely with access checks
- Secrets passed explicitly (no implicit access)
- Minimal permissions (least privilege)

---

## 🎯 Quick Reference

### Common Tasks

| Task | Location | Action |
|------|----------|--------|
| View workflows | `.github/workflows/` | Browse main triggers |
| Understand logic | `.github/workflows-reusable/` | See business logic |
| Check common operations | `.github/actions/` | Review composite actions |
| Quick help | `.github/WORKFLOWS_QUICKSTART.md` | Read guide |
| Complete reference | `.github/WORKFLOWS_REFERENCE.md` | Deep dive |

### Workflow Triggers

| Workflow | Automatic | Manual | Purpose |
|----------|-----------|--------|---------|
| PR Validation | PR events | ✓ | Validate changes |
| Main CI | Push to main | ✓ | Full validation |
| Nightly | Daily 23:30 UTC | ✓ | Build & publish |
| Docker Release | release/* branch | ✓ | Release Docker |
| Conda Release | release/* branch | ✓ | Release Conda |
| PyPI Release | release/* branch | ✓ | Release PyPI |
| Docs | Push to main | ✓ | Deploy docs |

---

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution Document | Section |
|-------|------------------|---------|
| Workflow not triggering | `WORKFLOWS_QUICKSTART.md` | Troubleshooting |
| Reusable workflow not found | `WORKFLOWS_REFERENCE.md` | Reusable Workflows |
| Secret not available | `README.md` | Required Secrets |
| Build failing | `WORKFLOWS_REFERENCE.md` | Docker Build |
| Integration tests failing | `WORKFLOWS_QUICKSTART.md` | Troubleshooting |

### Getting Help

1. **Check logs**: Actions tab → Workflow run → Job → Step
2. **Review docs**: Search in `.github/` documentation
3. **Test locally**: Run pre-commit and Docker builds
4. **Ask team**: Contact DevOps or maintainers
5. **Open issue**: Include logs and context

---

## 📚 File Index

### Documentation Files
```
.github/
├── INDEX.md                      ← You are here
├── README.md                     ← Start here (overview)
├── ARCHITECTURE.md               ← How it works (diagrams)
├── WORKFLOWS_QUICKSTART.md       ← Quick reference (developers)
└── WORKFLOWS_REFERENCE.md        ← Complete reference (advanced)
```

### Workflow Files
```
.github/
├── workflows/                    ← Main trigger workflows (7)
├── workflows-reusable/           ← Reusable components (7)
└── actions/                      ← Composite actions (3)
```

---

**For questions or issues, start with**: `.github/README.md`
