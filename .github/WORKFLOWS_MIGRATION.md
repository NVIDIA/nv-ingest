# GitHub Actions Migration Guide

This guide helps you migrate from the old workflow structure to the new refactored workflows.

## 📋 Overview of Changes

The GitHub Actions have been completely refactored to improve:
- **Reusability**: Common operations defined once in reusable workflows
- **Maintainability**: Changes to build logic happen in one place
- **Clarity**: Clear separation of concerns (CI, nightly, release)
- **Efficiency**: Better parallelization and faster feedback

## 🗂️ New Structure

```
.github/
├── workflows/                          # Main trigger workflows
│   ├── ci-pull-request.yml            # PR validation
│   ├── ci-main.yml                    # Main branch CI
│   ├── scheduled-nightly.yml          # Nightly builds (replaces 3 workflows)
│   ├── release-docker.yml             # Docker releases
│   ├── release-conda.yml              # Conda releases
│   ├── release-pypi.yml               # PyPI releases
│   └── docs-deploy.yml                # Documentation
│
├── workflows-reusable/                 # Reusable workflow components
│   ├── docker-build.yml               # Docker build logic
│   ├── docker-test.yml                # Docker testing
│   ├── conda-build.yml                # Conda package building
│   ├── conda-publish.yml              # Conda publishing
│   ├── pypi-build.yml                 # Python wheel building
│   ├── pypi-publish.yml               # PyPI publishing
│   └── integration-test.yml           # Integration testing
│
└── actions/                            # Composite actions
    ├── setup-docker-buildx/           # Docker Buildx setup
    ├── docker-login-ngc/              # NGC authentication
    └── determine-version/             # Version determination
```

## 🔄 Workflow Mapping (Old → New)

### Old Workflows → New Workflows

| Old Workflow | New Workflow(s) | Notes |
|-------------|-----------------|-------|
| `build-docs.yml` | `docs-deploy.yml` | Renamed for clarity |
| `conda-publish.yml` | `scheduled-nightly.yml`<br>`release-conda.yml` | Split: nightly vs release |
| `docker-build-arm.yml` | `ci-main.yml` | Integrated into main CI |
| `docker-build.yml` | `ci-pull-request.yml` | Part of PR validation |
| `docker-nightly-publish.yml` | `scheduled-nightly.yml` | Consolidated |
| `docker-release-publish.yml` | `release-docker.yml` | Renamed |
| `pre-commit.yml` | `ci-pull-request.yml`<br>`ci-main.yml` | Integrated into CI |
| `pypi-nightly-publish.yml` | `scheduled-nightly.yml` | Consolidated |
| `test-library-mode.yml` | `ci-pull-request.yml`<br>`ci-main.yml` | Integrated into CI |

## 📊 Feature Comparison

### Nightly Builds (Scheduled at 23:30 UTC)

**Old**: 3 separate workflows
- `conda-publish.yml`
- `docker-nightly-publish.yml`
- `pypi-nightly-publish.yml`

**New**: 1 unified workflow
- `scheduled-nightly.yml` - orchestrates all nightly builds
- Can selectively skip Docker/Conda/PyPI via workflow_dispatch
- Consistent version numbering across all artifacts

### Pull Request Validation

**Old**: 2-3 separate workflows
- `pre-commit.yml`
- `docker-build.yml`
- `test-library-mode.yml` (conditional)

**New**: 1 unified workflow
- `ci-pull-request.yml` - runs all PR checks
- Clear job dependencies and status reporting
- Better visibility with summary job

### Release Process

**Old**: Mixed manual/automatic triggers
- Branch creation for Docker only
- Manual dispatch for Conda/PyPI

**New**: Unified automatic release process
- `release-docker.yml` - automatic on branch creation OR manual
- `release-conda.yml` - automatic on branch creation OR manual
- `release-pypi.yml` - automatic on branch creation OR manual
- **Single branch creation triggers all three artifact types!**

## 🚀 How to Use the New Workflows

### For Developers (Pull Requests)

**Automatic on PR Creation:**
```
1. Pre-commit checks run
2. Docker build (amd64) + full test suite
3. Library mode tests (if approved)
```

**External Contributors:**
- Add `ok-to-test` label to enable integration tests

### For Maintainers (Main Branch)

**Automatic on Merge to Main:**
```
1. Pre-commit checks
2. Docker build + test (amd64 and arm64)
3. Library mode build + integration tests
```

### Nightly Builds (Automated)

**Runs Daily at 23:30 UTC:**
```bash
# All three publish automatically:
- Docker: YYYY.MM.DD tag to NGC
- Conda: dev channel
- PyPI: dev release to Artifactory
```

**Manual Trigger with Options:**
```bash
# GitHub UI → Actions → "Nightly Builds & Publishing" → Run workflow
Options:
  - Skip Docker: yes/no
  - Skip Conda: yes/no
  - Skip PyPI: yes/no
```

### Release Process

#### Unified Release Process (Recommended)

**Automatic** - Create release branch (triggers all three):
```bash
git checkout -b release/25.4.0
git push origin release/25.4.0
# Automatically triggers:
# 1. Multi-platform Docker build → NGC
# 2. Conda packages → main channel
# 3. PyPI wheels → Artifactory (release type)
```

#### Individual Manual Releases (for custom options)

**Docker Release:**
```bash
# GitHub UI → Actions → "Release - Docker" → Run workflow
Inputs:
  - Version: 25.4.0 (optional, extracted from branch if empty)
  - Source ref: main (or branch/tag)
```

**Conda Release:**
```bash
# GitHub UI → Actions → "Release - Conda" → Run workflow
Inputs:
  - Version: 25.4.0 (optional, extracted from branch if empty)
  - Channel: main (default) or dev (for testing)
  - Source ref: main
```

**PyPI Release:**
```bash
# GitHub UI → Actions → "Release - PyPI" → Run workflow
Inputs:
  - Version: 25.4.0 (optional, extracted from branch if empty)
  - Release type: release (default) or dev
  - Source ref: main
```

## 🔧 Customization Guide

### Modify Docker Build

Edit `.github/workflows-reusable/docker-build.yml`:
```yaml
# Change default platforms, add build args, modify build logic
```

All workflows using Docker builds will automatically use the updated logic.

### Modify Test Strategy

Edit `.github/workflows-reusable/docker-test.yml`:
```yaml
# Change pytest flags, coverage settings, test selection
```

### Change Nightly Schedule

Edit `.github/workflows/scheduled-nightly.yml`:
```yaml
on:
  schedule:
    - cron: "30 23 * * *"  # Change time here
```

### Add New CI Checks

Edit `.github/workflows/ci-pull-request.yml`:
```yaml
jobs:
  my-new-check:
    name: My Custom Check
    runs-on: ubuntu-latest
    steps:
      - # your steps here
      
  pr-validation-complete:
    needs:
      - pre-commit
      - docker-test
      - my-new-check  # Add to dependencies
```

## 🎯 Benefits of New Structure

### Before (Old Structure)
- **9 workflow files** with duplicated logic
- Docker build steps repeated **5+ times**
- Hard to change build process (edit 5+ files)
- Inconsistent versioning across artifacts
- Poor visibility into workflow relationships

### After (New Structure)
- **7 main workflows** + 6 reusable + 3 actions
- Docker build defined **once**
- Change build process **in one place**
- Consistent versioning with `determine-version` action
- Clear hierarchy and dependencies

### Metrics
- **~60% reduction** in duplicated code
- **Single source of truth** for each operation
- **Faster iteration** with reusable components
- **Better testing** of workflow components
- **Improved maintainability** with clear separation

## 🔐 Required Secrets (Unchanged)

All existing secrets are still required:
- `HF_ACCESS_TOKEN` - Hugging Face
- `DOCKER_PASSWORD` - NGC registry
- `DOCKER_REGISTRY` - NGC registry URL
- `NVIDIA_CONDA_TOKEN` - Conda publishing
- `NVIDIA_API_KEY` / `NGC_API_KEY` - NIM access
- `ARTIFACTORY_URL/USERNAME/PASSWORD` - PyPI publishing
- NIM endpoint secrets for integration tests

## 📝 Migration Checklist

- [ ] Review new workflow structure
- [ ] Test PR workflow with a test PR
- [ ] Verify nightly builds work (manual trigger test)
- [ ] Test release workflows (use non-prod if possible)
- [ ] Update team documentation/runbooks
- [ ] Archive old workflows (move to `.github/workflows-old/`)
- [ ] Monitor first few nightly runs
- [ ] Update CI status badges in README if needed

## ❓ Troubleshooting

### Workflow not triggering?
- Check branch protection rules
- Verify trigger conditions in workflow YAML
- Check GitHub Actions permissions in repo settings

### Reusable workflow not found?
- Reusable workflows must be in `.github/workflows-reusable/`
- Uses relative path: `uses: ./.github/workflows-reusable/name.yml`
- Must be on the same branch/commit

### Secret not available in reusable workflow?
- Secrets must be passed explicitly from caller
- Add to `secrets:` section in workflow call

### Docker build failing?
- Check `workflows-reusable/docker-build.yml` for build logic
- Verify base image and tags are correct
- Check build args are being passed correctly

## 📚 Additional Resources

- [GitHub Actions: Reusing Workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows)
- [GitHub Actions: Custom Actions](https://docs.github.com/en/actions/creating-actions/about-custom-actions)
- [GitHub Actions: Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

## 🤝 Support

For questions or issues:
1. Check this migration guide
2. Review workflow documentation in each YAML file
3. Contact DevOps team
4. Open an issue in the repository
