# GitHub Actions Quick Start

A quick reference guide for common GitHub Actions operations in nv-ingest.

## 🎯 For Developers

### Creating a Pull Request

**What runs automatically:**
1. ✅ Pre-commit checks (linting, formatting)
2. ✅ Docker build + full test suite (amd64)
3. ⏸️ Integration tests (needs approval for external contributors)

**Expected time:** ~15-20 minutes

**If you're an external contributor:**
- Wait for a maintainer to add the `ok-to-test` label
- Integration tests will run after approval

### Checking PR Status

```
GitHub PR page → Checks tab
```

Required checks:
- ✅ `pre-commit` - Must pass
- ✅ `docker-test` - Must pass
- ⚪ `library-mode-test` - Optional but recommended

### Common Issues

**Pre-commit failing?**
```bash
# Run locally and fix
pre-commit run --all-files
```

**Docker tests failing?**
```bash
# Build and test locally
docker build --target runtime -t nv-ingest:test .
docker run nv-ingest:test pytest -m "not integration"
```

---

## 🚀 For Maintainers

### Merging to Main

**What runs automatically:**
1. All PR checks
2. ARM64 build + tests (parallel with above)
3. Library mode integration tests

**Expected time:** ~30-45 minutes

### Nightly Builds

**Automatic:** Every day at 23:30 UTC

**Manual trigger:**
```
Actions → "Nightly Builds & Publishing" → Run workflow
```

**Options:**
- Skip Docker build
- Skip Conda publish
- Skip PyPI publish

### Releasing

#### Quick Release Checklist
1. Ensure main branch is stable
2. Create release branch
3. Verify artifacts in respective registries
4. Update release notes

#### Unified Release Process (Recommended)

**Automatic - All artifacts:**
```bash
git checkout -b release/25.4.0
git push origin release/25.4.0
# → Triggers ALL THREE automatically:
#   - Multi-platform Docker image to NGC
#   - Conda packages to main channel
#   - PyPI wheels (release type) to Artifactory
```

#### Manual Release (for custom options)

**Docker Release:**
```
Actions → "Release - Docker" → Run workflow
  Version: 25.4.0
  Source: main
```

**Conda Release:**
```
Actions → "Release - Conda" → Run workflow
  Version: 25.4.0
  Channel: main (or dev for testing)
  Source: main
```

**PyPI Release:**
```
Actions → "Release - PyPI" → Run workflow
  Version: 25.4.0
  Release type: release (or dev)
  Source: main
```

### Approving External Contributor PRs

**For integration tests to run:**
1. Review the PR code changes
2. Add label: `ok-to-test`
3. Integration tests will run automatically

---

## 📊 Understanding Workflow Status

### PR Workflow Status

| Symbol | Meaning |
|--------|---------|
| 🟢 | All checks passed - safe to merge |
| 🟡 | Checks in progress - wait |
| 🔴 | Checks failed - needs fixes |
| ⚪ | Optional check - review recommended |

### Nightly Build Status

**Check last night's build:**
```
Actions → "Nightly Builds & Publishing" → Latest run
```

**What to verify:**
- ✅ All three jobs completed (Docker, Conda, PyPI)
- ✅ Version tagged correctly (YYYY.MM.DD)
- ✅ No artifact upload failures

---

## 🔧 Common Operations

### Re-run Failed Workflows

```
Actions → Select workflow run → Re-run failed jobs
```

### Download Build Artifacts

```
Actions → Workflow run → Artifacts section → Download
```

Available artifacts:
- `pytest-coverage-*` - Test coverage reports
- `conda-packages` - Built conda packages
- `python-wheels` - Built Python wheels
- `test_artifacts` - Integration test data

### Check Workflow Logs

```
Actions → Workflow run → Select job → Expand step
```

**Tip:** Use browser search (Ctrl+F) to find errors quickly

### Cancel Running Workflows

```
Actions → Workflow run → Cancel workflow
```

**Note:** PR workflows auto-cancel on new pushes

---

## 🐛 Troubleshooting

### "Workflow not found" error

**Cause:** Reusable workflow not in correct location

**Fix:** Ensure `.github/workflows-reusable/*.yml` exists

### "Secret not found" error

**Cause:** Missing or incorrect secret name

**Fix:** Check Settings → Secrets → Actions

Required secrets:
- `HF_ACCESS_TOKEN`
- `DOCKER_PASSWORD`
- `DOCKER_REGISTRY`
- `NVIDIA_CONDA_TOKEN`
- `NVIDIA_API_KEY`
- `ARTIFACTORY_*`
- Multiple NIM endpoints

### Docker build timeout

**Cause:** Large builds on slow runners

**Fix:** 
- Use `linux-large-disk` runner
- Check base image availability
- Verify network connectivity

### Integration tests failing

**Cause:** NIM endpoints unavailable or credentials expired

**Fix:**
- Verify all NIM secrets are current
- Check endpoint availability
- Review test logs for specific failures

---

## 📚 More Information

- **Detailed Reference:** [WORKFLOWS_REFERENCE.md](./WORKFLOWS_REFERENCE.md)
- **Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **GitHub Actions Docs:** https://docs.github.com/actions

---

## 🆘 Getting Help

1. **Check workflow logs** in Actions tab
2. **Search similar issues** in repository
3. **Review documentation** in `.github/` folder
4. **Ask maintainers** or DevOps team
5. **Open an issue** with workflow logs attached

---

## 💡 Tips

- ✅ Always test locally before pushing
- ✅ Use pre-commit hooks to catch issues early
- ✅ Check PR status before requesting review
- ✅ Re-run failed jobs once (may be transient)
- ✅ Use workflow_dispatch for testing changes
- ✅ Monitor first few nightly builds after workflow changes
- ✅ Keep secrets up to date (especially API keys)
- ✅ Tag releases properly for traceability
