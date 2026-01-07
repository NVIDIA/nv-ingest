# Release Process Update - January 2025

## 🎉 What Changed

The Conda and PyPI release workflows have been updated to match the Docker release workflow behavior:

### **Before**
- ❌ Docker: Automatic on `release/*` branch creation
- ❌ Conda: Manual only
- ❌ PyPI: Manual only

### **After**
- ✅ **Docker**: Automatic on `release/*` branch creation OR manual
- ✅ **Conda**: Automatic on `release/*` branch creation OR manual
- ✅ **PyPI**: Automatic on `release/*` branch creation OR manual

---

## 🚀 New Unified Release Process

### **One Branch = Three Artifacts!**

Creating a release branch now automatically triggers **all three** release workflows in parallel:

```bash
git checkout -b release/25.4.0
git push origin release/25.4.0

# Automatically triggers:
# 1. Docker Release (multi-platform) → NGC
# 2. Conda Release (main channel) → RapidsAI
# 3. PyPI Release (release type) → Artifactory
```

### **What Gets Built**

| Artifact | Registry | Version | Channel/Type | Platforms |
|----------|----------|---------|--------------|-----------|
| Docker | NGC | 25.4.0 | - | amd64, arm64 |
| Conda | RapidsAI | 25.4.0 | main | linux-64 |
| PyPI | Artifactory | 25.4.0 | release | all |

---

## 🔧 Changes to Workflows

### `release-conda.yml`

**New Trigger**:
```yaml
on:
  create:
    branches:
      - 'release/*'
  workflow_dispatch:  # Still supports manual
```

**Smart Defaults** (when triggered by branch):
- **Version**: Extracted from branch name (`release/25.4.0` → `25.4.0`)
- **Channel**: `main` (production channel)
- **Source**: The release branch itself

**Manual Override** (workflow_dispatch):
- Can specify custom version
- Can choose `dev` or `main` channel
- Can build from different branch

### `release-pypi.yml`

**New Trigger**:
```yaml
on:
  create:
    branches:
      - 'release/*'
  workflow_dispatch:  # Still supports manual
```

**Smart Defaults** (when triggered by branch):
- **Version**: Extracted from branch name (`release/25.4.0` → `25.4.0`)
- **Release Type**: `release` (production release)
- **Source**: The release branch itself

**Manual Override** (workflow_dispatch):
- Can specify custom version
- Can choose `dev` or `release` type
- Can build from different branch

---

## 📋 Updated Release Workflow

### Option 1: Automatic (Recommended)

**Single command releases everything:**

```bash
# 1. Create release branch
git checkout -b release/25.4.0
git push origin release/25.4.0

# 2. Wait for workflows to complete (~30-45 minutes)
#    - Monitor in GitHub Actions tab

# 3. Verify artifacts:
#    - NGC: nvcr.io/.../nv-ingest:25.4.0
#    - Conda: conda install -c nvidia/main nv-ingest=25.4.0
#    - PyPI: Check Artifactory for wheels

# 4. Merge release branch to main (if needed)
git checkout main
git merge release/25.4.0
git push origin main
```

### Option 2: Manual (For Custom Options)

**Use when you need specific configurations:**

#### Custom Conda Channel (e.g., dev for testing)
```bash
Actions → Release - Conda → Run workflow
  Version: 25.4.0
  Channel: dev  # Test in dev before main
  Source: release/25.4.0
```

#### Custom PyPI Release Type (e.g., dev builds)
```bash
Actions → Release - PyPI → Run workflow
  Version: 25.4.0-rc1
  Release type: dev  # For release candidates
  Source: release/25.4.0
```

#### Build from Different Source
```bash
Actions → Release - Docker/Conda/PyPI → Run workflow
  Version: 25.4.0
  Source: hotfix/urgent-fix  # Build from hotfix branch
```

---

## 🎯 Benefits

### For Release Managers
- ✅ **One command** triggers all releases
- ✅ **Consistent versioning** across all artifacts
- ✅ **Parallel execution** for faster releases
- ✅ **Automatic by default**, manual when needed

### For DevOps
- ✅ **Less manual work** - no need to trigger 3 workflows
- ✅ **Fewer errors** - version extracted from branch name
- ✅ **Better audit trail** - all triggered from same event
- ✅ **Rollback friendly** - delete branch to prevent future triggers

### For Developers
- ✅ **Predictable behavior** - all releases work the same way
- ✅ **Easy to test** - create test release branch
- ✅ **Clear process** - one documented workflow

---

## 📊 Comparison: Old vs New

### Old Process (Multiple Steps)

```
1. Create release/25.4.0 branch
   → Docker automatically builds

2. Go to GitHub Actions
   → Manually trigger Conda release
   → Fill in version: 25.4.0
   → Fill in channel: main

3. Go to GitHub Actions again
   → Manually trigger PyPI release
   → Fill in version: 25.4.0
   → Fill in release type: release

Total: 3 separate actions, manual input required
```

### New Process (Single Step)

```
1. Create release/25.4.0 branch
   → Docker automatically builds
   → Conda automatically builds (main channel)
   → PyPI automatically builds (release type)

Total: 1 action, fully automatic
```

---

## ⚠️ Important Notes

### Version Extraction
The version is automatically extracted from the branch name:
- ✅ `release/25.4.0` → version `25.4.0`
- ✅ `release/1.0.0-rc1` → version `1.0.0-rc1`
- ✅ `release/2024.01.15` → version `2024.01.15`

### Default Settings (Branch Trigger)
When triggered by branch creation:
- **Conda Channel**: `main` (production)
- **PyPI Release Type**: `release` (production)

If you need different settings, use manual workflow_dispatch.

### Parallel Execution
All three workflows run in parallel:
- Fastest: PyPI (~15-20 min)
- Medium: Conda (~20-30 min)
- Slowest: Docker (~30-40 min)

Total time: ~40 minutes (vs sequential would be ~75 min)

---

## 🧪 Testing the New Process

### Test with Non-Production Settings

```bash
# Create test release branch
git checkout -b release/0.0.1-test
git push origin release/0.0.1-test

# This will trigger:
# - Docker: 0.0.1-test tag
# - Conda: main channel (!!!)
# - PyPI: release type (!!!)

# If you want dev/test channels:
# Manually trigger Conda with channel=dev
# Manually trigger PyPI with release-type=dev
```

**Recommendation**: For testing, use manual triggers with dev/test settings rather than branch creation.

---

## 📚 Documentation Updated

All documentation has been updated to reflect these changes:

- ✅ `README.md` - Overview updated
- ✅ `WORKFLOWS_QUICKSTART.md` - Quick reference updated
- ✅ `WORKFLOWS_REFERENCE.md` - Complete technical reference updated
- ✅ `WORKFLOWS_MIGRATION.md` - Migration guide updated
- ✅ `ARCHITECTURE.md` - Architecture diagrams updated
- ✅ `REFACTORING_SUMMARY.md` - Summary updated
- ✅ `MIGRATION_PLAN.md` - Migration checklist updated

---

## 🎓 Examples

### Example 1: Standard Release
```bash
git checkout -b release/26.2.0
git push origin release/26.2.0
# Wait for all three workflows
# Verify artifacts
```

### Example 2: Release Candidate
```bash
# Use manual triggers for RC
Actions → Release - Docker → Run workflow
  Version: 26.2.0-rc1

Actions → Release - Conda → Run workflow
  Version: 26.2.0-rc1
  Channel: dev

Actions → Release - PyPI → Run workflow
  Version: 26.2.0-rc1
  Release type: dev
```

### Example 3: Hotfix Release
```bash
git checkout -b release/26.1.1
git push origin release/26.1.1
# Automatically releases 26.1.1
```

---

## ✅ Checklist for First Release

- [ ] Understand the new automatic trigger
- [ ] Review default settings (main channel, release type)
- [ ] Create release branch
- [ ] Monitor all three workflows in Actions tab
- [ ] Verify Docker image in NGC
- [ ] Verify Conda package in RapidsAI main channel
- [ ] Verify PyPI wheels in Artifactory
- [ ] Merge release branch to main (if applicable)
- [ ] Tag the release in git
- [ ] Update CHANGELOG

---

## 🆘 Troubleshooting

### Workflow didn't trigger
- Check branch name matches `release/*` pattern
- Verify you pushed the branch (not just created locally)
- Check Actions tab for workflow runs

### Wrong version built
- Check branch name format: `release/X.Y.Z`
- Version is extracted from text after `release/`

### Need different channel/type
- Use manual workflow_dispatch
- Override default settings with inputs

### Want to stop a release
- Cancel running workflows in Actions tab
- Delete the release branch to prevent re-triggers

---

**Last Updated**: January 6, 2025  
**Change**: Unified automatic release process for all three artifact types  
**Impact**: Significantly simplified release workflow
