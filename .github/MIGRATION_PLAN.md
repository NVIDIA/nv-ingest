# Migration Plan: Old Workflows Handling

## 📋 Old Workflows to Archive/Remove

The following **9 workflows** in `.github/workflows/` are replaced by the new structure:

### Files to Eventually Remove

1. ✅ **`build-docs.yml`** → Replaced by `docs-deploy.yml`
2. ✅ **`conda-publish.yml`** → Replaced by `scheduled-nightly.yml` + `release-conda.yml`
3. ✅ **`docker-build-arm.yml`** → Integrated into `ci-main.yml`
4. ✅ **`docker-build.yml`** → Replaced by `ci-pull-request.yml`
5. ✅ **`docker-nightly-publish.yml`** → Replaced by `scheduled-nightly.yml`
6. ✅ **`docker-release-publish.yml`** → Replaced by `release-docker.yml`
7. ✅ **`pre-commit.yml`** → Integrated into `ci-pull-request.yml` + `ci-main.yml`
8. ✅ **`pypi-nightly-publish.yml`** → Replaced by `scheduled-nightly.yml`
9. ✅ **`test-library-mode.yml`** → Integrated into `ci-pull-request.yml` + `ci-main.yml`

---

## 🔄 Recommended Migration Approach

### Option 1: Gradual Migration (Recommended)

**Phase 1: Parallel Running (Week 1-2)**
```bash
# Keep old workflows active
# Enable new workflows
# Compare outputs side-by-side
```

**Phase 2: Switch Primary (Week 3)**
```bash
# Make new workflows primary
# Keep old workflows as backup (disabled)
```

**Phase 3: Cleanup (Week 4)**
```bash
# Archive old workflows
mkdir -p .github/workflows-old
mv .github/workflows/build-docs.yml .github/workflows-old/
mv .github/workflows/conda-publish.yml .github/workflows-old/
# ... etc for all 9 files
```

### Option 2: Clean Cut (Advanced)

**If confident after testing:**
```bash
# Disable all old workflows at once
# Move to archive folder
# Monitor new workflows closely
```

---

## 📝 Step-by-Step Migration Commands

### Step 1: Create Archive Directory
```bash
cd /home/jdyer/Development/nv-ingest
mkdir -p .github/workflows-old
```

### Step 2: Disable Old Workflows (Rename)
```bash
# This prevents them from running but keeps them for reference
cd .github/workflows

# Rename to .yml.old to disable
mv build-docs.yml build-docs.yml.old
mv conda-publish.yml conda-publish.yml.old
mv docker-build-arm.yml docker-build-arm.yml.old
mv docker-build.yml docker-build.yml.old
mv docker-nightly-publish.yml docker-nightly-publish.yml.old
mv docker-release-publish.yml docker-release-publish.yml.old
mv pre-commit.yml pre-commit.yml.old
mv pypi-nightly-publish.yml pypi-nightly-publish.yml.old
mv test-library-mode.yml test-library-mode.yml.old
```

### Step 3: After Successful Testing, Archive
```bash
# Once confident new workflows work
cd .github/workflows
mv *.yml.old ../workflows-old/

# Or delete if you're fully confident
# rm *.yml.old
```

---

## ✅ Validation Checklist

Before removing old workflows, verify:

### PR Workflow Validation
- [ ] Test PR created and validated
- [ ] Pre-commit checks passed
- [ ] Docker build completed
- [ ] Tests passed with coverage
- [ ] Library mode tests worked (with approval)

### Main Branch Validation
- [ ] Push to main triggered ci-main.yml
- [ ] AMD64 build and test passed
- [ ] ARM64 build and test passed
- [ ] Library mode integration tests passed

### Nightly Build Validation
- [ ] Manual trigger of scheduled-nightly.yml successful
- [ ] Docker image published to NGC
- [ ] Conda packages published to dev channel
- [ ] PyPI wheels published to Artifactory
- [ ] Version tagged correctly (YYYY.MM.DD)

### Release Validation
- [ ] Create test release branch (e.g., `release/0.0.1-test`)
- [ ] Verify Docker release workflow triggered automatically
- [ ] Verify Conda release workflow triggered automatically
- [ ] Verify PyPI release workflow triggered automatically
- [ ] All artifacts published correctly
- [ ] Version numbers extracted correctly from branch name
- [ ] Manual workflow dispatch tested with custom inputs

### Documentation Validation
- [ ] Docs deploy triggered on main push
- [ ] GitHub Pages updated
- [ ] Documentation accessible

---

## 🔍 Comparison Testing

### Before Removing Old Workflows

Run both old and new in parallel and compare:

#### Docker Images
```bash
# Old workflow tag: nv-ingest:YYYY.MM.DD
# New workflow tag: nv-ingest:YYYY.MM.DD (should match)

# Verify they're identical
docker pull <registry>/nv-ingest:YYYY.MM.DD-old
docker pull <registry>/nv-ingest:YYYY.MM.DD-new
docker image inspect <both> | jq '.[0].RootFS'
```

#### Conda Packages
```bash
# Verify package names and versions match
anaconda show nvidia/nv-ingest
# Compare build dates and versions
```

#### PyPI Wheels
```bash
# Verify wheel files are generated correctly
# Check version strings
# Verify all three packages (api, client, service)
```

---

## 🚨 Rollback Plan

If issues arise after disabling old workflows:

### Quick Rollback
```bash
cd .github/workflows
# Rename back to .yml
mv build-docs.yml.old build-docs.yml
mv conda-publish.yml.old conda-publish.yml
# etc...

# Or if archived:
cp ../workflows-old/*.yml ./
```

### Partial Rollback
```bash
# Only rollback specific workflow
cp ../workflows-old/docker-nightly-publish.yml ./
# Disable corresponding new workflow
mv scheduled-nightly.yml scheduled-nightly.yml.disabled
```

---

## 📊 Side-by-Side Comparison

### When Running Both (Testing Phase)

| Aspect | Old Workflow | New Workflow | Status |
|--------|-------------|--------------|--------|
| Docker tag | YYYY.MM.DD | YYYY.MM.DD | ✓ Same |
| Platforms | amd64,arm64 | amd64,arm64 | ✓ Same |
| Registry | NGC | NGC | ✓ Same |
| Conda channel | dev/main | dev/main | ✓ Same |
| PyPI repo | Artifactory | Artifactory | ✓ Same |
| Secrets used | (list) | (list) | ✓ Same |

---

## 🎯 Success Criteria

Before archiving old workflows, confirm:

✅ **All functionality preserved**
- Every trigger condition works
- All artifacts are published
- All tests pass

✅ **No regressions**
- Build times similar or better
- Success rate same or better
- Artifacts identical

✅ **Team confidence**
- Developers comfortable with new workflows
- Maintainers can perform releases
- DevOps can debug issues

✅ **Documentation complete**
- All guides reviewed
- Team trained on new workflows
- Runbooks updated

---

## 📅 Suggested Timeline

### Week 1: Testing Phase
- **Day 1-2**: Review documentation, understand changes
- **Day 3-4**: Manual trigger all new workflows
- **Day 5**: Create test PR to validate ci-pull-request.yml

### Week 2: Parallel Phase
- **Day 1**: Enable both old and new workflows
- **Day 2-4**: Monitor both, compare outputs
- **Day 5**: Document any differences

### Week 3: Transition Phase
- **Day 1**: Disable old workflows (rename to .old)
- **Day 2-5**: Monitor new workflows closely, ready to rollback

### Week 4: Cleanup Phase
- **Day 1-2**: Archive old workflows if all is well
- **Day 3**: Update team documentation
- **Day 4-5**: Final validation and celebration 🎉

---

## 💡 Tips

1. **Test thoroughly** before disabling old workflows
2. **Keep backups** (rename to .old before deleting)
3. **Monitor closely** after switching
4. **Have rollback plan** ready
5. **Update documentation** after migration
6. **Train team** on new workflows
7. **Celebrate success** when complete! 🎊

---

## 🆘 If Something Goes Wrong

### Immediate Actions
1. **Stop**: Pause any ongoing migrations
2. **Rollback**: Re-enable old workflow (see Rollback Plan above)
3. **Investigate**: Check logs, compare outputs
4. **Fix**: Correct the issue in new workflows
5. **Re-test**: Validate fix works
6. **Resume**: Continue migration when confident

### Common Issues & Fixes

**Issue**: New workflow not triggering
- **Check**: Branch protection rules
- **Fix**: Update required status checks

**Issue**: Secrets not accessible
- **Check**: Secret names in workflow
- **Fix**: Verify secrets are passed correctly

**Issue**: Artifacts missing
- **Check**: Upload/download artifact steps
- **Fix**: Ensure artifact names match

**Issue**: Different output
- **Check**: Build arguments, versions
- **Fix**: Align configurations

---

## ✅ Final Cleanup Commands

**After successful migration (Week 4+):**

```bash
cd /home/jdyer/Development/nv-ingest/.github

# Archive old workflows
mv workflows/*.yml.old workflows-old/

# Or delete if confident
# rm -rf workflows-old/

# Commit changes
git add .
git commit -m "Complete GitHub Actions refactoring - archive old workflows"
git push
```

---

## 📞 Need Help?

If you encounter issues during migration:
1. Check workflow logs in Actions tab
2. Review WORKFLOWS_MIGRATION.md troubleshooting section
3. Compare old vs new workflow configurations
4. Contact DevOps team
5. Don't hesitate to rollback if needed

**Remember**: It's okay to take your time with migration. Stability is more important than speed.

---

**Last Updated**: January 6, 2025  
**Status**: Ready for migration  
**Rollback**: Always possible via .old files
