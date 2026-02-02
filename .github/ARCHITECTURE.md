# GitHub Actions Architecture

Complete system architecture documentation for nv-ingest CI/CD pipeline.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRIGGER EVENTS                           │
│  PR │ Push:main │ Schedule │ Manual │ Branch:release/*      │
└──┬────────┬──────────┬────────┬──────────────┬──────────────┘
   │        │          │        │              │
   ▼        ▼          ▼        ▼              ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────────────┐
│  PR  │ │ Main │ │Night │ │Docs  │ │  Release Workflows   │
│      │ │  CI  │ │  ly  │ │      │ │ Docker│Conda│PyPI   │
└──┬───┘ └───┬──┘ └───┬──┘ └───┬──┘ └───┬──────┬─────┬─────┘
   │         │        │        │        │      │     │
   └─────────┴────────┴────────┴────────┴──────┴─────┘
                      │
                      ▼
         ┌────────────────────────────────────┐
         │    REUSABLE WORKFLOWS LAYER        │
         │  ┌──────────┐  ┌──────────────┐   │
         │  │  Docker  │  │    Conda     │   │
         │  │  Build   │  │    Build     │   │
         │  │  Test    │  │   Publish    │   │
         │  └────┬─────┘  └──────┬───────┘   │
         │       │               │            │
         │  ┌────┴─────┐  ┌──────┴───────┐   │
         │  │   PyPI   │  │ Integration  │   │
         │  │  Build   │  │    Test      │   │
         │  │ Publish  │  │              │   │
         │  └────┬─────┘  └──────┬───────┘   │
         └───────┼────────────────┼───────────┘
                 │                │
                 ▼                ▼
         ┌────────────────────────────────────┐
         │    COMPOSITE ACTIONS LAYER         │
         │  ┌──────────┐  ┌──────────────┐   │
         │  │  Setup   │  │   Docker     │   │
         │  │  Docker  │  │   Login      │   │
         │  │ Buildx   │  │    NGC       │   │
         │  └──────────┘  └──────────────┘   │
         │  ┌──────────────────────────────┐ │
         │  │    Determine Version         │ │
         │  └──────────────────────────────┘ │
         └────────────────────────────────────┘
```

---

## 🔄 Workflow Execution Flows

### Pull Request Flow

```
PR Opened/Updated
    │
    ▼
┌───────────────────────────────────────────────────┐
│         ci-pull-request.yml                       │
├───────────────────────────────────────────────────┤
│                                                   │
│  ┌──────────────┐                                │
│  │ pre-commit   │ ◄─── Fast fail (5 min)         │
│  └──────┬───────┘                                │
│         │ ✓                                       │
│         ▼                                         │
│  ┌──────────────┐     ┌────────────────────┐    │
│  │ docker-build │────▶│  Reusable:         │    │
│  │   (amd64)    │     │  docker-build.yml  │    │
│  └──────┬───────┘     └────────────────────┘    │
│         │ ✓                                       │
│         ▼                                         │
│  ┌──────────────┐     ┌────────────────────┐    │
│  │ docker-test  │────▶│  Reusable:         │    │
│  │   + coverage │     │  docker-test.yml   │    │
│  └──────┬───────┘     └────────────────────┘    │
│         │ ✓                                       │
│         ▼                                         │
│  ┌──────────────────────────────────────┐       │
│  │ library-mode-test (if approved)      │       │
│  │   ┌──────────┐    ┌──────────────┐  │       │
│  │   │  Build   │───▶│   Test       │  │       │
│  │   │  Conda   │    │ Integration  │  │       │
│  │   └──────────┘    └──────────────┘  │       │
│  └──────────────────────────────────────┘       │
│         │                                         │
│         ▼                                         │
│  ┌──────────────┐                                │
│  │   Summary    │ ◄─── All results               │
│  │  (required)  │                                │
│  └──────────────┘                                │
└───────────────────────────────────────────────────┘
         │
         ▼
    PR Status: ✓ or ✗
```

### Nightly Build Flow

```
Schedule: 23:30 UTC (or Push to main or Manual)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│          scheduled-nightly.yml                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────┐             │
│  │   Determine Version (YYYY.MM.DD)      │             │
│  └─────────────┬─────────────────────────┘             │
│                │                                         │
│                ▼                                         │
│  ┌─────────────┴────────────┬──────────────────────┐   │
│  │                           │                      │   │
│  ▼                           ▼                      ▼   │
│  ┌────────────┐    ┌────────────────┐    ┌────────┐   │
│  │   Docker   │    │     Conda      │    │  PyPI  │   │
│  │            │    │                │    │        │   │
│  │  Multi-    │    │  Build ───▶    │    │ Build  │   │
│  │ Platform   │    │        Publish │    │   │    │   │
│  │  Build +   │    │                │    │   ▼    │   │
│  │  Push NGC  │    │   To dev       │    │Publish │   │
│  │            │    │   channel      │    │Artif.  │   │
│  └────────────┘    └────────────────┘    └────────┘   │
│        │                    │                   │       │
│        └────────────────────┴───────────────────┘       │
│                             │                           │
│                             ▼                           │
│                    ┌──────────────┐                     │
│                    │   Summary    │                     │
│                    │   Report     │                     │
│                    └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
                             │
                             ▼
        All artifacts published with version YYYY.MM.DD
```

### Release Flow

```
Create release/X.Y.Z branch
    │
    └─────────────────┬──────────────────┬────────────────┐
                      │                  │                │
    Automatic trigger for ALL THREE:    │                │
                      │                  │                │
                      ▼                  ▼                ▼
                ┌────────┐         ┌──────────┐    ┌──────────┐
                │ Docker │         │  Conda   │    │   PyPI   │
                │Release │         │ Release  │    │ Release  │
                └────┬───┘         └─────┬────┘    └─────┬────┘
                     │                   │               │
                     │ All run in parallel               │
                     │                   │               │
                     ▼                   ▼               ▼
                     │                   │               │
                     │   Uses reusable workflows:        │
                     │   - docker-build.yml              │
                     │   - conda-build.yml ──────────────┘
                     │   - conda-publish.yml
                     │   - pypi-build.yml ────────────────────┘
                     │   - pypi-publish.yml
                     │
                     ▼
    All artifacts published to respective registries with version X.Y.Z

Alternative: Manual trigger for individual releases with custom options
```

---

## 🔗 Component Dependencies

### Docker Build Chain

```
Any Workflow
    │
    ├─ Calls: docker-build.yml
    │       │
    │       ├─ Uses: setup-docker-buildx (action)
    │       │       └─ Sets up QEMU (if needed)
    │       │       └─ Sets up Buildx
    │       │
    │       ├─ Uses: docker-login-ngc (action)
    │       │       └─ Authenticates with NGC
    │       │
    │       └─ Returns: image-digest
    │
    └─ Calls: docker-test.yml
            └─ Uses: built image from docker-build.yml
            └─ Returns: test results + coverage
```

### Package Build & Publish Chain

```
Release/Nightly Workflow
    │
    ├─ Conda Path:
    │   ├─ Calls: conda-build.yml
    │   │       └─ Uses: determine-version (action)
    │   │       └─ Uploads: artifacts
    │   │
    │   └─ Calls: conda-publish.yml
    │           └─ Downloads: artifacts
    │           └─ Publishes: to channel
    │
    └─ PyPI Path:
        ├─ Calls: pypi-build.yml
        │       └─ Uses: determine-version (action)
        │       └─ Uploads: wheels
        │
        └─ Calls: pypi-publish.yml
                └─ Downloads: wheels
                └─ Publishes: to Artifactory
```

---

## 📊 System Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────┐
│            MAIN WORKFLOWS (Layer 1)             │
│                                                 │
│  7 workflows that respond to GitHub events:    │
│  - Pull request validation                     │
│  - Main branch CI                              │
│  - Scheduled nightly builds                    │
│  - Release automation (3x)                     │
│  - Documentation deployment                    │
│                                                 │
│  Purpose: Orchestration and event handling     │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│        REUSABLE WORKFLOWS (Layer 2)             │
│                                                 │
│  7 reusable components for business logic:     │
│  - Docker build                                │
│  - Docker test                                 │
│  - Conda build & publish                       │
│  - PyPI build & publish                        │
│  - Integration testing                         │
│                                                 │
│  Purpose: Reusable business logic              │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         COMPOSITE ACTIONS (Layer 3)             │
│                                                 │
│  3 actions for common operations:              │
│  - Docker Buildx setup                         │
│  - NGC authentication                          │
│  - Version determination                       │
│                                                 │
│  Purpose: Shared utilities                     │
└─────────────────────────────────────────────────┘
```

### Benefits of This Architecture

- ✅ **DRY Principle** - Each operation defined once
- ✅ **Clear Separation** - Orchestration vs logic vs utilities
- ✅ **Easy Maintenance** - Change logic in one place
- ✅ **Better Testing** - Test components independently
- ✅ **Type Safety** - Defined inputs/outputs with validation

---

## 📈 Execution Timeline Examples

### Pull Request (Total: ~15-20 minutes)

```
0:00 ─ Pre-commit checks start
       │
0:05 ─ Pre-commit complete ✓
       │
       └─ Docker build starts (amd64)
          │
0:12 ─────┘ Docker build complete ✓
          │
          └─ Docker test starts (with coverage)
             │
0:17 ────────┘ Docker test complete ✓
             │
             └─ [If approved] Library mode tests start
                │
0:47 ───────────┘ Integration tests complete ✓

Total: ~20 minutes (or ~50 min with integration tests)
```

### Nightly Build (Total: ~45-60 minutes)

```
23:30 ─ Scheduled trigger
        │
        ├─ Version determination (YYYY.MM.DD)
        │
        ├──────────────┬──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
    Docker        Conda Build    PyPI Build    (parallel)
    Multi-        │              │
    Platform      ▼              ▼
    Build     Conda Publish  PyPI Publish
        │
00:15 ──┴─ All jobs complete ✓

Total: ~45 minutes (parallel execution)
```

### Release Process (Total: ~40 minutes)

```
Create release/25.4.0
        │
        ├──────────────┬──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
    Docker        Conda          PyPI         (all parallel)
    Release       Release        Release
        │              │              │
        │              │              │
00:40 ──┴──────────────┴──────────────┴─ All complete ✓

Total: ~40 minutes (parallel execution)
```

---

## 🔐 Security Architecture

### Access Control Flow

```
┌─────────────────────────────────────────┐
│   External Contributor PR               │
└─────────────┬───────────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Check association   │
    │ MEMBER/COLLAB/OWNER?│
    └──────┬──────┬───────┘
           │      │
      NO   │      │  YES
           │      │
           ▼      ▼
    ┌──────────┐ ┌─────────────────┐
    │  Check   │ │  Run all tests  │
    │  Label   │ │   immediately   │
    │ok-to-test│ └─────────────────┘
    └────┬─────┘
         │
    ┌────┴────┐
    │  YES    │  NO
    ▼         ▼
┌────────┐  ┌────────┐
│  Run   │  │  Skip  │
│ Tests  │  │ Tests  │
└────────┘  └────────┘
```

### Secret Management

- Secrets passed explicitly to reusable workflows
- No implicit secret access
- Minimal permissions (least privilege principle)
- `pull_request_target` used safely with access checks

---

## 📚 Documentation Structure

```
.github/
    │
    ├─ README.md ◄───────────── Start here
    │   └─ Points to all other docs
    │
    ├─ INDEX.md ◄───────────────── Complete index
    │   └─ Documentation navigation guide
    │
    ├─ WORKFLOWS_QUICKSTART.md ◄─ For developers
    │   └─ Quick reference, common tasks
    │
    ├─ WORKFLOWS_REFERENCE.md ◄─── Complete reference
    │   └─ All workflows, inputs, outputs, secrets
    │
    └─ ARCHITECTURE.md ◄────────── This file
        └─ System design and architecture
```

---

## 💡 Design Principles

### 1. Separation of Concerns
- **Main workflows**: Event handling and orchestration
- **Reusable workflows**: Business logic and operations
- **Composite actions**: Common utilities

### 2. Single Source of Truth
- Docker build logic exists in one place
- Version determination centralized
- Authentication handled consistently

### 3. Type Safety
- Inputs/outputs explicitly defined
- Required vs optional parameters clear
- Validation at workflow boundaries

### 4. Parallel Execution
- Independent jobs run simultaneously
- Nightly builds publish in parallel
- Release workflows trigger together

### 5. Fail Fast
- Pre-commit checks run first
- Quick validation before expensive operations
- Clear error reporting

---

## 🎓 Key Concepts

### Reusable Workflows

Workflows that can be called from other workflows:

```yaml
jobs:
  build:
    uses: ./.github/workflows-reusable/docker-build.yml
    with:
      platform: 'linux/amd64'
      push: false
    secrets:
      HF_ACCESS_TOKEN: ${{ secrets.HF_ACCESS_TOKEN }}
```

**Benefits:**
- Define once, use many times
- Type-safe interfaces
- Centralized logic

### Composite Actions

Custom actions combining multiple steps:

```yaml
- uses: ./.github/actions/setup-docker-buildx
  with:
    use-qemu: 'true'
    platforms: 'linux/amd64,linux/arm64'
```

**Benefits:**
- Reusable across workflows
- Consistent setup steps
- Easy to maintain

### Version Determination

Smart version extraction from multiple sources:

```yaml
- uses: ./.github/actions/determine-version
  with:
    branch-name: ${{ github.ref }}
```

**Logic:**
1. Check explicit version input
2. Extract from branch name (`release/X.Y.Z`)
3. Generate from date (`YYYY.MM.DD`)

---

## 🔄 Data Flow

### Artifact Flow

```
Build Job
    └─ Uploads: artifacts
            │
            ├─ conda-packages
            ├─ python-wheels
            └─ test-artifacts
                    │
                    ▼
            Download Job
                └─ Uses artifacts for:
                    - Publishing
                    - Testing
                    - Deployment
```

### Secret Flow

```
Repository Secrets
    │
    ├─ Main Workflow
    │   └─ Passes to Reusable Workflow
    │       └─ Uses in steps
    │
    └─ Direct to Composite Actions
        └─ Uses for authentication
```

---

**This architecture provides:**
- ✅ Clear separation of concerns
- ✅ Maximum reusability
- ✅ Easy maintenance
- ✅ Type-safe interfaces
- ✅ Comprehensive documentation
- ✅ Scalable design
