# development tooling for `jvs`

| Tool | Description | Covers |
|---|---|---|
| **uv** | Python project/package manager | Python versions, virtual environments, dependencies, lockfiles, running tools, building, publishing |
| **Ruff** | Fast Python linter + formatter | Formatting, linting, import sorting, style issues, many common bugs |
| **ty** | Static type checker + language tooling | Type checking, type diagnostics, editor/LSP support |
| **pytest** | Python testing framework | Unit tests, integration tests, fixtures, parametrization, test discovery |
| **coverage.py** | Code coverage measurement | Statement/branch coverage, missing test coverage reports |
| **Hypothesis** | Property-based testing framework | Generated test inputs, edge cases, invariants, fuzz-like testing |
| **pytest-xdist** | pytest parallelization plugin | Parallel/distributed test execution |
| **pytest-benchmark** | pytest benchmarking plugin | Repeatable performance benchmarks |
| **pip-audit** | Python dependency vulnerability scanner | Known CVEs/security issues in Python dependencies |
| **OSV-Scanner** | General dependency vulnerability scanner | Vulnerabilities across lockfiles/packages/ecosystems |
| **pre-commit** | Git hook framework | Automatically running Ruff, ty, tests, etc. before commits |
| **GitHub Actions** | CI/CD platform | Automated linting, typing, testing, builds, publishing, version matrices |
| **MkDocs** | Documentation site generator | Project documentation, guides, API/reference sites |
| **Sphinx** | Advanced Python documentation system | API docs, cross-references, larger technical documentation |
| **timeit** | Python standard-library benchmark utility | Simple microbenchmarking |
| **tox** | Environment/test matrix manager | Testing across multiple Python versions/configurations; often unnecessary with uv + CI |

**Core stack:** `uv + Ruff + ty + pytest`

**Good additions:** `coverage.py + Hypothesis + GitHub Actions`

Everything else is more situational.
