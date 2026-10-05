# Development

## Directory conventions

Documentation and tests mirror the directories beneath `src/jvs/`:

| Implementation | Documentation | Tests |
| --- | --- | --- |
| `src/jvs/_cli/` | Add when useful | `tests/_cli/` |

Use `docs/<subpackage>/README.md` as the documentation entry point. Add further
documents by subject and tests as `test_<subject>.py`; a source module can have
several focused test files. Create documentation and test directories when they
have content. Repository guidance stays directly in `docs/`, and exploratory
notes stay in `docs/notes/`.

Pytest searches recursively beneath `tests/`. The configured `importlib` mode
allows repeated test filenames in different subpackages without adding
`__init__.py` files to test directories. Tests import the installed `jvs` package
using absolute imports. Keep fixtures local to the subpackage that needs them;
use a shared `tests/conftest.py` when fixtures are needed across subpackages.

## Setup and checks

Run from the repository root:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Keep `uv run ty check` project-wide so it also checks consumers of changed code.

## CLI and terminal libraries

- `cyclopts`: CLI command parsing.
- `rich`: Terminal rendering and formatting.
- `textual`: TUI framework.
- `prompt_toolkit`: Interactive prompts and REPL behavior.

## Tooling reference

The core development stack is uv, Ruff, ty, and pytest. The table also includes
optional tools to consider as the project grows.

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

## Releases

Use `utils/release.sh` from inside the repository with `git`, `uv`, `gh`, and
`tar` installed. Authenticate with `gh auth login` first. Local `main` must match
`origin/main`, and all changes, including untracked files, must be committed or
stashed before running the script.

Preview a patch release:

```bash
bash utils/release.sh --dry-run patch
```

Release after reviewing the checks and confirming the prompt:

```bash
bash utils/release.sh patch
```

Use `minor` or `major` instead of `patch` for those version bumps. Omitting the
bump prompts for it; `--help` shows usage.

Each check prints one result line, with a green `[PASS]` or red `[FAIL]` marker
in supported terminals. The running check appears on the same line until its
result replaces it. Redirected output uses plain text without terminal control
sequences; set `NO_COLOR=1` to disable colors in a terminal.

Add `--verbose` (or `-v`) to show captured output after every check:

```bash
bash utils/release.sh --dry-run --verbose patch
```

Failed checks always show their output, including without `--verbose`. The
publishing workflow is watched live in both modes.

The script checks the lockfile and runs `uv sync --locked` to prepare the
development environment. It then runs Ruff linting, formatting checks, and ty
type checking, and builds the proposed version in a temporary copy of the
committed source. Both this candidate build and the publishing workflow use
`uv build --no-sources`. Tests are not currently part of the release checks.

A dry run may create or update `.venv`, download dependencies or Python, and
update caches. It does not change source files, the lockfile, Git refs, or
releases. Temporary build files are removed when the script exits.

After confirmation, the script rechecks the branch, checked commit, working
tree, remote main, and tag availability. It updates the version and lockfile,
commits them, pushes the release commit to `origin/main`, and creates a GitHub
release. `.github/workflows/publish.yml` publishes that release to PyPI using
the configured `pypi` environment and trusted publisher.

The script polls for roughly a minute for the release-triggered workflow for that
exact commit to appear, then watches it to completion. A failed, cancelled,
skipped, or missing run exits unsuccessfully. A workflow waiting for environment
approval remains pending until it is approved or cancelled. Success means the
publishing workflow completed successfully; it is not a separate check of PyPI
index availability.

Workflow tracking uses [`gh run list`](https://cli.github.com/manual/gh_run_list)
to find the matching release commit and
[`gh run watch --exit-status`](https://cli.github.com/manual/gh_run_watch)
to monitor its result.

### Recovering an interrupted release

If the script fails or is interrupted after version changes begin, it prints
the version, tag, release commit when known, last confirmed steps, and commands
to inspect or finish the release. It leaves completed changes in place.

Do not immediately rerun the script: that would propose another version bump.
Inspect local and remote state first, because a push or release request may
have succeeded even if the command reported an error.

- If the version changed but no release commit exists, inspect the changes,
  finish the intended version and lockfile, and commit those two files.
- If the release commit exists but was not pushed, push that same commit.
- If the GitHub release is missing, create it for the existing release commit.
- If the release exists, inspect its publishing workflow and resume watching
  it. Resolve any workflow failure before deciding whether to rerun failed
  jobs; check for artifacts already published to PyPI first.

The script prints commands with the actual tag, commit, and workflow run ID
where available. It does not automatically roll back changes or retry publishing.
