# jvs

My personal, opinionated Python package for doing the things I want to do in the way I think they should be done.

`jvs` is modular and heterogeneous: each subpackage has its own purpose, rationale, and conventions, with shared foundations where useful.

## Install

[uv](https://docs.astral.sh/uv/) is recommended. Install in a Python [virtual environment](https://docs.astral.sh/uv/pip/environments/):

```bash
uv pip install jvs
```

Or add it to your [project](https://docs.astral.sh/uv/concepts/projects/):

```bash
uv add jvs
```

## Structure

| Directory | Purpose |
| --- | --- |
| `src/jvs/` | Subpackages and shared implementation |
| `docs/` | Package documentation, development guidance, and notes |
| `tests/` | Automated tests grouped by subpackage |
| `scripts/` and `notebooks/` | Experiments and examples |
| `utils/` | Repository maintenance tooling |

Package documentation and tests mirror subpackage directories: `src/jvs/_cli/` corresponds to `tests/_cli/`, with documentation added under `docs/_cli/` when useful. Each documented subpackage uses a `README.md` as its entry point; individual documents and tests follow their subjects.

## Development

See [development guidance](docs/development.md) for setup, directory conventions, checks, terminal tooling, and releases.
