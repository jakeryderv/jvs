# development tooling for `jvs`

| Concern | Tool | Role |
|---|---|---|
| Python/version management | **uv** | Python installs, `.venv`, dependencies, lockfile |
| Packaging | **uv** | build + publish |
| Formatting | **Ruff** | formatter |
| Linting | **Ruff** | lint rules, imports, common bugs |
| Type checking | **ty** | static typing |
| Editor diagnostics | **ty LSP** | type-aware IDE feedback |
| Testing | **pytest** | unit/integration tests |
| Coverage | **coverage.py** | measure what tests exercise |
| CI | **GitHub Actions** | run everything across pushes/PRs |
