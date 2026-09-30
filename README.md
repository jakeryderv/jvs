# jvs

> My personal, opinionated, python package to help handle the things I want to do in the way I think they should be done.

Overall, just a personal medium to do things how I want without imposing anyone else's rules or conventions. No clear purpose or rationale at the root package level beyond just shared foundational stuff . However, this is intended to be heterogeneous, meaning the subpackages retain an orthogonal purpose and rationale.

## structure

`src/jvs/` -> Root package: Modular package/subpackage structure. Root just contains shared foundational stuff (types, errors, configs, interfaces, etc).

`scripts/` & `notebooks/`-> python, to test and mess around with implementing `jvs` stuff

`tests/` -> tests for stuff in `jvs`

`justfile` & `tools/` -> helper tooling to streamline repo management type of stuff; static/dynamic code analysis stuff, tests, etc.

`docs/` -> notes and documentation

#### subpackages

| subpackage name | philosphy |
| :--- | :---: |
| `mathviz` | The visualization is a view of mathematical state, not the owner of that state. |


## tech stack

#### cli & tui

`cyclopts` for CLI command parsing, `rich` for terminal rendering/formatting, `textual` for full TUI framework, `prompt_toolkit` for interactive input / REPL / shell behavior
