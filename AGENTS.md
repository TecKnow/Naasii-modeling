# Repository Guidelines

## Project Structure
[naasii-modeling.py](/home/perkinsd/Naasii-modeling/naasii-modeling.py) is the single Marimo notebook, with narrative cells, reactive analysis, helper functions, and notebook-local tests. [README.md](/home/perkinsd/Naasii-modeling/README.md) documents workflow. Local skills live in `.agents/skills/`, and [`.claude/`](/home/perkinsd/Naasii-modeling/.claude) mirrors them for Claude.

## Build, Test, and Development Commands
Bootstrap a fresh checkout:
```bash
uv sync --frozen
```

Then use the project virtualenv:
```bash
.venv/bin/marimo edit naasii-modeling.py
.venv/bin/marimo run naasii-modeling.py
.venv/bin/marimo check naasii-modeling.py
.venv/bin/pytest naasii-modeling.py
.venv/bin/python -m py_compile naasii-modeling.py
```
`uv sync --frozen` creates and populates `.venv` from `uv.lock`; rerun `uv sync` after dependency changes. After bootstrap, prefer the project `.venv/bin/...` tools for local work and automated tooling to avoid repeated `uv` cache-permission failures in restricted environments. In this Codex sandbox, `marimo check` appears to hang on this notebook, so use `pytest` and `py_compile` for routine validation here and run `marimo check` outside the sandbox when needed. Use the `uv run --with marimo[recommended] marimo ... --sandbox` forms only when you specifically need Marimo's isolated sandbox and `uv` can write to its cache.

## Notebook Conventions
Target Python 3.13+ with 4-space indentation. Use `snake_case` for functions and locals and `UPPER_CASE` for constants such as `SIDES` and `TRIAL_STEPS`. Keep reusable logic in helper cells near the bottom and pass dependencies through cell arguments. Prefer ordinary cells over `app.setup(...)`. Avoid duplicate exported names across cells, especially `fig`, `ax`, and `rng`.

Teaching sections should be student-facing: use question-driven headings, introduce formal statistical terms in plain language before going deeper, and end larger units with a brief Naasii bridge to the next question. Narrative and UI-facing cells should usually use `hide_code=True`; show code only when it is part of the lesson.

Prefer built-in Marimo widgets when they fit. Ask before adding a custom widget unless the user explicitly requested one, and suggest a custom widget only when it is clearly better.

## Testing Guidelines
Tests live in notebook test cells, not a separate `tests/` tree. Keep them limited to `test_*` functions or test classes, and group related tests when that keeps Marimo output readable. Favor deterministic contract tests for helpers. In a fresh checkout, run `uv sync --frozen`, then `.venv/bin/pytest naasii-modeling.py`. Use the `uv run` variant only when you intentionally want isolated dependency resolution and the environment allows `uv` cache access. Manually exercise affected sliders and plots.

## Commit and PR Guidelines
Use short imperative sentence-case commit messages. Pull requests should summarize notebook behavior changes, list validation commands, mention Marimo test results when tests changed, and include screenshots for UI or plot changes.

## Skills
Repo-local skills are the source of truth for framework guidance. Use `marimo-notebook` for notebook structure and reactivity questions, `wasm-compatibility` before making WASM or Pyodide claims, and `anywidget-generator` only when a custom widget is warranted.
