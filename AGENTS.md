# Repository Guidelines

## Project Structure
This repo centers on [naasii-modeling.py](/home/perkinsd/Naasii-modeling/naasii-modeling.py), a single Marimo notebook containing narrative cells, reactive analysis, helper functions, and notebook-local tests. [README.md](/home/perkinsd/Naasii-modeling/README.md) documents the workflow. `LICENSE` covers distribution terms. Local skills live in `.agents/skills/`; [`.claude/`](/home/perkinsd/Naasii-modeling/.claude) intentionally mirrors them for Claude. There is no `tests/`, `assets/`, or package directory.

## Build, Test, and Development Commands
```bash
uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
uv run --with marimo[recommended] marimo run --sandbox naasii-modeling.py
uv run --with-requirements naasii-modeling.py pytest naasii-modeling.py
python3 -m py_compile naasii-modeling.py
```
Use `edit` for authoring, `run` for app-mode behavior, `pytest` for notebook-local tests, and `py_compile` for a fast syntax check.

## Notebook Conventions
Target Python 3.13+ with 4-space indentation. Use `snake_case` for functions and locals and `UPPER_CASE` for constants such as `SIDES` and `TRIAL_STEPS`. Keep reusable logic in helper cells near the bottom of the notebook and pass dependencies explicitly through cell arguments. Prefer ordinary cells over `app.setup(...)`. Avoid duplicate exported names across cells, especially `fig`, `ax`, and `rng`.

Prefer built-in Marimo widgets when they fit. Ask before adding a custom widget unless the user explicitly requested one. Suggest a custom widget only when it provides a clear advantage.

## Testing Guidelines
Tests live in notebook test cells, not a separate `tests/` tree. Keep test cells limited to `test_*` functions or test classes. Group related tests into one cell when that makes Marimo’s per-cell output easier to read. Favor deterministic contract tests for notebook helpers. When notebook logic changes, run the documented pytest command and manually exercise the affected sliders and plots.

## Commit and PR Guidelines
Use short imperative sentence-case commit messages, for example `Add initial roll_d12s tests`. Pull requests should summarize notebook behavior changes, list the validation commands run, mention Marimo test results when tests changed, and include screenshots for UI or plot changes.

## Skills
Repo-local skills are the source of truth for generic framework guidance. Use `marimo-notebook` for notebook structure and reactivity questions, `wasm-compatibility` before making WASM or Pyodide claims, and `anywidget-generator` only when a custom widget is actually warranted.
