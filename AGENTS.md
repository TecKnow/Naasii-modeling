# Repository Guidelines

## Project Structure & Module Organization
This repo is intentionally small. The main artifact is [naasii-modeling.py](/home/perkinsd/Naasii-modeling/naasii-modeling.py), a single Marimo notebook with narrative, helper, reactive analysis, and notebook-local test cells. [README.md](/home/perkinsd/Naasii-modeling/README.md) holds the overview and launch command, and `LICENSE` covers distribution terms. Local skills live under `.agents/skills/`, with versions recorded in `skills-lock.json`. The [`.claude/`](/home/perkinsd/Naasii-modeling/.claude) directory is intentional: it mirrors those skills for Claude via symlinks and should be kept in sync with `.agents/`. There is no `tests/`, `assets/`, or package directory, so keep related logic in the notebook unless a new file clearly simplifies the project.

## Build, Test, and Development Commands
Use these commands for local work:

```bash
uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
```
Edit the notebook.

```bash
uv run --with marimo[recommended] marimo run --sandbox naasii-modeling.py
```
Run it in app mode.

```bash
python3 -m py_compile naasii-modeling.py
```
Quick syntax check before committing. If notebook logic changes, also run or inspect the Marimo test cells and exercise the sliders and plots.

## Coding Style & Naming Conventions
Target Python 3.13+ and use 4-space indentation. Use `snake_case` for functions and locals and `UPPER_CASE` for constants like `SIDES` and `TRIAL_STEPS`. Keep reusable logic in helper cells near the bottom of the notebook, with short docstrings where useful, and pass dependencies explicitly through cell arguments. Prefer ordinary cells over `app.setup(...)`. Avoid duplicate top-level names such as `fig`, `ax`, or `rng` across cells. Prefer built-in Marimo widgets; ask before adding a custom widget unless the user explicitly requested one. Suggest a custom widget only when it offers a clear advantage.

## Testing Guidelines
`pytest` is pinned in the notebook’s PEP 723 metadata and used through Marimo test cells. Keep tests in cells that contain only `test_*` functions or test classes. Group closely related tests into one cell when that makes Marimo’s per-cell output easier to read. Prefer deterministic contract tests for notebook helpers.

## Commit & Pull Request Guidelines
Use short imperative sentence-case commit messages, for example `Use log-spaced trial sliders` or `Add initial roll_d12s tests`. Pull requests should summarize notebook behavior changes, list the validation commands you ran, mention Marimo test results when tests changed, and include screenshots for UI or plot changes.

## Local Skills
Use repo-local skills when they match the task:
- `marimo-notebook` for notebook structure, reactivity, dependency metadata, and checks
- `wasm-compatibility` before claiming WASM, Pyodide, or playground support
- `anywidget-generator` only when a custom widget is warranted; prefer built-ins first
