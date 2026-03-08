# Repository Guidelines

## Project Structure & Module Organization
This repository is intentionally small. The main artifact is [naasii-modeling.py](/home/perkinsd/Naasii-modeling/naasii-modeling.py), a Marimo notebook script that contains both reusable probability helpers and reactive notebook cells. [README.md](/home/perkinsd/Naasii-modeling/README.md) provides the project overview, game-resource links, and the canonical local run command. `LICENSE` covers distribution terms. There is no dedicated `tests/`, `assets/`, or package directory yet, so keep related logic grouped inside the notebook and add new files only when they meaningfully reduce complexity.

## Build, Test, and Development Commands
Use the notebook’s documented command for local work:

```bash
uv run --with marimo[recommended] marimo edit --sandbox naasii-modeling.py
```

This launches the Marimo editor with dependencies resolved from the script metadata. For a fast syntax check before committing, run:

```bash
python3 -m py_compile naasii-modeling.py
```

If you add more files later, prefer lightweight validation commands that do not mutate tracked files.

## Coding Style & Naming Conventions
Target Python 3.13+ and use 4-space indentation. Follow standard Python naming: `snake_case` for functions and locals, `UPPER_CASE` for module-level constants such as `SIDES` or `TRIAL_STEPS`. Keep reusable notebook logic in `@app.function` helpers with short docstrings. In Marimo cells, avoid duplicate top-level names like `fig`, `ax`, or `rng` across cells; Marimo treats those as conflicting definitions. Favor explicit, deterministic cell outputs over hidden state.

## Testing Guidelines
There is no automated test framework yet. Minimum validation for changes is:

1. `python3 -m py_compile naasii-modeling.py`
2. Launch the notebook with the `uv run ... marimo edit --sandbox ...` command
3. Exercise the sliders and confirm plots and markdown update cleanly
4. Check browser DevTools for Marimo errors such as `multiple-defs` or `NameError`

## Commit & Pull Request Guidelines
Recent commits use short imperative messages in sentence case, for example `Use log-spaced trial sliders` and `Expand introductory statistics notebook`. Follow that pattern. Pull requests should summarize the notebook behavior changes, list the validation commands you ran, and include screenshots when UI layout, plots, or control behavior changed.
