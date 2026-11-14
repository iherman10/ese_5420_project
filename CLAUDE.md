# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ESE 5420 (Statistics for Data Science, Fall 2026) solo final project: predicting whether an unplanned New York hospital admission lasts more than 7 days (NYSDOH SPARCS inpatient discharges, 2023–24 pooled), comparing course models on AUC and calibration, with a fairness check by payer and race. Deliverable is a ≤6-page NeurIPS-format paper plus a 5-minute video.

Analysis code lives in `analysis/`; `analysis/verify_data.py` downloads the raw CSVs into `data/raw/` (gitignored) and checks the proposal's data claims. `analysis/common.py` holds the steps every model shares (data, cohort, split, cleaning, encoder, CV folds, saved results); model notebooks import it with `from analysis import common` and must run from the repo root.

## Docs

- `docs/notes/` — dated project notes (see below). Read `2026-10-30-original-analysis-plan.md` and any newer notes before writing analysis code or making analysis decisions.
- `docs/course_materials/proposal.md` — submitted proposal.
- `docs/course_materials/proposal_v2.txt` — re-submitted proposal in the topic-guide template (topics 20 and 23).
- `docs/course_materials/topic_guide.md` — instructor's research topic guide and project rules.
- `docs/course_materials/project_requirements.md` — course rules (methods must come from course modules; paper needs an AI-use statement in the acknowledgements).
- `docs/notes/2026-10-14-project-guide.html` is generated and gitignored.

## Notes

Before implementing the next part of the project, record its key decisions in a new note.

- Put notes directly under `docs/notes/`, with no subfolders, as Markdown files named `YYYY-MM-DD-short-topic.md`.
- Start each note with this header; everything below it is free-form:

  ```markdown
  # Title
  Created: YYYY-MM-DD
  Scope: What this note covers.
  ```

- Notes are a dated record, not a living spec. When a decision changes, write a new note rather than revising the old one; the newest note on a topic wins.

## Environment

Python 3.13, managed with uv. Dependencies are added as needed.

```bash
uv sync                      # create/update .venv
uv add <pkg>                 # add a runtime dependency
uv run python script.py
```

VS Code and Jupyter use `.venv/bin/python` with the workspace root as the notebook working directory.

New machine: install uv, and on macOS run `brew install libomp` (XGBoost needs the OpenMP library, which uv can't install). Clone, run `uv sync`, then `uv run python analysis/verify_data.py` to re-download `data/`. Accept VS Code's prompt to install the recommended extension.

## Notebooks

Notebooks are paired with percent-format `.py` files of the same name through `[tool.jupytext]` in `pyproject.toml`. Plain scripts such as `analysis/verify_data.py` have no pair and stay plain scripts.

- Edit the `.py` file, using `# %%` for code cells and `# %% [markdown]` for markdown cells, then sync so the `.ipynb` matches. Never let the two drift.
- Only the `.ipynb` holds outputs, and syncing keeps old outputs, so re-execute after changing code.
- Commit the `.py` and `.ipynb` together.

```bash
uv run jupytext --sync analysis/<name>.py                 # sync .py -> .ipynb (pass the .ipynb to sync the other way)
uv run jupytext --to ipynb --set-kernel python3 --execute --run-path . analysis/<name>.py   # rebuild and execute end to end
uv run python -m analysis.<name>                          # quick run without Jupyter
```

In the execute command, `--set-kernel python3` uses the kernel inside `.venv`, so nothing needs registering per machine, and `--run-path .` runs from the repo root, matching VS Code.

## Git

- Commit and push directly to `main`, unless the user has said to work on a separate branch.
- On a separate branch, push only to that branch, and don't merge it until the user says to.
