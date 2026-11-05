# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ESE 5420 (Statistics for Data Science, Fall 2026) solo final project: predicting whether an unplanned New York hospital admission lasts more than 7 days (NYSDOH SPARCS inpatient discharges, 2023–24 pooled), comparing course models on AUC and calibration, with a fairness check by payer and race. Deliverable is a ≤6-page NeurIPS-format paper plus a 5-minute video, due Fri 11 Dec 2026; early draft due Fri 6 Nov 2026.

Analysis code lives in `analysis/`; `analysis/verify_data.py` downloads the raw CSVs into `data/raw/` (gitignored) and checks the proposal's data claims.

## Docs

- `docs/notes/` — dated project notes (see below). Start with `2026-10-30-original-analysis-plan.md`; check its "Open decisions" list before choosing subsample size, race/payer groupings, or facility exclusions.
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

New machine: install uv, clone, run `uv sync`, then `uv run python analysis/verify_data.py` to re-download `data/`. Accept VS Code's prompt to install the recommended extension.

## Notebooks

Notebooks are paired with percent-format `.py` files of the same name through `[tool.jupytext]` in `pyproject.toml`. Plain scripts such as `analysis/verify_data.py` have no pair and stay plain scripts.

- Edit the `.py` file, using `# %%` for code cells and `# %% [markdown]` for markdown cells, then sync so the `.ipynb` matches. Never let the two drift.
- Only the `.ipynb` holds outputs, and syncing keeps old outputs, so re-execute after changing code.
- Commit the `.py` and `.ipynb` together.

```bash
uv run jupytext --sync analysis/<name>.py                 # sync .py -> .ipynb (pass the .ipynb to sync the other way)
uv run jupytext --to ipynb --set-kernel python3 --execute --run-path . analysis/<name>.py   # rebuild and execute end to end
uv run python analysis/<name>.py                          # quick run without Jupyter
```

In the execute command, `--set-kernel python3` uses the kernel inside `.venv`, so nothing needs registering per machine, and `--run-path .` runs from the repo root, matching VS Code.

## Git

- Commit and push directly to `main`, unless the user has said to work on a separate branch.
- On a separate branch, push only to that branch, and don't merge it until the user says to.

## Rules from the analysis plan that code must respect

These apply unless a newer note changes them.

- **Scope:** keep emergency, urgent and trauma admissions; drop elective, newborn and "Not Available" admission types.
- **Target:** `y = 1` if length of stay ≥ 8. LOS is text with a top value of `"120 +"` (2023) or `"120+"` (2024).
- **Pooling 2023 + 2024:** rename `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`; strip thousands commas from 2023 charge columns; add a year indicator.
- **Never use as features (leakage):** total charges/costs, patient disposition, APR severity of illness, APR risk of mortality, APR-DRG, CCSR procedure, birth weight.
- **Encoding:** all features are categorical — one-hot with rare levels grouped into "Other", never integer label-encoding; fit encoders on training data only.
- **Splits:** random 80/20 train/test stratified on the target; K-fold CV on train for tuning; a held-back training slice for recalibration (Platt scaling and isotonic regression); the test set is used once, and all models are scored on the same test rows.
- Every model outputs probabilities. AUC above ~0.9 suggests leakage (published benchmarks are 0.75–0.81).
- Credit NYSDOH SPARCS; no State logos or seals in the paper.
