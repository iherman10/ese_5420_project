# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ESE 5420 (Statistics for Data Science, Fall 2026) solo final project: predicting whether an unplanned New York hospital admission lasts more than 7 days (NYSDOH SPARCS inpatient discharges, 2023–24 pooled), comparing course models on AUC and calibration, with a fairness check by payer and race. Deliverable is a ≤6-page NeurIPS-format paper plus a 5-minute video, due Fri 11 Dec 2026; early draft due Fri 6 Nov 2026.

No analysis code exists yet — the repo is currently docs plus an empty Python environment.

## Docs

- `docs/analysis_plan.md` — **the authoritative spec.** Where it disagrees with `docs/project_ideas/hospital_stays/` (background research and sources), the plan wins. Check its "Open decisions" list before choosing subsample size, race/payer groupings, or facility exclusions.
- `docs/proposal.md` — submitted proposal.
- `docs/project_requirements.md` — course rules (methods must come from course modules; paper needs an AI-use statement in the acknowledgements).
- `docs/project_ideas/project_guide.html` is generated and gitignored.

## Environment

Python 3.13, managed with uv. Dependencies are added as needed.

```bash
uv sync                      # create/update .venv
uv add <pkg>                 # add a runtime dependency
uv run python script.py
```

VS Code and Jupyter use `.venv/bin/python` with the workspace root as the notebook working directory.

## Rules from the analysis plan that code must respect

- **Scope:** keep emergency, urgent and trauma admissions; drop elective, newborn and "Not Available" admission types.
- **Target:** `y = 1` if length of stay ≥ 8. LOS is text with a top value of `"120+"`.
- **Pooling 2023 + 2024:** rename `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`; strip thousands commas from 2023 charge columns; add a year indicator.
- **Never use as features (leakage):** total charges/costs, patient disposition, APR severity of illness, APR risk of mortality, APR-DRG, CCSR procedure, birth weight.
- **Encoding:** all features are categorical — one-hot with rare levels grouped into "Other", never integer label-encoding; fit encoders on training data only.
- **Splits:** random 80/20 train/test stratified on the target; K-fold CV on train for tuning; a held-back training slice for recalibration (Platt scaling and isotonic regression); the test set is used once, and all models are scored on the same test rows.
- Every model outputs probabilities. AUC above ~0.9 suggests leakage (published benchmarks are 0.75–0.81).
- Credit NYSDOH SPARCS; no State logos or seals in the paper.
