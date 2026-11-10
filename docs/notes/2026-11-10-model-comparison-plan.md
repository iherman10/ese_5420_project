# Model Comparison Plan
Created: 2026-11-10
Scope: The order to build the models after the baseline, and what each model notebook must produce and save. How results are compared and reported will go in a later note.

## Features

The 17 features from `2026-11-06-cleaning-encoding-baseline.md` are final. An occlusion study, adding variables back to see how much they help, is an optional extra for the end.

## Model order

| Step | Model | Why here |
| :--- | :--- | :--- |
| 1 | Ridge, lasso, elastic net logistic regression | Same pipeline as the baseline, plus a penalty |
| 2 | Decision tree | Fast, sets up the tree encoding, readable reference model |
| 3 | Gradient-boosted trees (XGBoost) | With the baseline, this covers the proposal's fallback paper |
| 4 | Random forest | Slower, so keep the tuning grid small |
| 5 | MLP | Slowest; the first to cut if it won't finish |
| 6 | Stacking ensemble | A logistic regression on the other models' predictions, so it comes last. Optional |

## Code

- **`analysis/common.py`:** a plain module with the shared steps from `baseline_model.py`: loading, cohort, cleaning, split, feature list, encoder settings, CV folds and the load-or-compute helper. Every notebook imports it, so every model sees the same rows and features.
- **`baseline_model.py`:** changed to import from `common.py`. Its results must not change.
- **One notebook per model**, paired with an `.ipynb` as usual.

## Rules for every model notebook

- **Folds:** stratified 5-fold CV on train with the same folds as the baseline (seed 42). Stacking needs this.
- **Inside the folds:** the encoder and anything else learned from data, such as XGBoost early stopping, is fit only on each fold's training part, never on its validation fold.
- **Encoding:** `min_frequency=100` and `handle_unknown="infrequent_if_exist"` for every model. The baseline keeps `drop="first"`, which doesn't change an unpenalized model's predictions. Every other model uses full one-hot with no level dropped, because with a penalty the dropped level would change the fit.
- **Tuning:** a comparable search budget for each model. Each notebook shows its own tuning curve (CV AUC against each tuned setting) and states the chosen settings.
- **Recalibration and test sets:** never used for fitting or tuning. The notebook only predicts on them with the final model.

## What each model notebook saves

In `results/<model>/`, through the load-or-compute helper, so nothing slow runs twice:

| File | What it holds | In git? |
| :--- | :--- | :--- |
| `cv_scores.csv` | One row per fold: AUC, Brier, fit time | Yes |
| `tuning.csv` | CV AUC and Brier for every setting tried (tuned models only) | Yes |
| `tuning.png` | The tuning curve | Yes |
| `oof_predictions.parquet` | One out-of-fold risk per training stay, with its row ID | No |
| `model.joblib` | The final model, refit on all of train with the chosen settings | No |
| `recal_predictions.parquet` | The final model's raw risk for each recalibration-set stay, with its row ID | No |
| `test_predictions.parquet` | The final model's raw risk for each test stay, with its row ID. Made only at the end | No |

These predictions are enough to compute every planned score, curve and comparison later: CV AUC and Brier, calibration curves, precision and recall, Platt and isotonic recalibration (fit on the recalibration-set predictions), and test-set results overall and by payer and race. Every prediction file is checked against the current row IDs, as in `2026-11-07-saving-results.md`.

## From the `payer-calibration` branch

We don't merge it. We cherry-pick its note (`2026-11-08-group-calibration-figure.md`), the group-number fix in `docs/course_materials/baseline.txt`, and its payer and race group definitions, for later use.

## Decided

- No CV within the recalibration set. Platt and isotonic are both fit on the recalibration set for every model.

## Later

Recalibration details, how the recommended model is chosen, and the final tables and figures.
