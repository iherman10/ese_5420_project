# Model Comparison Plan
Created: 2026-11-10
Scope: The plan after the baseline checkpoint: the order to build the models, how the code and saved results are organized, how recalibration fits in, and what we report for every model so they can be compared.

## Features

The 17 features from `2026-11-06-cleaning-encoding-baseline.md` are final. The diagnosis-feature questions in `2026-11-07-diagnosis-feature-review.md` aren't pursued now. If time allows at the end, an optional occlusion study could add variables back to see how much they help, but it isn't the point of the paper.

## Model order

| Step | Model | Why here |
| :--- | :--- | :--- |
| 0 | Evaluation and recalibration code, on the baseline | Every later model reuses it |
| 1 | Ridge, lasso, elastic net logistic regression | Same pipeline as the baseline, plus a penalty |
| 2 | Decision tree | Fast, sets up the tree encoding, readable reference model |
| 3 | Gradient-boosted trees (XGBoost) | With the baseline, this covers the proposal's fallback paper |
| 4 | Random forest | Slower, so keep the tuning grid small |
| 5 | MLP | Slowest and least certain; the first to cut if it won't finish |
| 6 | Stacking ensemble | A logistic regression on the other models' predictions, so it comes last. Stacking isn't a named course module, so the paper must justify it, or we drop it |

- Every model is tuned with stratified 5-fold CV on train, using the **same folds** (seed 42) as the baseline. Stacking needs this.
- Give each model a comparable tuning budget, as the course expects.

## Code

- **`analysis/common.py`:** a plain module with the shared steps from `baseline_model.py`: loading, cohort, cleaning, split, feature list, encoder settings, folds, and the evaluation and recalibration helpers. Every notebook imports it, so every model sees the same rows and features.
- **One notebook per model**, such as `analysis/penalized_logistic.py`, paired with an `.ipynb`.
- **One comparison notebook** that reads every model's saved results and builds every table and figure. Model notebooks don't make comparison tables or figures.
- `baseline_model.py` is changed to import from `common.py` instead of repeating the code, and its results must not change.

## Encoding

- **Baseline:** keeps `drop="first"`. For an unpenalized model, which level is dropped doesn't change the predictions, so its results stand.
- **Every other model:** full one-hot, with no level dropped. With a penalty, dropping a level makes the fit depend on which level was dropped, and trees don't need a reference level.
- **Everyone:** `min_frequency=100` and `handle_unknown="infrequent_if_exist"`, as before. The encoder is fit inside each fold.
- The paper says this in one sentence.

## Saved results

Each model gets a folder `results/<model>/`, saved with the existing load-or-compute helper. We save anything slow to recompute.

| File | What it holds | In git? |
| :--- | :--- | :--- |
| `cv_scores.csv` | One row per fold: AUC, Brier, fit time, chosen hyperparameters | Yes |
| `tuning.csv` | CV score for every hyperparameter setting tried (tuned models only) | Yes |
| `oof_predictions.parquet` | One out-of-fold risk per training stay, with its row ID | No |
| `model.joblib` | The final model, refit on all of train with the chosen settings | No |
| `recal_predictions.parquet` | The final model's raw risk for each recalibration-set stay | No |
| `test_predictions.parquet` | The final model's raw risk for each test stay, made only at the final stage | No |

The stale-results check from `2026-11-07-saving-results.md` applies to every prediction file.

## Recalibration

**What it is:** a model can rank patients well but get the percentages wrong. For example, of the patients it gives 30%, 40% might stay long. Recalibration learns a correction from the model's raw risk to the real rate. It doesn't change who is ranked above whom.

- **Platt scaling:** fits a smooth S-shaped correction.
- **Isotonic regression:** fits a step function that can take any shape, as long as it never goes down.

**Why it needs its own set:** the correction must be learned on stays the model never trained on. On its training stays, a model looks better calibrated than it is.

**Order of operations, for every model:**

1. Tune with CV on train, then refit on all of train.
2. Use that model to predict a raw risk for each of the 40k recalibration-set stays.
3. Fit Platt and isotonic on those raw risks and the actual outcomes.
4. Choose none, Platt or isotonic for that model, using 5-fold CV inside the recalibration set, by Brier and the calibration curve.
5. At the final stage, apply the corrections to the model's raw test risks.

Every model is recalibrated, not just the favorite, so the comparison is fair.

## Development stage (no test set)

Built in the comparison notebook and refreshed as models are added. It uses out-of-fold predictions on train, plus CV inside the recalibration set for step 4 above.

- **Table A:** one row per model: CV AUC and Brier (mean ± standard error across the 5 folds), and precision and recall at 0.5 from the out-of-fold predictions. The baseline is always the first row. Precision and recall are for awareness only, not for choosing models.
- **Figure A:** each model's out-of-fold calibration curve, with 10 equal-sized risk groups.
- **Tuning curves:** CV AUC against each tuned hyperparameter, such as penalty strength or tree depth. Appendix material.
- **Recalibration check:** each model's Brier and calibration curve raw, after Platt and after isotonic, from CV inside the recalibration set.

## Choosing before the test set

Before the test set is opened, we write down the recommended model and its recalibration choice, from development-stage evidence only. The test set doesn't choose anything, and nothing is changed after seeing it.

## Final stage (test set, once)

Every model is scored once on the same 100,000 test stays. This is a comparison study, so all models are reported, not just the favorite.

**Error bars:** 95% bootstrap intervals from 1,000 resamples of the test set. Model differences use a **paired** bootstrap: each resample scores both models on the same stays, and the middle 95% of the differences is the interval. If it excludes zero, the difference is real. These intervals cover test-sample noise only, not training noise, and the paper says so.

**Main text**

- **Table 1, overall:** every model, raw and with its recalibration choice: AUC and Brier with intervals, AUC difference from the baseline with a paired interval, and precision and recall at 0.5.
- **Table 2, by group:** AUC and Brier with intervals and stay counts, by main payer and by race, for the baseline and the recommended model. All race and payer groups except blank payer, including small ones such as Multi-racial.
- **Figure 1, the key figure:** calibration curves by group for the baseline and the recommended model after recalibration, with bootstrap bands. Two rows of three panels: Medicaid, commercial (Private Health Insurance + Blue Cross/Blue Shield) and Medicare; White, Black/African American and Other Race.

**Appendix**

- Table 1 with raw, Platt and isotonic rows for every model.
- Table 2 for every model.
- Each model's test calibration curve raw, after Platt and after isotonic.
- Tuning curves.

## From the `payer-calibration` branch

We don't merge it. We cherry-pick:

- `docs/notes/2026-11-08-group-calibration-figure.md`, as part of the record.
- The group-number fix in `docs/course_materials/baseline.txt`.
- The `GROUPS` definitions and the 2×3 panel layout, moved into `common.py` as the template for Figure 1.

Its training-set figure, `calibration_by_group.png`, isn't brought over.

## Not included

- **Calibration slope and intercept:** none of the reference papers report them, and the curves plus Brier already cover calibration.
- **AUPRC and other threshold metrics:** precision and recall at 0.5 are enough.
- **DeLong's test:** the paired bootstrap covers AUC and Brier.
- **Rerunning without diagnosis features:** see Features.
