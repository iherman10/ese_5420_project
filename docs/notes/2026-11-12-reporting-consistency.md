# Reporting Consistency
Created: 2026-11-12
Scope: One shared way for model notebooks to score, save and print CV results, and which outputs the baseline and penalized logistic notebooks no longer print. Replaces the precision and recall method in `2026-11-06-cleaning-encoding-baseline.md` and the unseen-level counts in `2026-11-12-recalibration-and-comparison.md`.

## Shared helpers in `common.py`

- **`fold_scores(fold, y_true, risk, seconds, **details)`** builds one row of a CV scores table: `fold`, `auc`, `brier`, `precision`, `recall`, then model-specific details, then `seconds`. Tuning parameters such as `l1_ratio` and `C` come before `fold`.
- **`logistic_details(model)`** adds `nonzero_coef_share`, `iterations` and `converged` for any logistic regression. The tree models and the MLP will pass their own details.
- **`cv_results(model, scores)`** builds the one-row results table: mean and SD of AUC, Brier, precision and recall across the folds.
- **`show_scores(df)`** prints any scores table with 4 significant figures.

The baseline's own CV loop and results function, and the penalized notebook's own scoring, now use these.

## Precision and recall

- Scored on each validation fold at `THRESHOLD = 0.5`, like AUC and Brier, so the results table has an SD for each. Precision is NaN if a fold flags no stays.
- This replaces computing them once over all out-of-fold predictions. The two agree to 4 digits for the baseline and the chosen lasso, since the folds are stratified and equal-sized.
- Still for awareness only, not for choosing models.

## Nonzero coefficients

- `nonzero_coef_share`, the share of encoded columns with a nonzero coefficient, replaces the separate `nonzero` and `columns` counts in saved scores and printed tables. It's shown as a decimal.
- The final lasso no longer prints its nonzero count.

## Saved results

- The baseline CV and the chosen lasso CV (`cv_scores.csv`, `oof_predictions.parquet`) were recomputed in about 2 minutes. AUC, Brier and iterations were unchanged.
- The penalized tuning grid took about 300 CPU-minutes, so `tuning.csv` was converted in place, with `nonzero_coef_share = nonzero / columns`, instead of refit. It has no precision or recall, so the tuning tables show AUC and Brier only. Refitting the grid would add them. The chosen setting's refit checks its AUC and `nonzero_coef_share` against the tuning rows.

## Outputs dropped

- **Unseen levels on the recalibration set:** no longer printed or discussed in either notebook, and `common.unseen_levels` is removed. The penalized notebook's counts were all zero. The baseline's Encoding section still explains how new levels are encoded.
- **Platt slope and intercept:** no longer printed. Both recalibrators are still fit and saved, and they're judged on the test set in the comparison notebook, as planned.
- **Baseline row and `at_edge`** in the penalized best-setting table: the `auc_vs_baseline` and `brier_vs_baseline` columns cover the baseline, and the markdown says all three best `C` values are at the top of the grid.

## Notebook layout

- No step numbers in headings. The penalized notebook has Setup, Tuning, Recalibration and Findings. The baseline has Data, Cohort and sample, Split, Cleaning, Encoding, Cross-validation and Recalibration.
- OOF calibration stays in the baseline only. Calibration across models belongs in the comparison notebook.
