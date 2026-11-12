# Base-Rate Threshold
Created: 2026-11-12
Scope: The threshold for precision and recall moves from 0.5 to the training set's long-stay rate. Replaces `THRESHOLD = 0.5` in `2026-11-12-reporting-consistency.md`; everything else there stands.

## Decision

- `common.THRESHOLD` is the share of training stays that are long: 0.2306 (83,026 of 360,000). A stay is flagged if its risk is at or above it.
- **Why:** long stays are 23% of the data, so 0.5 flags only stays the model rates as more likely long than not. That's about 9% of stays, giving the lasso a CV recall of 0.26. The base rate flags any stay the model rates as riskier than the average patient, which reads plainly because the models are calibrated.
- **Cost reading:** with calibrated risks, a threshold `t` treats a missed long stay as `(1 − t) / t` times as costly as a false alarm, about 3.3 here.
- **Training data only:** the rate comes from the training set, not the recalibration or test sets. The CV folds are stratified, so each fold's training part has the same rate to within rounding, and one constant serves all folds.
- **Same threshold for every model,** applied to recalibrated risks on the test set, so differences in precision and recall come from the models.
- Still for awareness only, not for choosing models. AUC, Brier and calibration don't use a threshold.

## Expected effect (lasso, out-of-fold)

| Threshold | Flagged | Precision | Recall |
|---|---|---|---|
| 0.5 | 9% | 0.65 | 0.26 |
| 0.2306 | 40% | 0.41 | 0.70 |

The threshold maximizing F1 is about 0.26, close to the base rate.

## Rerun

- Nothing is refit differently. Only the yes/no flags derived from the risks change.
- The baseline and lasso `cv_scores.csv` and `oof_predictions.parquet` are deleted and recomputed so the saved files match the code. AUC, Brier and the out-of-fold risks should match the earlier run exactly.
- `tuning.csv`, the final models, recalibrators and recalibration-set predictions are unaffected and kept.
- Notebooks assert that `THRESHOLD` matches the training set's long-stay rate.
