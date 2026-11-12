# Recalibration and Comparison Split
Created: 2026-11-12
Scope: Where recalibration happens, how Platt and isotonic are fit, and what moves to a final comparison notebook. Replaces the recalibration and test-prediction parts of `2026-11-10-model-comparison-plan.md`.

## The split

**Each model's notebook never touches the test set.** It:

- tunes the model with CV on train and saves the outputs listed in `2026-11-10-model-comparison-plan.md`, except `test_predictions.parquet`
- fits the final model on all of train and predicts the recalibration set (`model.joblib`, `recal_predictions.parquet`)
- fits Platt and isotonic on those recalibration-set risks and saves them as `recalibrators.joblib`

Model notebooks have no calibration plots. Every calibration figure is in the comparison notebook, so the before-and-after comparisons are all in one place.

**A comparison notebook, run once at the end, is the only place the test set is used.** It:

- loads each model's `model.joblib` and `recalibrators.joblib`, and predicts raw, Platt and isotonic risks on the test set
- builds every final table and figure: AUC, Brier, the calibration figures below, and results by payer and race
- reports the change in AUC from isotonic, whose ties can shift it slightly. Platt can't change AUC, since it keeps the order of stays.

`test_predictions.parquet` moves from the model notebooks to the comparison notebook.

## Calibration figures

Both are in the comparison notebook.

1. **Out-of-fold calibration on train:** one plot with the raw risks of every chosen model. It uses training data only, so it can be drawn before the test set is touched, and it shows which models need recalibration.
2. **Test-set calibration:** one small panel per model, each showing raw, Platt and isotonic on the test set. Raw is included so before and after are compared on the same stays and the same final model. Comparing figure 1 with calibrated test risks would mix the recalibration's effect with the change of stays and models. Panels instead of one plot, since every model's three versions on one axes would be about 18 lines.

If figure 2 is too much for the paper, the paper shows raw vs. the better recalibrator for each model, with the full figure in the notebook.

## Why recalibration isn't judged in the model notebooks

The recalibrators are small models whose only input is the main model's risk. They're fit on the recalibration set, so scoring them there would be in-sample, and isotonic especially would look nearly perfect. As decided in `2026-11-10-model-comparison-plan.md`, there's no CV within the recalibration set. The before-and-after comparison happens on the test set, in the comparison notebook.

## How the recalibrators are fit

One shared function in `analysis/common.py`, so every model is recalibrated the same way. It takes a model's recalibration-set risks and the recalibration set's `y`, and returns both recalibrators.

- **Platt:** an unpenalized one-variable logistic regression on the log-odds of the raw risk. A model that's simply too confident or not confident enough is off by a straight-line correction on that scale. Risks are clipped to [1e-6, 1 − 1e-6] first, since some models, such as trees, can output exactly 0 or 1, where the log-odds are infinite.
- **Isotonic:** `IsotonicRegression(out_of_bounds="clip")` on the raw risk. Test risks outside the range seen in the recalibration set get the nearest fitted value instead of an error.

## Baseline

The baseline notebook doesn't have a final model yet (`2026-11-07-saving-results.md`). It gets the same steps as the other models: `model.joblib`, `recal_predictions.parquet` and `recalibrators.joblib`. Its existing `calibration.png` was the baseline checkpoint's figure and can stay, but the comparison notebook's figure 1 replaces it in the paper.

## Shared code

From a review of the notebooks and `common.py`. Only steps every model notebook repeats move to `common.py`:

- **`load_or_compute`** also loads and saves `.joblib` files, so the final model and recalibrators use the same helper.
- **`encode_folds(X, y, make_encoder=...)`** encodes each fold once, fitting the encoder on the fold's training part only. A model that needs a different encoder passes its own.
- **`oof_frame(index, folds, risks)`** builds the out-of-fold table: one row per training stay, with its fold and risk.
- **`predict_risk(model, df)`** returns one risk per stay, indexed by row ID. It's the only place the "Found unknown categories" warning is silenced when predicting the recalibration or test set.
- **`unseen_levels(encoder, df)`** counts the rows whose level was never seen in training, for each feature without an infrequent column. Those rows get all zeros for that feature (`2026-11-06-cleaning-encoding-baseline.md`). Every notebook prints the counts when it predicts the recalibration or test set.
- **`fit_recalibrators(risk, y)` and `recalibrate(recalibrators, risk)`**, kept together so the clipping and log-odds are the same when the comparison notebook applies them.

Model-specific code stays in each notebook: the grid, the fit function, the per-fold diagnostics, the tuning plot and the final model's pipeline.

## Stale results

- **Final model:** loaded through `load_or_compute` if `model.joblib` exists, then checked against the chosen settings.
- **Recalibration-set predictions and recalibrators:** recomputed from the final model on every run and saved, never loaded. Both take seconds, so an old prediction file can never outlive a refit model.

## Order

1. Add the shared code to `common.py`.
2. Penalized notebook: switch to the shared code and add the recalibration step for the chosen lasso.
3. Baseline notebook: clean the recalibration set, add the final model, recalibration-set predictions, recalibrators and unseen-level check. The CV results must not change.
