# Saving Results
Created: 2026-11-07
Scope: A simple way to save slow results, such as model fits and cross-validation, so they only have to run once. Built as part of Step 6, in the baseline notebook.

## How it works

Each model gets one folder:

```
results/
  baseline_logistic/
    cv_scores.csv
    oof_predictions.parquet
    model.joblib
```

| File | What it holds | In git? |
| :--- | :--- | :--- |
| `cv_scores.csv` | Scores for each fold | Yes |
| `oof_predictions.parquet` | One out-of-fold probability per training stay, with its row ID | No |
| `model.joblib` | The model fitted on all of train | No |

A small helper loads a saved result if the file exists. Otherwise, it computes the result and saves it. On another machine, the files not in git need a rerun.

## Stale results

- **Check:** when loading predictions, stop with an error if their row IDs don't match the current training set.
- **Rerun:** if something earlier in the notebook changes, delete the old results and rerun. The check won't catch every change, such as a new cutoff.

## What we don't save

- **The 5 fold models:** we only need their scores and predictions.
- **A full baseline model yet:** the checkpoint only uses CV. We fit on all of train when we get to recalibration and test.
