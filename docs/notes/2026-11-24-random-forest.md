# Random Forest
Created: 2026-11-24
Scope: How the random forest (step 4 of `2026-11-10-model-comparison-plan.md`) is encoded, tuned and reported, laid out by the notebook's sections.

## Setup

- **Notebook:** `analysis/random_forest.py`, paired with an `.ipynb`. Results in `results/random_forest/`.
- **Model:** `RandomForestClassifier(random_state=42)` with the default Gini criterion and bootstrap samples.
- **No `class_weight`:** reweighting the classes would distort the risks.
- **Encoding:** full one-hot from `common.make_encoder()`, as for every non-baseline model. Each fold is encoded once with `common.encode_folds`.

## Tuning

### Grid

| Setting | Values | Count |
| :--- | :--- | :--- |
| `max_features` | `"sqrt"` (about 25 columns, 0.04), 0.1, 0.2, 0.3 | 4 |
| `min_samples_leaf` | 5, 10, 25 | 3 |

12 settings × 5 folds = 60 fits, the same as XGBoost.

- **Why these two:** ISL §8.2.2, Breiman (2001) and the scikit-learn docs name `max_features` as the setting to tune. In Probst, Boulesteix & Bischl (2019), it was the forest setting that tuning helped most. Leaf size controls how smooth each tree's risks are, which matters for Brier.
- **`max_features` above √p:** it counts encoded columns, not the 17 features. 445 of a fold's 609 columns are 1 in fewer than 1% of stays, so `"sqrt"` mostly draws rare dummy levels. Breiman §5.1 says that with many categorical features, `max_features` must rise to two to three times its usual value. Probst's best values, 0.26–0.67 of the features, point the same way. In timing tests on fold 1, 0.1 beat `"sqrt"` at every leaf size, and at a leaf size of 25, 0.3 beat 0.1. 0.3 is in the grid from the start, rather than left to the edge check, since 0.2 seemed likely to win.
- **No leaf size of 1:** the textbook default, but on fold 1 it scored worst (AUC 0.740 against 0.777 at 5), took 5× longer, and a 500-tree forest needs about 7 GB.
- **Leaf size below the tree's 25:** averaging many trees removes variance, so each tree can be grown deeper than the single tree.

### Fixed settings

| Setting | Value | Why |
| :--- | :--- | :--- |
| `n_estimators` | 100 to tune, 300 for the chosen model | On fold 1, 100 trees was within 0.001 AUC of 300, and AUC was flat past 200. More trees never overfit (ISL §8.2.2, Breiman Thm 1.2) |
| `bootstrap`, `max_samples` | True, None (full bootstrap) | Subsampling to 25% halved the run time but cost 0.004 AUC |
| `criterion`, `max_depth` | Gini, None | No source recommends tuning them |

- **Tree count check:** the chosen setting's fold fits grow 300 trees. With the same seed, their first 100 trees are the tuning fit's trees, so the scores of those 100 must match the tuning rows, as XGBoost's did.

### Choice and run

- **Choice:** the best mean CV AUC, as for the other models. No one-standard-error rule. Probst recommends tuning on Brier, but every model here is chosen on AUC, and Brier is reported beside it.
- **CV, not OOB:** OOB is a valid error estimate for a forest (ISL §8.2.1), but every model is compared on the same 5 folds.
- **Edge check:** if the best setting is at an edge of a range, extend that range by one step: `max_features` 0.4, or a leaf size of 2 or 50.
- **How it runs:** the fits run one at a time, each on all 8 cores, keeping only scores and predictions, so only one forest is in memory at a time. Input stays sparse: on fold 1 it was twice as fast as dense, with identical results. Fit time grows roughly in proportion to `max_features`. The estimated tuning time is about 2.5–3 hours, with the 0.3 settings alone about 1.5 hours. The 300-tree fold fits add about 15 minutes at `"sqrt"`, up to about 3 hours if 0.3 with a leaf size of 5 wins.

## To check

- ESL §15.3 wording on `max_features`, node size and OOB before citing ESL. The research agent couldn't reach the book.
- Run time and memory at `max_features` 0.2 and 0.3 with leaf sizes 5 and 10 were extrapolated, not timed. Time one fit before running the grid.

## Notebook

The same path as the decision tree and XGBoost notebooks.

1. **Setup:** the shared data and features from `common`, and the baseline's CV scores for reference.
2. **Tuning**
   - **The model:** the fixed settings and the grid.
   - **Fits:** the grid on the 5 folds, saved to `tuning.csv`.
   - **Tuning curve:** CV AUC and Brier across the grid, saved as `tuning.png`.
   - **Best setting table:** each setting's scores and their difference from the baseline.
   - **Chosen model:** `CHOSEN`, with asserts that it's the best setting and passes the edge check.
   - **Out-of-fold predictions:** the chosen setting refit on the 5 folds, saved to `cv_scores.csv` and `oof_predictions.parquet`. The scores must match the tuning rows.
   - **Results:** the standard results table from `common.cv_results`.
3. **Final model:** refit on all of train, saved as `model.joblib`. Feature importance: impurity importance summed over each feature's one-hot columns, as a table and a bar chart (`feature_importance.png`), with a short commentary. No readable tree.
4. **Recalibration:** the final model predicts the recalibration set (`recal_predictions.parquet`), and `common.fit_recalibrators` fits Platt and isotonic (`recalibrators.joblib`). The test set isn't touched.
5. **Findings:** the chosen setting, the comparison with the other models, what tuning showed, importance and recalibration.

## Outputs

`results/random_forest/` has the files in `2026-11-10-model-comparison-plan.md`, plus `recalibrators.joblib` and `feature_importance.png`.
