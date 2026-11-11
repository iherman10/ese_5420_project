# Penalized Logistic Regression
Created: 2026-11-11
Scope: How the ridge, lasso and elastic net logistic regressions (step 1 of `2026-11-10-model-comparison-plan.md`) are trained and tuned, plus the leftovers from the `payer-calibration` branch. How the results are reported and the model is chosen is left open.

## From the `payer-calibration` branch

This replaces that section of `2026-11-10-model-comparison-plan.md`.

- The group-number fix in `docs/course_materials/baseline.txt` is cherry-picked.
- The payer and race group definitions are in `common.GROUPS`. They are only for reporting results by group. They don't change the rows, features or encoding the models see.
- The note `2026-11-08-group-calibration-figure.md` and the group calibration figure and its code are not brought over. The final fairness figure on the test set replaces them.

## Baseline check

The refactor into `common.py` re-ran the baseline. `cv_scores.csv` matches the earlier run exactly on every fold for AUC, Brier, columns, iterations and convergence. Only fit times differ.

## One elastic net model

Ridge and lasso are the two ends of elastic net. Its penalty is `l1_ratio × lasso penalty + (1 − l1_ratio) × ridge penalty`, so `l1_ratio = 0` is ridge and `l1_ratio = 1` is lasso. Tuning one elastic net over both settings fits the same models as tuning the three separately, and is the standard approach (glmnet's guidance; scikit-learn's `LogisticRegressionCV`).

## Grid

| Setting | Values | Count |
| :--- | :--- | :--- |
| `l1_ratio` | 0 (ridge), 0.5 (elastic net), 1 (lasso) | 3 |
| `C` | `np.logspace(-5, 0, 11)`, 1e-5 to 1 | 11 |

That is 33 settings × 5 folds = 165 fits.

- **What C is:** scikit-learn's inverse penalty strength. Small C is a strong penalty, and large C approaches the unpenalized baseline.
- **In terms of λ:** λ = 1/(C·n) on glmnet's per-row scale. With n = 360,000 training stays, the grid runs from λ ≈ 0.28 down to λ ≈ 0.0000028. Inside the CV folds n is 288,000, so λ is about 25% larger for the same C.
- **Why this range:** log-spaced over 5 orders of magnitude, a bit wider than glmnet's default of 4. C = 1 is already a very weak penalty at this sample size.
- **Edge check:** if the best C is at either end of the grid, extend the grid that way.

## Fitting

- **Notebook:** `analysis/penalized_logistic.py`, paired with an `.ipynb`.
- **Model:** `LogisticRegression(solver="saga", l1_ratio=..., C=..., max_iter=1000, random_state=42)`. No `penalty=` argument, which is deprecated in the installed scikit-learn (1.9.1). saga is the only solver that handles every `l1_ratio`.
- **Pipeline:** the full one-hot encoder (`common.make_encoder()`, no level dropped), then the model. `GridSearchCV` on `common.FOLDS` refits the encoder inside each fold.
- **No scaling:** every column is a 0/1 dummy.
- **Scores:** CV AUC and Brier for every setting. The tuning curve plots CV AUC against C, one line per `l1_ratio`.
- **Outputs:** the files listed in `2026-11-10-model-comparison-plan.md`, through `common.load_or_compute`.

## Checks

- Watch for convergence warnings, especially at large C. Raise `max_iter` if needed.
- Expect AUC close to the baseline's 0.774. With 360,000 rows and about 600 columns, the baseline has little room to overfit.

## Open questions

- Report ridge, lasso and elastic net as three separate models, or only the winner? This decides whether results go in one folder or three.
- Pick C by the best CV AUC, or by the one-standard-error rule (the strongest penalty within one standard error of the best)?
- Is this grid size the search budget the later models must match?
- Add `l1_ratio` 0.25 and 0.75 (55 settings) if 0.5 comes out clearly best?
- Fairness groups, for later: where "Managed Care, Unspecified" goes among the payer groups, and how Multi-racial stays are handled.
