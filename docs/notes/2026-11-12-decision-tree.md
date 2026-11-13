# Decision Tree
Created: 2026-11-12
Scope: How the single decision tree (step 2 of `2026-11-10-model-comparison-plan.md`) is encoded, tuned and reported.

## Model

- **Notebook:** `analysis/decision_tree.py`, paired with an `.ipynb`. Results in `results/decision_tree/`.
- **Model:** `DecisionTreeClassifier(random_state=42)` with the default Gini criterion. The criterion isn't tuned, since Gini and entropy rarely pick different splits.
- **No `class_weight`:** reweighting the classes would distort the risks.
- **Encoding:** full one-hot from `common.make_encoder()`, as for every non-baseline model. scikit-learn's trees have no native categorical splits, so each split separates one level from the rest. Ordering levels by their long-stay rate was considered and dropped as too complicated.

## Tuning

Grow a tree and prune it back with cost-complexity pruning, choosing α by CV, as in ISL ch. 8 and ESL §9.2. A minimum leaf size is tuned alongside α.

| Setting | Values | Count |
| :--- | :--- | :--- |
| `ccp_alpha` | 0, 1e-5, 3e-5, 1e-4, 3e-4 | 5 |
| `min_samples_leaf` | 25, 50, 100, 200, 400, 800 | 6 |

30 settings × 5 folds = 150 fits, in the same range as the penalized search.

- **Why these two:** `ccp_alpha` is the course's way of choosing tree size. In Probst, Bischl & Boulesteix (2019), minimum leaf size was the tree setting that tuning helped most. `max_depth`, `max_leaf_nodes`, `min_samples_split` and `min_impurity_decrease` are cruder ways of limiting tree size, so they stay at their defaults.
- **Leaf size and risks:** a leaf's risk is its share of long stays. Small leaves give noisy risks near 0 or 1, and large leaves give few distinct risks, which costs AUC.
- **Leaf-size range:** published defaults, such as rpart's minimum of about 12 stays per leaf, come from much smaller datasets. With about 288,000 stays per fold, 25 to 800 is a sensible range.
- **`ccp_alpha` range:** α is the drop in the tree's average Gini impurity that a split must earn. Because it's an average over all stays, its useful values don't depend on the sample size. The root impurity is about 2 × 0.23 × 0.77 ≈ 0.35, so useful values are small, and 0 is no pruning.
- **Leaf size as a count:** the final model refits on 360,000 stays rather than 288,000, so the same count is a slightly smaller share of the data. That's accepted, since counts are easier to read than fractions.
- **Choice:** the best mean CV AUC, as for the lasso. No one-standard-error rule.
- **Edge check:** if the best setting is at an edge of either range, extend that range by one step, for example α = 3e-6 below 1e-5, or a leaf size of 1,600.
- **How it runs:** as in the penalized notebook. Each fold is encoded once with `common.encode_folds`, and the 150 fits run in parallel with joblib.

## Reporting

- **Per-fold details:** depth and number of leaves, passed to `common.fold_scores`. They stay in the notebook, since the random forest and XGBoost report different details.
- **Tuning curve:** CV AUC and Brier against `ccp_alpha`, one line per leaf size.
- **Readable tree:** the top 3 levels of the final tree, with the encoder's level names, saved as `tree.png`.
- **Feature importance:** impurity-based importance summed over each feature's one-hot columns, so one row per original feature.
- **Recalibration:** the same steps as the lasso. The final model refits on all of train and predicts the recalibration set, then `common.fit_recalibrators` fits Platt and isotonic. The test set isn't touched. A tree's risks take only as many values as it has leaves, so isotonic sees a step-shaped input.
- **Outputs:** the files in `2026-11-10-model-comparison-plan.md`, plus `recalibrators.joblib` and `tree.png`.

## To check

- ISL and ESL wording and section numbers before quoting them in the paper.
