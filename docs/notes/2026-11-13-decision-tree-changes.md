# Decision Tree Changes
Created: 2026-11-13
Scope: What changed from `2026-11-12-decision-tree.md` while building `analysis/decision_tree.py`, and the shared table display added to `common.show_scores`.

## Leaf-size grid

- The first grid's best setting was `ccp_alpha = 1e-5` with `min_samples_leaf = 25`, the smallest leaf size tried, so the edge check applied.
- Instead of adding one step, the leaf-size range was extended down to 1: `min_samples_leaf` in {1, 3, 6, 12, 25, 50, 100, 200, 400, 800}. A leaf size of 1 is the full tree pruned back with α alone, the textbook method in ISL ch. 8, so the grid also tests whether leaf size needs tuning once α is tuned.
- 50 settings × 5 folds = 250 fits, about 117 CPU-minutes (15 minutes on 8 cores). The first grid's 150 settings were refit and gave identical scores.
- 25 is still best and now inside the range. Leaf sizes 6–25 are within 0.001 AUC, less than the fold-to-fold SD. `ccp_alpha` is unchanged, and 1e-5 is inside its range.

## Notebook layout

- **Edge check:** a sentence in the best-setting markdown, as in the penalized notebook, not a separate section.
- **Tuning curve:** symlog x-axis (linear between 0 and 1e-5), so α = 0 is shown. It plots 6 of the 10 leaf sizes (1, 6, 25, 100, 400, 800) for readability, and the table has all 10.
- **Best-setting table:** the best α at each leaf size, shown with the gradient display below.
- **Sections:** Setup, Tuning, Final model, Recalibration and Findings. The final model gets its own section, since the readable tree and feature importance use it. Recalibration holds only Platt and isotonic.
- **Readable tree:** each box asks a plain question ("feature = level?"), with the description for diagnosis, MDC and facility codes, its share of stays and its long-stay share. The arrows read No and Yes, replacing scikit-learn's True and False for `<= 0.5`, which read backwards.
- **Feature importance:** a table and a bar chart, saved as `feature_importance.png`, plus a short commentary on what the importances say. Summing the one-hot columns' importances is exact for a single tree, since each split is on one column. The commentary states the two limits of impurity importance: it favours features with many levels, and it's measured on training data. Permutation importance was considered and dropped as not worth the extra work.

## Gradient tables

- **`common.show_scores(scores, highlight=False)`:** with `highlight=True` in a notebook, the table is styled. AUC, Brier and their `_vs_baseline` differences are shaded darker the better they are, and the best value in each is bold. Outside a notebook it prints as before.
- **Signed differences:** numeric `*_vs_baseline` columns always show a sign and 5 decimal places, in printed and styled tables.
- **Where it's used:** the best-setting tables in the decision tree and penalized notebooks. The penalized notebook's `_vs_baseline` columns are now numbers rather than pre-formatted text. Nothing else in that notebook changed.

## Outputs

`results/decision_tree/` has the files in `2026-11-10-model-comparison-plan.md`, plus `recalibrators.joblib`, `tree.png` and `feature_importance.png`.
