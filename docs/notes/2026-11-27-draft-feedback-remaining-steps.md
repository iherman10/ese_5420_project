# Draft Feedback and Remaining Steps
Created: 2026-11-27
Scope: The professor's feedback on the early draft, the remaining steps it leaves, and the decisions still open before the test set is opened. The final paper is due Friday 11 December 2026.

## Feedback in brief

- **Question:** make calibration by group the centerpiece, with the model comparison as its setup. Skip the random forest and the MLP.
- **Checks:** rerun the baseline in CV without the three discharge-coded diagnosis features, so "somewhat optimistic" becomes a number. Fit on 2023 stays and score 2024 stays as a robustness check, planned before the test set is opened.
- **Baseline:** right baseline, same table and folds. The low end of the boosting grid is unlikely to move AUC: fix the grid and move on.
- **Error bars:** the paired bootstrap is right, and calling the lasso a tie is right (0.0003 AUC is consistent across folds but too small to matter). For fairness, define one number per group in advance: observed rate minus mean predicted risk, with its bootstrap interval, plus the Medicaid minus commercial difference. Choose Platt or isotonic before the test set, or report both.
- **Paper:** shorten Section 2 to make room for the calibration and fairness figures. Remove the TODO notes and the milestone list. Add the code link and the AI use statement.
- **Limitation to add:** one pooled recalibration map can't remove a gap specific to one group.

## Remaining steps

### Before the test set

1. **Drop the random forest and the MLP.** The models are the baseline, the lasso, the decision tree and XGBoost. `analysis/random_forest.py` and `2026-11-24-random-forest.md` stay as a record but aren't run or reported. This replaces steps 4–6 of `2026-11-10-model-comparison-plan.md`.
2. **Freeze XGBoost** at the setting in `2026-11-24-xgboost-chosen-model.md`. The `min_child_weight = 1` edge stays, as the feedback advises. The local `results/xgboost/` has no `model.joblib`, `recal_predictions.parquet` or `recalibrators.joblib`, so re-run the notebook's final-model and recalibration sections on this machine.
3. **Diagnosis ablation:** the baseline in 5-fold CV on the same folds, without `ccsr_diagnosis_code`, `apr_mdc_code` and `apr_medical_surgical_description`. Report its AUC and Brier and their differences from the full baseline. This uses train only.
4. **2023→2024 check:** fit on 2023 stays and score 2024 stays, with `discharge_year` dropped as a feature. The design is open (below).
5. **Pre-registration note**, written before the test set is opened:
   - **Bootstrap:** paired. Resample the 100,000 test stays with replacement, score every model on the same resample, and repeat 1,000 times. Report 95% percentile intervals for AUC, Brier and each model's difference from the baseline. A difference whose interval includes zero is a tie. Group results resample within each group.
   - **Fairness statistic:** for each group, observed long-stay rate minus mean predicted risk, with its bootstrap interval. The pre-registered contrast is that gap for Medicaid minus the gap for commercial.
   - **Recalibrator:** Platt, isotonic or both (below).

### Test set, once

6. **Comparison notebook:** the only place the test set is used, as in `2026-11-12-recalibration-and-comparison.md`. It scores every model raw, with Platt and with isotonic. It produces:
   - the AUC and Brier table with bootstrap intervals
   - the out-of-fold and test-set calibration figures
   - the fairness statistics and the calibration-by-group figure
   - the 2023→2024 result

### Paper

7. **Reframe:** the title, abstract and introduction lead with calibration by payer and race. The model comparison is the setup.
8. **Shorten Section 2** to make room for the calibration and fairness figures.
9. **Add results:** the diagnosis ablation and the 2023→2024 check, plus the pooled-recalibration limitation.
10. **Clean up:** remove every `\todo`, the random forest and MLP mentions and the milestone list. Add the code link and the AI use statement in the acknowledgements. Check the page count and `pdffonts`.
11. **Video:** 5 minutes. The feedback doesn't mention it, but it's still due.

## Open decisions

Each has a proposed default. Settle them in the pre-registration note.

- **What the 2023→2024 check is scored on.** Proposed: fit on the 2023 training stays, recalibrate on the 2023 recalibration stays, and score the 2024 test stays in the one test run. The alternative is to score the 2024 training stays. That never touches the test set, but it's a weaker check: those stays shaped the encoder levels and the tuning of the main models. Also open: whether the 2023 fit reuses each model's chosen settings unchanged (proposed) or is retuned.
- **Which models get the robustness checks.** The feedback names the baseline for the ablation. Proposed: the ablation on the baseline only, and the 2023→2024 check on the baseline and XGBoost.
- **Platt vs. isotonic.** Proposed: report both everywhere. Isotonic is picked in advance for the headline precision, recall and fairness numbers, because it's the more flexible correction and the recalibration set is large. Check whether the out-of-fold calibration figure argues for Platt instead.
- **Ablation in the paper.** Proposed: a row in the main CV table, rather than only a sentence.
- **Race contrast.** Payer has a pre-registered contrast (Medicaid minus commercial). Should race have one too (Black minus White)? Proposed: yes, so the race claim doesn't rest on reading curves either.
- **Model for the fairness analysis.** Proposed: report the fairness statistics for every model. XGBoost, as the best-AUC model, is the one discussed in the text. Tie this to the rule for choosing the recommended model, which the draft lists as still to be recorded.
