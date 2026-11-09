# Group Calibration Figure
Created: 2026-11-08
Scope: A first version of the key figure for the baseline checkpoint (`baseline.png`).

- **What:** the baseline's calibration curves in two rows of three panels, using the groups from the fairness plan.
  - Main payer: Medicaid, commercial (Private Health Insurance + Blue Cross/Blue Shield) and Medicare.
  - Race: White, Black/African American and Other Race. Multi-racial is too small for a curve and appears only in the final table.
- **Data:** the saved out-of-fold predictions on the training set. The recalibration and test sets stay untouched.
- **Bins:** 10 equal-sized groups by predicted risk within each group, as in the overall curve.
- **No error bands yet:** bootstrap bands come with the final test-set figure.
- **Saved as:** `results/baseline_logistic/calibration_by_group.png`, in git like the other figures. Copy it to `baseline.png` for the submission.
