# Predicting long unplanned hospital stays in New York, fairly across groups

## Group

1030482

## Topic

20, 23

## Question

For unplanned (emergency, urgent or trauma) admissions to New York hospitals, how well can course methods predict a stay longer than 7 days from information available near admission, and are the predicted probabilities equally reliable for Medicaid and commercially insured patients and across racial groups?

## Why it matters

Bed managers and discharge planners could use a calibrated early estimate of which patients will stay more than a week to plan beds and start discharge arrangements sooner. If the probabilities are systematically off for Medicaid patients or particular racial groups, those patients would be planned for wrongly, so reliability matters as much as accuracy.

## Data

NYSDOH SPARCS Hospital Inpatient Discharges (de-identified), one row per inpatient stay, 33 columns, 2023 and 2024 pooled with a year indicator:
https://health.data.ny.gov/d/46xm-urtu (2023, 2,125,754 stays)
https://health.data.ny.gov/d/sf4k-39ay (2024, 2,196,737 stays)
Keeping only emergency, urgent and trauma admissions leaves about 3.2M stays, about 23% of which last more than 7 days (the target). The inputs are 17 categorical columns known roughly near admission: age group, gender, race, ethnicity, 3-digit ZIP, three payer fields, admission type, ED indicator, facility, county, health service area, body system (APR MDC), medical vs. surgical, principal diagnosis (CCSR) and discharge year. Charges, costs, disposition, severity, mortality risk, APR-DRG and procedure are excluded as leakage.

## Baseline

Unpenalized logistic regression on one-hot encoded features. It has no hyperparameters; the regularized versions below are tuned by K-fold cross-validation on the training set.

## Comparison

Lasso, ridge and elastic net logistic regression; a single decision tree, random forest and gradient-boosted trees; an MLP; and a stacking ensemble that combines them. All are tuned by K-fold cross-validation on the training set and recalibrated with Platt scaling and with isotonic regression on a held-back training slice.

## Evaluation

Random 80/20 train/test split stratified on the target; the test set is used once, and every model is scored on the same test rows. Metrics: AUC for ranking, and calibration curves for whether predicted probabilities match observed rates, before and after each recalibration. Both are repeated by payer (Medicaid vs. commercial, with Medicare for context) and by race. Error bars: bootstrap resamples of the test set for AUC and for each calibration bin. Published admission-time benchmarks are AUC 0.75-0.81 (Zeleke et al. 2023; Rajkomar et al. 2018); above about 0.9 would signal leakage.

## Key figure

Calibration curves on the test set: x-axis is mean predicted probability of a stay over 7 days (binned), y-axis is the observed long-stay rate, with the diagonal as perfect calibration. One panel per payer group (and a second set by race), with a curve for the baseline and for the best model after recalibration, each with bootstrap bands. A companion table gives test AUC with intervals for every model, overall and by group.

## Risks and fallback

The slower models (random forest, MLP, stacking) may be too slow on 3.2M rows, so I will fit on a random subsample of a few hundred thousand, checking that the fairness subgroups stay large enough. Flexible models may not beat logistic regression, and there may be no gap between groups; both would be reported as findings. Fallback paper: logistic regression against gradient-boosted trees only, with AUC and calibration before and after Platt scaling and isotonic regression, overall and by payer.

## Who does what

Isaac: everything (solo project).

## AI plan

Claude Code for data cleaning, model and plotting code, debugging, and tightening prose. I will read every line I keep, rerun experiments with fixed seeds, and open every reference myself. The acknowledgements will include an AI use statement.
