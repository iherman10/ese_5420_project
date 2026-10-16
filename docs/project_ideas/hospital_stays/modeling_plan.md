# Modeling Plan: Predicting Long Hospital Stays

*Drafted 2026-10-15. Builds on the decisions in `scoping.md` and the Summary of `dataset_overview.md`.*

## The prediction task

- **Unit:** one unplanned inpatient stay (emergency, urgent or trauma admission), SPARCS 2023 + 2024 pooled.
- **Target:** $y = 1$ if length of stay > 7 days (LOS ≥ 8), else 0. About 20% of stays are positive.
- **Output:** every model produces a probability $\hat p = P(y = 1 \mid x)$, not just a yes/no label.

## Data

**Variables.** These are roughly what hospital staff know near the time of admission:

- Age group, gender, race, ethnicity
- 3-digit patient ZIP code
- Payer (primary, secondary, tertiary)
- Type of admission (emergency, urgent, trauma) and ED indicator
- Facility, county and health service area
- Body-system category (APR MDC), medical vs. surgical, and principal diagnosis category (CCSR)
- Discharge year

**Split.** Random 80/20 train/test split. Hyperparameters are tuned with K-fold cross-validation on the training set, and a slice of the training set is held back to fit the recalibration (see below). The test set is used once, at the end.

## Models

| Model | Course module | Notes |
| :--- | :--- | :--- |
| Logistic regression (unpenalized) | Classification | **Baseline** |
| Lasso logistic regression | Regularization | L1 penalty; drops unhelpful variables |
| Ridge logistic regression | Regularization | L2 penalty; shrinks all coefficients |
| Elastic net logistic regression | Regularization | Blend of lasso and ridge; mixing parameter tuned by CV |
| Decision tree | Trees & ensembles | Single readable tree; shows what the ensembles improve on |
| Random forest | Trees & ensembles | |
| Gradient-boosted trees (XGBoost) | Trees & ensembles | |
| MLP | Neural networks | |
| Stacking ensemble | Trees & ensembles | A logistic regression that combines the other models' predictions |

## How we compare them

All on the test set:

1. **AUC:** does the model rank long stays above short ones?
   - **In plain terms:** pick one patient who stayed longer than a week and one who didn't, at random. AUC is the chance the model gives the long-stay patient the higher risk. An AUC of 0.80 means it orders such a pair correctly 80% of the time.
   - **Reading the number:**
     - 0.5: no better than a coin flip.
     - 0.6–0.7: weak.
     - 0.7–0.8: decent. Published admission-time models for this target land around 0.75–0.81.
     - 0.8–0.9: strong.
     - Above 0.9: suspicious here. It usually means a variable is leaking the answer.
   - **What it doesn't tell you:** AUC only cares about the *order* of patients, not the probabilities themselves. A model that says 90% vs. 60% gets the same AUC as one that says 30% vs. 20%. That's why we also check the Brier score and calibration.
2. **Calibration curves:** of the patients given about 30%, do about 30% actually stay longer than a week?
   - **How the curve is built:**
     1. Sort test patients into about 10 bins by predicted risk.
     2. For each bin, plot the average predicted risk (x-axis) against the share who actually stayed long (y-axis).
   - **Reading the curve:**
     - **On the diagonal:** the probabilities mean what they say.
     - **Below the diagonal:** the model over-predicts. It says 40%, but only 25% stay long.
     - **Above the diagonal:** the model under-predicts.
     - **An S-shape** (below at the high end, above at the low end): the model is over-confident, pushing probabilities too close to 0 and 1. Boosted trees often do this.
   - **Brier score:** reported as a one-number summary. It's the mean squared error of the probabilities, so lower is better.
   - **Recalibration:** a fix applied *after* training. It learns a correction from the model's raw probability to a better one, like a translation table: "when the model says 40%, it really means 25%."
     - **Method: Platt scaling.** It fits a smooth S-shaped correction. It is just a one-variable logistic regression that takes the model's score as its input.
     - **It doesn't change the order of patients,** so AUC stays the same and only the probabilities improve.
     - **The correction is learned on data the model wasn't trained on:** a slice of the training set held back for this purpose, never the test set.
   - **Recalibration in the analysis:** for each model, fit Platt scaling on the held-back slice, then plot the calibration curves before and after on the same axes and compare Brier scores. Expected result: the boosted trees improve the most, and logistic regression barely changes because it's usually well calibrated already.
3. **Fairness:** repeat the calibration curves separately by payer (Medicaid vs. commercial) and by race.
   - **In plain terms:** this isn't a new analysis, just the checks above run separately for each group. A model can look well calibrated overall while being wrong in opposite directions for different groups.
     - **Example:** the model says 30% for Medicaid patients who actually stay long 45% of the time, and 30% for commercial patients who stay long only 20% of the time. On average the errors cancel, but the model underestimates Medicaid patients' needs and overestimates commercial patients'.
   - **Calibration by group:** plot each group's curve on the same axes.
     - Above the diagonal: the model under-predicts for that group.
     - Below the diagonal: it over-predicts.
   - **AUC by group:** within each group, how well does the model sort long stays from short ones? A lower AUC for one group means the model is simply less useful for those patients, even if its probabilities are honest.
