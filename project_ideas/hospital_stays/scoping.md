# E3 Scoping Discussion: Predicting Long Hospital Stays in New York

*Session date: 2026-10-15*

Summary of the first discussion of project idea E3 from `project_ideas/project_guide.html`: what "fairly" means, whether the temporal holdout is needed, and which data to use.

## Decisions

| Topic | Decision |
| :--- | :--- |
| Data | SPARCS de-identified inpatient discharges, **2024 only** |
| Train/test split | **Random**: 80/20 train/test, K-fold CV on the training set for tuning |
| Temporal holdout | **Dropped.** Optional extra only: train on 2023, score on 2024 |
| 2021–22 data | **Excluded.** COVID-era years would only add noise |
| Sample size | Probably subsample to a few hundred thousand rows so the random forest and MLP train quickly |

## Why the temporal holdout was dropped

- The original guide trained on 2021–22 and tested on 2023. That was largely a carryover from the forecasting framing used for other project ideas.
- Each discharge is close to an independent record. There's no time-series dependence: dates are stripped to the year and there's no patient ID. A random split is therefore a valid, honest held-out evaluation.
- A temporal split would answer a different question: does the model still work next year? That question is legitimate, because COVID waves in 2021–22 and the 2023 end of Medicaid continuous enrollment shift the patient mix. It isn't needed for the core questions, though.
- The temporal split never meant training separate models per year. It only changed which rows go in the test set.
- The project requirements ask for an honest evaluation on held-out data. They only require time-respecting splits for time-series data, and this isn't time series.

## What "fairly" means

It refers to **subgroup fairness**, not the split. The question is whether predicted probabilities are equally reliable for Medicaid and commercially insured patients, and across racial groups.

### Ranking vs. calibration

The model outputs each patient's probability of staying more than 7 days. There are two ways to judge those numbers:

- **Ranking (AUC):** if you sort patients by predicted risk, do the ones who actually stayed long end up near the top? A model can rank perfectly while its numbers are off. Saying 90% vs. 60% sorts patients just as well as 30% vs. 20%.
- **Calibration:** are the numbers literally true? Of the patients given about 30%, roughly 30% should actually stay more than a week. Weather-forecaster analogy: if it rains on only half the days they say "80% rain", they're poorly calibrated, even if they're good at telling rainy days from dry ones.
- **Fairness check:** compare calibration curves by payer and by race. A model can be calibrated overall yet say 30% for Medicaid patients who actually stay long 45% of the time, systematically underestimating their needs.

Expected storyline: boosted trees win on AUC but are overconfident, and recalibration (Platt or isotonic) fixes it. Results that don't favour the fanciest model count as legitimate findings.

### Fit with the project requirements

- **Course methods:** logistic regression, lasso, random forest, XGBoost and an optional MLP, compared on AUC, calibration curves and Brier score. This is the syllabus's "careful comparison of course methods."
- **Public, usable dataset:** SPARCS is free and already clean.
- **Honest reporting:** negative and mixed results are reported as findings.
- **The fairness angle isn't required.** It's what sets the project apart, because SPARCS is a well-worn dataset.

## Data availability (checked 2026-10-15)

| Year | Discharges | Dataset ID |
| :--- | :--- | :--- |
| 2023 | 2,125,754 | [`46xm-urtu`](https://health.data.ny.gov/d/46xm-urtu) |
| 2024 | 2,196,737 | [`sf4k-39ay`](https://health.data.ny.gov/d/sf4k-39ay) |

2025 isn't published yet. Earlier years back to 2009 are available on health.data.ny.gov.

## Open issue: when is the prediction made?

This matters more than the split. Severity of illness, APR-DRG and procedure codes are all assigned **at discharge**, from the full coded record. A model that uses them predicts a long stay given information only known after the stay.

The project needs to choose one framing before selecting features:

- **Retrospective or risk-adjustment model:** use discharge-coded features, and frame it explicitly as retrospective.
- **Admission-time model:** restrict features to what is known at admission, such as age group, admission type, payer, facility and admission diagnosis if available. This gives lower AUC but a more honest prediction setting.
