# Analysis Plan: Predicting Long Hospital Stays in New York

Created: 2026-10-15 (last edited 2026-10-30)
Scope: The original plan for the project's question, data, models and evaluation, plus open decisions.

## Question

For unplanned (emergency, urgent or trauma) admissions to New York hospitals, how well can course methods predict a stay longer than 7 days from the information available near admission, and are the predicted probabilities equally reliable for Medicaid and commercially insured patients and across racial groups?

## Data

**Source.** NYSDOH SPARCS Hospital Inpatient Discharges (de-identified). Each row is one inpatient stay, with 33 columns.

| Year | Stays | Dataset |
| :--- | :--- | :--- |
| 2023 | 2,125,754 | [`46xm-urtu`](https://health.data.ny.gov/d/46xm-urtu) |
| 2024 | 2,196,737 | [`sf4k-39ay`](https://health.data.ny.gov/d/sf4k-39ay) |

- **Years:** 2023 and 2024, pooled, with a year indicator.
  - 2021–22 are excluded because of COVID-era noise.
  - 2025 isn't published yet.
  - The 2023–24 Medicaid "unwinding" spans both years, so it isn't a reason to drop 2023.
- **Privacy:** the publisher de-identifies the data. There is no patient ID, age is in 5 bands, ZIP is cut to 3 digits, there are no dates beyond the discharge year, and abortion records have their location fields blanked.
- **Terms:** the [OPEN-NY Terms of Use](https://data.ny.gov/download/77gx-ii52/application/pdf) permit reuse and redistribution of derived figures without pre-approval. Credit NYSDOH SPARCS, and keep State logos and seals out of the paper.

**Scope.**

| Rows | Decision |
| :--- | :--- |
| Emergency, urgent and trauma admissions | **Keep.** About 3.2M stays across both years |
| Elective admissions | Drop. Their length of stay is largely planned |
| Newborns | Drop. Their stays work differently |
| "Not Available" admission type | Drop |
| 10 specialty facilities (rehab, long-term care) | Show their length-of-stay distribution in the exploratory analysis, then most likely drop them |

The analysis will probably use a random subsample of a few hundred thousand rows, so the slower models train in reasonable time. Make sure the fairness subgroups stay large enough.

**Cleanup when pooling the two years:**
- Rename `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`.
- Strip the thousands commas from the 2023 charge columns.
- Convert the text columns to numbers. Length of stay is stored as text, and its top value is "120 +" in 2023 and "120+" in 2024.

## Target and features

- **Target:** $y = 1$ if length of stay > 7 days (LOS ≥ 8), else 0. About 23% of stays are positive.
- **Output:** every model predicts a probability, not just a label.

**Features.** These are roughly what staff know near the time of admission:
- Age group, gender, race, ethnicity
- 3-digit ZIP code
- Payer (primary, secondary, tertiary)
- Admission type and ED indicator
- Facility, county and health service area
- APR MDC (body system), medical vs. surgical, and CCSR principal diagnosis
- Discharge year

MDC, medical/surgical and diagnosis are coded at discharge. They stand in for the working diagnosis staff have at admission, and the paper should say so.

**Never used as features:**

| Column | Why |
| :--- | :--- |
| Total charges, total costs | They grow with every day of the stay, so they're essentially the target in dollars |
| Patient disposition | Only known at discharge |
| APR severity of illness, risk of mortality | Computed from all discharge diagnoses, including complications that arose during the stay |
| APR-DRG, CCSR procedure | Partly a response to how the stay went |
| Birth weight | Almost always blank once newborns are dropped |

## Split

- **Train/test:** a random 80/20 split, stratified on the target.
  - This isn't time-series data: rows are near-independent stays with no dates or patient IDs. So a random split is an honest evaluation, and no temporal holdout is needed.
- **Tuning:** hyperparameters are tuned with K-fold cross-validation on the training set.
- **Recalibration slice:** a slice of the training set is held back to fit the recalibrators (Platt scaling and isotonic regression).
- **Test set:** used once, at the end. Every model is scored on the same test rows.

## Models

| Model | Course module | Notes |
| :--- | :--- | :--- |
| Logistic regression (unpenalized) | Classification | **Baseline** |
| Lasso, ridge, elastic net logistic regression | Regularization | Penalty strength (and the elastic net's mixing parameter) tuned by CV |
| Decision tree | Trees & ensembles | Single readable tree; shows what the ensembles improve on |
| Random forest | Trees & ensembles | |
| Gradient-boosted trees (XGBoost) | Trees & ensembles | |
| MLP | Neural networks | |
| Stacking ensemble | Trees & ensembles | A logistic regression that combines the other models' predictions |

All features are categorical. One-hot encode them, grouping rare levels into "Other", and never use integer label-encoding. Fit the encoders on the training data only.

## Evaluation (test set)

1. **AUC: ranking.**
   - **What it measures:** pick one patient who stayed long and one who didn't, at random. AUC is the chance the model gives the long-stay patient the higher risk.
   - **Scale:** 0.5 is a coin flip. Published admission-time models reach about 0.75–0.81. Above 0.9 would suggest leakage.
   - **What it ignores:** whether the probabilities themselves are right.
2. **Calibration curves: are the probabilities literally true?**
   - **How the curve is built:** bin test patients by predicted risk, then plot the mean prediction against the observed long-stay rate.
   - **Reading it:**
     - On the diagonal: well calibrated.
     - Below: the model over-predicts.
     - Above: it under-predicts.
     - An S-shape: the model is over-confident.
   - **Recalibration:**
     - **Methods:** two recalibrators, each fit on the held-back training slice:
       - Platt scaling: a one-variable logistic regression on the model's score. It fits a single S-shaped curve, so it fixes simple over- or under-confidence.
       - Isotonic regression: a non-decreasing step function of the model's score. It can fit any monotone shape, so it shows whether a model has calibration errors Platt scaling can't fix.
     - **What to plot:** calibration curves before and after each recalibrator, for every model.
     - **Effect on AUC:** none for Platt scaling, because it doesn't change the order of patients. Isotonic regression maps nearby scores to the same value, and those ties can shift AUC slightly; report the change.
     - **Expected result:** boosted trees improve the most.
3. **Fairness.**
   - **What to compute:** calibration curves and AUC repeated by payer (Medicaid vs. commercial = Private + Blue Cross, with Medicare for context) and by race.
   - **Calibration by group:** a group whose curve sits above the diagonal has its risk underestimated.
   - **AUC by group:** a lower AUC means the model is less useful for that group.
   - **Why not equal error rates:** calibration is the headline check. When base rates differ between groups, calibration and equal error rates can't both hold.

**Benchmarks for the paper:**
- [Zeleke et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426288/): AUC 0.754 for LOS > 6 days, using triage data only. Gradient boosting and logistic regression were nearly tied. See Table 4.
- [Rajkomar et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550175/): AUROC 0.80–0.81 at admission for LOS ≥ 7 days, using full electronic health records. See Table 2.

Mixed or null results, such as the flexible models not beating logistic regression or no gap between groups, are reported as findings.

## Caveats to state in the paper

- **Discharge-coded stand-ins:** the diagnosis features are coded at discharge, so they make performance somewhat better than a true admission-day model would get.
- **Race:** recorded unreliably in SPARCS, and "Other Race" (about 28%) lumps Asian, Pacific Islander and Native American patients together.
- **No patient ID:** repeat stays by the same person can't be linked.
- **Group gaps:** a payer gap may reflect discharge barriers, such as waiting for a nursing-home bed, rather than clinical need. Report it as an association, not a cause.
- **Payer field:** uninsured patients are sometimes enrolled in Medicaid during a long stay.

## Open decisions

- [ ] Race/ethnicity grouping for the fairness check, including what to do with "Other Race".
- [ ] Where "Managed Care, Unspecified" goes among the payer groups.
- [ ] Subsample size.
- [ ] Specialty-facility exclusion, to be confirmed in the exploratory analysis.

## Optional extras (not committed)

- **Hindsight comparison:** add APR-DRG, procedure, severity and mortality risk as a second feature set, and report how much they inflate AUC.
- **Temporal check:** train on 2023, test on 2024.
- **Psychiatric per diem appendix:** length-of-stay histograms for psychiatric stays (MDC 19) by payer, looking for spikes at payment-change days. This is descriptive only; see `2026-10-15-psych-per-diem.md`.

## Deliverables

| Stage | Due |
| :--- | :--- |
| Proposal (`docs/course_materials/proposal.md`) | Fri 16 Oct 2026 |
| Early draft | Fri 6 Nov 2026 |
| Paper (≤ 6 pages, NeurIPS format) + 5-minute video | Fri 11 Dec 2026 |
