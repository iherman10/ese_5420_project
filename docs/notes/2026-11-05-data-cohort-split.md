# Data, Cohort and Split

Created: 2026-11-05
Scope: Decisions for getting the data, building the cohort and sample, and splitting it (steps 1–3 toward the baseline model).

## Code

- All code goes in `analysis/baseline_model.py`, paired with `analysis/baseline_model.ipynb`.
- `analysis/verify_data.py` stays as it is.
- Random seed: 5420.

## Step 1: Get the data

- Download the official 2023 and 2024 CSVs into `data/raw/`, unless they're already there.
- Stop with an error unless the files have these shapes, which match the 30 Oct run of `verify_data.py`:

| Year | Rows | Columns |
| :--- | ---: | ---: |
| 2023 | 2,125,754 | 33 |
| 2024 | 2,196,737 | 33 |

## Step 2: Build the cohort and sample

This step only lines up the columns, chooses the rows and creates the target. It replaces the pooling cleanup in the analysis plan: comma stripping isn't needed because charges aren't used, and a year indicator isn't added because the raw data appears to have `discharge_year` already (to confirm after download).

1. Pool the two years, renaming `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`.
2. Keep emergency, urgent and trauma admissions.
3. Create the target: $y = 1$ if length of stay ≥ 8, treating "120 +" and "120+" as 120.
4. Check the cohort, stopping with an error if a count differs:
   - 3,211,172 stays: 1,572,202 in 2023 and 1,638,970 in 2024.
   - No unparseable length-of-stay values.
   - About 23% positive.
5. Draw one simple random sample of 500,000 stays. This settles the plan's open decision on subsample size. All models use this sample.

## Step 3: Split

Split the sample into three sets, stratified on $y$:

| Set | Rows | Used for |
| :--- | ---: | :--- |
| Train | 360,000 | Fitting, and K-fold CV for tuning |
| Recalibration | 40,000 | Fitting Platt scaling and isotonic regression only |
| Test | 100,000 | Final scoring, once, on the same rows for every model |

- There is no separate validation set, because K-fold CV does that job.
- The test set stays untouched until the end, so the baseline checkpoint score comes from CV on the training set.
- After splitting, check the positive rate in each set and print the test-set counts by payer and race.

## Not included

- **Extra Multi-racial test rows:** we won't score Multi-racial stays from outside the sample. The group will have about 1,300 test rows, so its results will be noisier (estimated AUC within about ±0.03 and calibration bins within about ±7 points, at 95%). The paper will say so.
- **Specialty facilities:** this waits for EDA. If they're excluded, the same publisher-listed facilities come out of all three sets.
- **Age labels:** the first cleaning task is to use one spelling across years ("0 to 17" vs. "0-17") and check for 5 age groups.
- **EDA, other cleaning, encoding and the baseline fit:** these come after steps 1–3.
