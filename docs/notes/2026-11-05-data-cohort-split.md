# Data, Cohort and Split

Created: 2026-11-05
Scope: Decisions for getting the data, building the cohort and sample, and splitting it (steps 1–3 toward the baseline model).

## Code

- All code goes in `analysis/baseline_model.py`, paired with `analysis/baseline_model.ipynb`.
- `analysis/verify_data.py` stays as it is.
- Random seed: 42.
- The notebook must explain itself. Before any step that filters rows, show a simple look at the data and say what we're seeing and what we're doing about it.

## Step 1: Get the data

- Download the official 2023 and 2024 CSVs into `data/raw/`, unless they're already there.
- Stop with an error unless the files have these shapes, which match the 30 Oct run of `verify_data.py`:

| Year | Rows | Columns |
| :--- | ---: | ---: |
| 2023 | 2,125,754 | 33 |
| 2024 | 2,196,737 | 33 |

## Step 2: Build the cohort and sample

This step only lines up the columns, chooses the rows and creates the target. It replaces the pooling cleanup in the analysis plan: comma stripping isn't needed because charges aren't used, and a year indicator isn't added because the raw data already has `Discharge Year`.

1. Pool the two years, converting column names to snake_case and renaming `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`.
2. Create the target: $y = 1$ if length of stay ≥ 8, treating "120 +" and "120+" as 120.
3. Remove stays that are out of scope:

| Rule | Removes | Stays removed |
| :--- | :--- | ---: |
| Admission type isn't emergency, urgent or trauma | Elective, newborn and "Not Available" admission types | 1,111,319 |
| APR MDC code is 15 | Newborns admitted under the other admission types | 10,250 |
| `permanent_facility_id` is blank | Records with facility and location redacted by the publisher (`facility_name` contains "Redacted"), because the facility is known in any real use of the model | 8,640 (34 are also newborns) |
| `permanent_facility_id` is on the publisher's specialty list | Facilities whose discharges are mostly rehab or life support: 000340, 000775, 000831, 000968, 001046, 001138, 001460, 001486, 009431, 010223 | 176 |

4. Check the cohort, stopping with an error if a count differs:
   - 3,192,140 stays: 1,562,648 in 2023 and 1,629,492 in 2024.
   - No unparseable length-of-stay values.
   - About 23.09% positive.
5. Draw one simple random sample of 500,000 stays. This settles the plan's open decision on subsample size. All models use this sample.

Kept on purpose:
- **Patients who died:** 98,675 stays (45.4% long). Death isn't known at admission, so removing them would use future information.
- **Exact duplicate rows:** 184 extra rows. They can't be identified as errors without patient IDs.
- **Other facilities with many long stays:** only the publisher's list defines specialty facilities. For example, six sites with the highest long-stay rates (11,990 stays, 78% long) stay in, because choosing sites by long-stay rate would select on the outcome.
- **Blank ZIP codes outside redacted records:** 50,889 stays. ZIP is the patient's home area, so these are handled in cleaning.

To state in the paper: the newborn, redacted and specialty exclusions, the kept long-stay facilities, and that stays are cut short for patients who died.

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
- **Labels that differ by year:** the first cleaning task is to use one spelling across years for age ("0 to 17" vs. "0-17", then check for 5 groups), service area (`Capital/Adirond` vs. `Capital/Adirondacks`) and county (`Manhattan` vs. `New York`, `St Lawrence` vs. `Saint Lawrence`). Facility, CCSR diagnosis and MDC use their code columns, because their descriptions differ by year.
- **EDA, other cleaning, encoding and the baseline fit:** these come after steps 1–3.
