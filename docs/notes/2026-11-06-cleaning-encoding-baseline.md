# Cleaning, Encoding and Baseline

Created: 2026-11-06
Scope: Decisions for cleaning and encoding the training data, fitting the baseline model and evaluating it (steps 4–6 toward the baseline checkpoint), plus what's still open.

## Ground rules

- **Train only:** EDA and anything learned from the data use the training set. The same fixes are applied unchanged to the recalibration and test sets when those are used.
- **Discovery in the notebook:** before each fix, show the simplest look at the training data that explains it. A decision shouldn't appear only in this note.
- **Libraries:** scikit-learn for encoding, CV, models and metrics. Pandas for EDA and label fixes, which scikit-learn doesn't do.

## Features

**Used (17)**

| Column | Plain meaning |
| :--- | :--- |
| `age_group` | Five age bands |
| `gender` | M, F or U (unknown) |
| `race` | White, Black/African American, Multi-racial or Other |
| `ethnicity` | Hispanic, not Hispanic, multi-ethnic or unknown |
| `zip_code` | First 3 digits of the patient's home ZIP, or out of state |
| `payment_typology_1` | Main payer |
| `payment_typology_2` | First backup payer |
| `payment_typology_3` | Second backup payer |
| `type_of_admission` | Emergency, urgent or trauma |
| `emergency_department_indicator` | Whether the bill shows an emergency department charge |
| `permanent_facility_id` | Which hospital |
| `hospital_county` | The hospital's county |
| `health_service_area` | The hospital's region |
| `apr_mdc_code` | Body system involved |
| `apr_medical_surgical_description` | Whether the case was medical or surgical |
| `ccsr_diagnosis_code` | Main reason for the stay |
| `discharge_year` | 2023 or 2024 |

MDC, medical/surgical and CCSR diagnosis are recorded at discharge. They stand in for the working diagnosis at admission, and the paper will say so.

**Never used as features (11)**

| Column | Plain meaning | Why not |
| :--- | :--- | :--- |
| `total_charges`, `total_costs` | The bill and its estimated cost | They grow with each day, so they nearly reveal the answer |
| `patient_disposition` | Where the patient went afterward, including death | Only known at discharge |
| `apr_severity_of_illness_code`, `apr_severity_of_illness_description` | How sick the patient was, scored 1–4 | Includes complications that arose during the stay |
| `apr_risk_of_mortality` | Likelihood of dying, scored 1–4 | Same problem |
| `apr_drg_code`, `apr_drg_description` | Billing category for the whole stay | Partly reflects what happened during the stay |
| `ccsr_procedure_code`, `ccsr_procedure_description` | Main procedure performed | Often a response to how the stay went |
| `birth_weight` | Weight at birth | Almost always blank once newborns are removed |

**Not features for other reasons (5)**

- **Target:** `length_of_stay`.
- **Duplicates:** `facility_name`, `operating_certificate_number`, `ccsr_diagnosis_description` and `apr_mdc_description`.

## Step 4: Clean

1. **Fix labels that differ by year.** Show each feature's levels by year in train, then use one spelling. Expected fixes:

   | Column | Change | Levels after |
   | :--- | :--- | ---: |
   | `age_group` | "0 to 17" → "0-17", and the same for the other bands | 5 |
   | `health_service_area` | `Capital/Adirond` → `Capital/Adirondacks` | 8 |
   | `hospital_county` | `Manhattan` → `New York`, `St Lawrence` → `Saint Lawrence` | 57 |

2. **Use codes, not descriptions.** Show that each code has one description within a year in train, while the descriptions differ across years. One-hot encode the codes, and use the descriptions only to label results. We aren't analyzing how categories change from year to year.

3. **Make blanks their own level.** Show the blank share and long-stay rate for ZIP and the three payer columns in train, then fill blanks with "Unknown".
   - A one-hot "Unknown" column already marks that the value was blank and gets its own coefficient, so we don't add a separate missingness indicator.
   - A blank payer 2 or 3 probably means there's no backup payer, rather than a missing value.

## Step 5: Encode

- One-hot encode all 17 features with scikit-learn's `OneHotEncoder`. Never use integer label-encoding.
- Group rare levels into "Other". New levels in other sets also go to "Other". The cutoff is still open (see below).
- **Baseline only:** use `drop="first"`, which drops the alphabetically first level that isn't rare as the reference. This is revisited for the penalized and tree models.
- Keep all overlapping features, such as facility and county. Their coefficients won't be interpreted, because this overlap means they still aren't uniquely identified after dropping a level.

## Step 6: Fit and evaluate the baseline

- **Model:** unpenalized logistic regression, with no hyperparameters to tune. Check that the fit converges.
- **CV:** stratified 5-fold on the training set, seed 42. Encoding is refit inside each fold using a scikit-learn `Pipeline`.
- **Recalibration and test sets:** not used in this step.

**Checkpoint report**

- AUC and Brier score, each as the mean and spread across the 5 folds.
- One calibration curve from the out-of-fold predictions.
- AUC compared with the published range of 0.75–0.81.

This brings the Brier score back in, which the 30 Oct plan had left out. Accuracy isn't used, because always predicting "not long" already scores about 77%.

**The metrics in plain terms**

- **AUC:** pick one long stay and one short stay at random. AUC is the chance the model gives the long stay the higher risk. 0.5 is a coin flip and 1.0 is perfect. It only measures the ordering, not whether the probabilities are right.
- **Calibration:** of the patients given about a 30% risk, do about 30% actually stay long? The curve groups patients by predicted risk, then plots average predicted risk against the share who actually stayed long. On the diagonal, the probabilities mean what they say.
- **Brier score:** the average squared gap between the predicted probability and what happened (1 if long, 0 if not). Lower is better. Saying 80% costs 0.04 if the stay was long and 0.64 if it wasn't. Giving everyone the overall 23% rate scores about 0.177, so a useful model must beat that.

**Error bars:** the spread across folds for now. Bootstrap intervals on the test set come at the end.

## Still open

- **Rare-level cutoff:** must be decided before the baseline is fit. It needs to be set before seeing any model results.
- **Diagnoses with almost no long stays:** PRG026 has 0 long stays among 1,171 in train, so its coefficient has no finite best value in an unpenalized model. Decide this with the cutoff.
- **Recall and precision:** whether to report them for awareness only, not for choosing models, and at which threshold.
- **Race grouping and "Managed Care, Unspecified":** needed for the fairness analysis, not for the baseline.

## Not included

- **Other imputation strategies:** we could test one later with CV inside the pipeline on train. It wouldn't be KNN, which suits this many one-hot categorical columns poorly.
- **Year-to-year category changes:** out of scope, including facility names that appear under two IDs and reworded MDC categories.
