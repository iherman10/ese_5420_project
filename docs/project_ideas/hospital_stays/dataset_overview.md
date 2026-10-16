# SPARCS Inpatient Discharges: Dataset Overview

*Compiled 2026-10-15 from the official documentation (Socrata metadata, data dictionary, overview PDF, release notes) and published literature. A few aggregate count queries against the public API filled gaps in the documentation; no record-level data has been downloaded yet.*

## Summary

*Decisions in this summary supersede the detailed research sections below. In particular, the M0–M4 feature ladder was replaced by a main model plus one alternate model, and the DRG and procedure were moved out of the main model.*

### The data in one paragraph

New York State publishes a record of every inpatient hospital stay in the state, one row per stay. We'll use **2023 and 2024 pooled: about 4.3 million stays, 33 columns**. Keeping only unplanned admissions (emergency, urgent, trauma) leaves roughly 3.3 million. It's free and fairly clean. Patient privacy is protected by coarsening: age is in bands, ZIP is cut to 3 digits, there are no dates beyond the year, and there's no patient ID. **About 20% of stays last longer than 7 days**, which is the outcome we'll predict. The typical stay is 3 days.

Pooling the two years needs a little cleanup:
- rename two columns;
- strip commas from the 2023 charges;
- add a year indicator to absorb any small year-to-year differences.

**Why keep 2023, despite the Medicaid "unwinding".** During COVID, states got extra federal money on the condition that they didn't remove anyone from Medicaid. That freeze ended in spring 2023, and New York restarted yearly eligibility checks from mid-2023, working through enrollees over roughly a year. Some people moved to the Essential Plan, an employer plan, or no insurance.
- **Expected effect:** a slightly smaller Medicaid share, and possibly a sicker remaining Medicaid group, since those who left were often healthier or higher-income.
- **Why it doesn't argue for dropping 2023:** the unwinding ran well into 2024, so both years are in the transition.
- **Safeguards:** the year indicator absorbs any overall shift. The exploratory analysis compares payer mix by year and checks how Essential Plan stays are coded in the payer field.

### Acronyms

| Acronym | Stands for | What it is |
| :--- | :--- | :--- |
| SPARCS | Statewide Planning and Research Cooperative System | New York's hospital discharge reporting system, and the name of the dataset |
| CCSR | Clinical Classifications Software Refined | A federal (AHRQ) tool that groups the ~70,000 detailed ICD-10 diagnosis codes, and separately the procedure codes, into a few hundred clinically meaningful categories |
| AHRQ | Agency for Healthcare Research and Quality | The U.S. federal agency that maintains CCSR |
| ICD-10 | International Classification of Diseases, 10th revision | The standard detailed coding system for diagnoses (ICD-10-CM) and inpatient procedures (ICD-10-PCS) |
| APR | All Patient Refined | Prefix for the 3M software family that groups stays and scores severity. "All patient" means it covers every payer, not just Medicare |
| DRG | Diagnosis Related Group | A billing category for the whole stay, built from diagnosis and procedure codes. Hospitals are paid by it |
| APR-DRG | All Patient Refined Diagnosis Related Group | 3M's version of the DRG, used by New York Medicaid and in SPARCS |
| MDC | Major Diagnostic Category | A coarse body-system grouping of DRGs, e.g. circulatory or respiratory |
| ED / ER | Emergency Department / Emergency Room | Same thing |
| ZIP | Zone Improvement Plan | U.S. postal code |
| OOS | Out of State | Value used in the ZIP column for non-New York patients |
| AUC | Area Under the (ROC) Curve | Ranking metric: 0.5 = coin flip, 1.0 = perfect |
| ROC | Receiver Operating Characteristic | The curve of true-positive rate vs. false-positive rate as the threshold moves |
| XGBoost | eXtreme Gradient Boosting | A popular boosted-trees library |
| MLP | Multilayer Perceptron | A basic feed-forward neural network |

### What each variable means, and whether we use it

**Known when the patient arrives: used in the main model**

| Variable | Plain meaning |
| :--- | :--- |
| Age Group | Patient's age band: 0–17, 18–29, 30–49, 50–69, 70+ |
| Gender | M, F or U (unknown) |
| Race | White, Black, Multi-racial or "Other". Other also covers Asian and Native American patients |
| Ethnicity | Hispanic, not Hispanic, multi-ethnic or unknown |
| Zip Code (ZIP = Zone Improvement Plan) | First 3 digits of the patient's home ZIP, or "OOS" (Out of State) |
| Payment Typology 1 / 2 / 3 | Who is expected to pay (Medicare, Medicaid, private insurance, Blue Cross, self-pay, …). 1 is the main payer; 2 and 3 are backups, usually blank. Small caveat: uninsured patients on long stays are sometimes enrolled in Medicaid *during* the stay. Worth a sentence in the paper because payer is the fairness variable |
| Type of Admission | How the patient came in: emergency, urgent or trauma. Elective and newborn stays are excluded; see below |
| Emergency Department (ED) Indicator | Whether the stay started in the emergency room |
| Facility / County / Health Service Area | Which hospital, its county, and its region of the state |

Birth Weight is dropped: once newborns are removed it's almost always blank.

**Recorded at discharge, but a reasonable stand-in for what staff know at admission: used in the main model**

Ranked from least to most leakage concern. All are low to moderate.

| Variable | Plain meaning | Leakage concern |
| :--- | :--- | :--- |
| APR MDC (All Patient Refined Major Diagnostic Category) | Body system the stay is about, in 26 groups (e.g. circulatory, respiratory) | Low. Taken from the principal diagnosis, which usually agrees with the admission diagnosis at the body-system level. Exception: a few procedure-based groups (transplant, tracheostomy, prolonged ventilation) |
| APR (All Patient Refined) Medical/Surgical | Whether the case was medical or surgical | Low–moderate. Known in advance for elective surgery and usually early for urgent surgery. A complication that leads to an unplanned operation flips it |
| CCSR Diagnosis (Clinical Classifications Software Refined) | The main reason for the stay, in about 480 categories (e.g. "heart failure", "sepsis"). The first 3 letters give the body system (e.g. `CIR` = circulatory) | Moderate. Disagrees with the admission diagnosis in about 12–28% of cases, and mismatched cases tend to stay longer |

**Too much hindsight: never in the main model, only in the alternate model (sensitivity analysis)**

Ranked from least to most concerning.

| Variable | Plain meaning | Why it's excluded |
| :--- | :--- | :--- |
| CCSR Procedure (Clinical Classifications Software Refined) | The main procedure performed, if any (e.g. "C-section", "knee replacement"), in about 320 categories. Blank for 29% of stays | For emergency admissions it's often a *response* to how the stay went, e.g. tracheostomy after long ventilation, or a return to surgery |
| APR-DRG (All Patient Refined Diagnosis Related Group) | A billing category for the whole stay, combining diagnosis and procedure (about 340 groups) | Inherits the procedure problem. Some groups are defined by long ventilation or tracheostomy. What it adds beyond the diagnosis is mostly hindsight |
| APR (All Patient Refined) Risk of Mortality | The likelihood of dying, scored 1 (minor) to 4 (extreme) | Computed from all diagnoses at discharge, including complications that arose in hospital |
| APR (All Patient Refined) Severity of Illness | How sick the patient was, scored 1–4 the same way | Same construction, and more directly tied to resource use. Complications both raise it and lengthen the stay |

**Never used in any model**

| Variable | Plain meaning | Why |
| :--- | :--- | :--- |
| Patient Disposition | Where the patient went at the end: home, nursing home, died, etc. | Only known at discharge, and strongly tied to long stays |
| Total Charges / Total Costs | The bill and its estimated cost | They grow with each day, so they're essentially the target in dollars |
| Length of Stay | **The target.** Days in hospital, with 120+ grouped together. "Long stay" = more than 7 days | |

### Which stays are in scope

| Admission type | Approx. 2024 rows | Decision |
| :--- | :--- | :--- |
| Emergency | 1.46M | Keep |
| Urgent | 169k | Keep |
| Trauma | 10k | Keep |
| Elective | 356k | **Drop**: length of stay is largely planned in advance |
| Newborn | 198k | **Drop**, since newborn stays work differently |
| Not Available | 3.7k | Drop |

**Scope: unplanned admissions only.** Predicting long stays is most useful where the stay isn't scheduled. Keeping electives wouldn't help predictions for unplanned admissions:
- Unplanned admissions already number about 1.6M per year.
- Elective stays are driven by the scheduled procedure rather than by how sick the patient is on arrival.

The commercial-insurance group remains in the hundreds of thousands, which is plenty for the fairness check.

### Caveats to handle or note in the paper

- **Specialty facilities:** ten rehab and long-term facilities have very long stays. Show their length-of-stay distribution against typical hospitals in the exploratory analysis, then most likely exclude them.
- **Race:** recorded unreliably by many hospitals, and Asian patients can't be separated out. Race is kept as a feature and in the fairness analysis, with this caveat in the paper.
- **No patient ID:** repeat visits by the same person can't be detected. Note this in the paper.

### Models and evaluation

- **Main model:** everything known on arrival, plus MDC, medical/surgical and CCSR diagnosis as stand-ins for what staff know at admission.
- **Alternate model (sensitivity analysis):** the main model plus CCSR procedure, APR-DRG, risk of mortality and severity of illness. It's reported at the end as a best case that partly reflects hindsight, and the gap to the main model shows how much the hindsight features inflate performance.
- **Methods:** logistic regression/lasso, random forest, XGBoost, optional MLP.
- **Split:** random 80/20 train/test, with cross-validation on the training set for tuning.
- **Evaluation:**
  - **Ranking:** AUC, i.e. does the model put long stays above short ones?
  - **Calibration and Brier score:** are the predicted probabilities literally true?
- **Fairness:** calibration checked separately for Medicaid vs. commercial insurance and by race.
- **Benchmarks:** published admission-time models for stays of 7+ days reach AUC ≈ 0.75–0.81. Because the main model uses discharge-coded stand-ins, expect it to score somewhat higher than a truly admission-time model would.

### Still to decide

- Exact payer and race groupings for the fairness check, deferred to the analysis stage. Commercial = private insurance + Blue Cross; "Managed Care, Unspecified" needs a home.

*The sections below give the full research detail and sources.*

---

## At a glance

| | 2024 (main) | 2023 (optional) |
| :--- | :--- | :--- |
| Dataset ID | [`sf4k-39ay`](https://health.data.ny.gov/d/sf4k-39ay) | [`46xm-urtu`](https://health.data.ny.gov/d/46xm-urtu) |
| Rows (1 row = 1 discharge) | 2,196,737 | 2,125,754 |
| Columns | 33 | 33 |
| Share with LOS > 7 days | ≈ 20.1% | ≈ 20.0% |
| Median LOS | 3 (mode 2) | 3 (mean 5.8, SD 8.8) |
| Diagnosis/procedure grouping | CCSR v2025.1 | CCSR v2025.1 |
| APR-DRG version | v41 | v40 |

The LOS shares come from the portal's cached value counts, not from the raw data.

## Columns, grouped by when they are known

### Known at admission: safe features

| Column | Meaning and coding | Notes |
| :--- | :--- | :--- |
| Health Service Area | Hospital region, 8 levels | Blank for redacted (abortion) records |
| Hospital County | 57 levels | Same redaction |
| Permanent Facility Id | Hospital ID, 205 levels | Same redaction. Facility name and operating certificate number duplicate it |
| Age Group | 0–17, 18–29, 30–49, 50–69, 70+ | Exact age not released |
| Zip Code | First 3 digits, or "OOS" for out of state | Blank for about 1.9% of rows (small-population areas, small cells, redaction) |
| Gender | M, F, U | U is 385 rows |
| Race | White, Black/African American, Multi-racial, Other Race | "Other Race" (≈ 28%) includes Asian, Pacific Islander and Native American patients. Known to be noisy (see quality issues) |
| Ethnicity | Spanish/Hispanic, Not Span/Hispanic, Multi-ethnic, Unknown | Unknown ≈ 13% |
| Type of Admission | Emergency 1.46M, Elective 356k, Newborn 198k, Urgent 169k, Trauma 10k, Not Available 3.7k | |
| Payment Typology 1 | Primary payer: Medicare 929k, Medicaid 636k, Private 318k, Blue Cross/Blue Shield 204k, Self-Pay 35k, Managed Care Unspecified 30k, Misc/Other 22k, Federal/State/Local/VA 21k, Dept of Corrections 1.6k | Commercial = Private + BCBS ≈ 522k |
| Payment Typology 2 and 3 | Secondary and tertiary payer | Null for 55% and 89% of rows. Having one at all is informative (e.g. dual Medicare/Medicaid eligibility) |
| Emergency Department Indicator | Y/N: whether the bill had an ED revenue code | Y is 1.38M rows |
| Birth Weight | Grams, rounded to the nearest 100 g | Filled almost only for newborns. "UNKN" means unknown |

### Coded at discharge: the open decision

These are assigned from the final coded record, after the stay. Using them makes the model "given the discharge record", not "at admission".

| Column | Meaning and coding | Notes |
| :--- | :--- | :--- |
| CCSR Diagnosis Code / Description | Category of the principal ICD-10-CM diagnosis, about 478 levels | Least leaky of the group, since it approximates the reason for admission |
| CCSR Procedure Code / Description | Category of the principal ICD-10-PCS procedure, about 320 levels | Empty for 29% of rows (no procedure) |
| APR DRG Code / Description | 3M APR-DRG grouping, about 336 levels | Built from the whole discharge record |
| APR MDC Code / Description | Major diagnostic category (body system), 26 levels | Coarse grouping derived from the DRG |
| APR Severity of Illness | 0 Undetermined, 1 Minor, 2 Moderate, 3 Major, 4 Extreme | Uses all discharge diagnoses, including complications that arose during the stay |
| APR Risk of Mortality | Same 0–4 scale | Same concern |
| APR Medical Surgical Description | Medical / Surgical / Not Applicable | Derived from the DRG |

### Leakage: never use as features

| Column | Why |
| :--- | :--- |
| Total Charges | Grows mechanically with days. Candidate second target, as log charges |
| Total Costs | Charges × facility cost-to-charge ratio, so a near-copy of charges. Also a leak for a charges model |
| Patient Disposition | 19 discharge destinations (Home, SNF, Expired, Left AMA, rehab, hospice …), only known at discharge |

### Target and constants

- **Length of Stay:** (discharge date − admission date) + 1, so a same-day stay = 1 and "> 7" means LOS ≥ 8.
  - Stored as text.
  - Values are "1"–"119" plus "120+" (2,404 rows).
- **Discharge Year:** constant within a file.

## Licence and terms of use

*Checked 2026-10-15.*

**Verdict:** the OPEN-NY Terms of Use permit our use, including publishing figures derived from the data, with no pre-approval.

- **Licence field:** the 2024 metadata gives the licence only as "See Terms of Use" (`SEE_TERMS_OF_USE`), with no link to the terms. Attribution: "New York State Department of Health", linking to health.ny.gov/statistics/sparcs. The 2023 page hasn't been checked but is almost certainly the same.
- **Governing terms:** OPEN-NY Terms of Use, last modified 8 March 2013. The copy we read is hosted on data.ny.gov. We assume health.data.ny.gov, part of the same OPEN-NY program, uses the same terms; its footer link hasn't been checked.
- **Licence grant:** "a non-exclusive, revocable license to use the Content … in a manner consistent with the Terms of Use." The preamble says to use it "however you wish," as long as the use is lawful.
- **Redistribution and derived figures:** allowed. The terms "do not contain restrictions requiring members of the public to use attribution, to re-post the license terms with any re-uses of the data, to impose share-alike or technical restrictions, nor require the public to obtain pre-approval before re-use."
- **Attribution:** not required. We'll credit NYSDOH SPARCS anyway.
- **Exclusions:** State logos, the State seal and trademarks are not licensed, so keep them out of the paper. The State can revoke the licence or tell a user to stop using the data.
- **No warranty:** the State makes no warranty about accuracy, and doesn't vouch "for the quality or accuracy of any analyses or re-uses."
- **Dataset-specific terms:** the ToU allow other terms to be imposed on specific Content. The 2024 Overview PDF has none: no rules on use, citation, re-identification or small cells. The data dictionary and release notes haven't been checked.
- **Data use agreement:** the SPARCS data use agreement, which bans attempts to identify individuals, applies only to the limited and identifiable tiers. The public de-identified file has none. We'll still avoid publishing very small cells.

**Proposal wording:** "Public, de-identified NYSDOH SPARCS file on health.data.ny.gov, released under the OPEN-NY Terms of Use, which permit reuse and redistribution of derived figures without pre-approval; attribution to NYSDOH SPARCS given as good practice."

## Transformations the publisher already applied

- **Dates:** only the discharge year is released, so there is no month, day of week or seasonality.
- **Diagnoses and procedures:** only the principal ones, as CCSR categories rather than ICD codes. Secondary diagnoses, secondary procedures and revenue codes are removed.
- **Age:** five bins. **ZIP:** three digits. **Birth weight:** rounded to 100 g.
- **Race:** collapsed to 4 values. **Payer:** grouped into the categories above; the mapping from raw codes isn't documented.
- **Costs:** estimated from charges using audited cost-to-charge ratios. 2024 uses the 2021 ratios and 2023 the 2019 ratios. When a ratio exceeds the Medicare ceiling, the hospital average is used.
- **Abortion records** get "enhanced de-identification": facility, county, region and ZIP are blanked, and facility name reads "Redacted for Confidentiality". About 5,300 rows (0.24%).

## Quality issues and caveats

- **Everything is stored as text,** including LOS, charges, costs, birth weight and facility ID. Numeric columns need converting.
- **No patient ID.** The same patient can appear several times (readmissions, transfers) and can't be linked. Rows are therefore not fully independent, and there's no prior-utilization feature.
- **Race is noisy.** A NYC health department report (Reilly et al. 2018) found SPARCS race agreed poorly with Medicaid and vital records: below 70% at most NYC facilities, and below 50% at over a third.
- **Newborns are about 9% of rows.** Their stays follow a different mechanism, and Medicaid covers most births, so including them shifts the Medicaid vs. commercial comparison.
- **Specialty facilities.** The publisher flags 10 facilities whose discharges are mostly rehab or life support, so they have very long stays:

  | Facility Id | Facility |
  | :--- | :--- |
  | 000340 | United Memorial Bank St |
  | 000775 | Helen Hayes |
  | 000831 | Sunnyview |
  | 000968 | Garnet Health Catskills G. Hermann |
  | 001046 | Burke Rehab |
  | 001138 | Blythedale Children's |
  | 001460 | NY Eye & Ear |
  | 001486 | Henry J. Carter Specialty |
  | 009431 | St. Peter's Addiction Recovery |
  | 010223 | Calvary Hospital |

- **Provisional and incomplete.** The data is labelled provisional. Some facilities didn't fully meet reporting requirements; the documentation doesn't say which ones.
- **Zero and odd values.** Charges and costs have $0 rows. Prior work also found rows where costs exceed charges.
- **Small "unknown" levels.** "Not Available" admission type (3.7k), "Undetermined" severity and mortality risk (456), blank primary payer (227).

### If pooling 2023 and 2024

- Rename two columns: `hospital_service_area` → `health_service_area` and `zip_code_3_digits` → `zip_code`.
- Strip thousands commas from 2023 charges and costs.
- Remove the zero-padding from 2023 birth weights.
- Check that the age-label spelling matches across years.
- The APR-DRG version differs: v40 in 2023, v41 in 2024.
- Pre-2018 files use the older CCS codes, which aren't comparable without a crosswalk.

## Prior work

| Study | Data | What they found |
| :--- | :--- | :--- |
| [Jain et al. 2024, *BMC Health Serv Res*](https://pmc.ncbi.nlm.nih.gov/articles/PMC11288104/) | 2016 SPARCS. Used discharge-coded fields and disposition; dropped charges and costs; modeled newborns separately | LOS-bin classification AUC ≈ 0.78 for CatBoost and random forest vs. ≈ 0.55 for multinomial logistic regression. Regression R² ≈ 0.42 (non-newborns) and 0.82 (newborns). No fairness analysis |
| [Reddad et al. 2026, IJACSA](https://thesai.org/Downloads/Volume17No3/Paper_86-Machine_Learning_Application_in_Healthcare.pdf) | 2023 SPARCS | R² = 0.87, but the top features were Total Costs and Charges, i.e. leakage. Without them, linear regression got R² ≈ 0.53 |
| [annnova2007 on GitHub](https://github.com/annnova2007/hospital-length-of-stay-prediction) | about 2.2M rows | Dropped post-admission features; CatBoost R² ≈ 0.21. Discharge-coded fields carry much of the signal |
| [Szydziak & Sheather 2018](https://ww2.amstat.org/meetings/proceedings/2018/data/assets/pdf/866999.pdf) | 2014 SPARCS, septicemia charges | Facility was the dominant predictor. Removed rows with costs > charges and implausibly low charges |

### Fairness

- No published fairness or calibration audit of SPARCS length-of-stay models was found.
- Closest work: [Wang et al. 2024, *JMIR*](https://www.jmir.org/2024/1/e47125), a readmission-model fairness study on Maryland and Florida discharge data. Recalibration reduced disparities.
- Background theory:
  - Obermeyer et al. 2019 (*Science*): a model can be well calibrated by group and still biased because of the label it predicts.
  - Kleinberg et al. / Chouldechova: calibration within groups and equal error rates can't both hold when base rates differ, which is likely here between Medicaid and commercial patients.

## Pitfalls to design around

1. **Leakage:** drop charges, costs and disposition from LOS models.
2. **Discharge-coded fields:** fit with and without them, and report the gap.
3. **High-cardinality codes** (diagnosis, DRG): don't label-encode them as integers. Use one-hot with a minimum count, target encoding fit inside the CV folds, or roll up to MDC or the CCSR body-system prefix.
4. **Newborns:** exclude them or model them separately.
5. **Specialty facilities:** exclude them, flag them, or run a sensitivity check.
6. **Facility effects are strong.** Optionally, run grouped CV by facility as a robustness check.

## Discharge-coded features as proxies for admission-time knowledge

*Researched 2026-10-15.* The question: clinicians form a working picture at admission (likely diagnosis, how sick the patient is, planned procedures). Can the discharge-coded fields stand in for that internal knowledge, for example for a hospital's own bed planning?

### Verdict

**Feasible and logical for the diagnosis and procedure fields, but not for severity.** The model is defensible if it's framed as a *best case with discharge information* (an upper bound), not as a deployable admission-day tool. The fix is to report a feature ladder, which turns the leakage question into a result.

### By feature

| Field | Known near admission? | Evidence |
| :--- | :--- | :--- |
| Admission type, ED indicator, payer, demographics, facility | Yes | Recorded at registration and triage |
| **CCSR diagnosis, coarse level** (body-system prefix, e.g. `CIR`) or **APR MDC** | Mostly | The admitting diagnosis is coded on the claim at admission. The ED admission diagnosis disagrees with the discharge diagnosis in about 12–28% of cases at the exact level (published range 6.5–75.6% depending on definition). Agreement at the body-system level is probably much higher, but no study measures it directly |
| **CCSR diagnosis, full category** | Partly | As above, with more disagreement at finer detail. **Disagreement is correlated with the outcome:** discrepant cases stayed longer (10.3 vs. 6.9 days, Hautz 2019; 4.2 vs. 3.4 days, Johnson 2009) and had higher mortality |
| **CCSR procedure** | Yes for elective admissions; no for emergency | Elective surgery is booked and prior-authorized days ahead with a procedure code. Emergency-admission procedures are often a *response* to how the stay goes (e.g. tracheostomy, return to the operating room) |
| **APR-DRG** | Partly; the leakiest grouping | Hospitals do assign a provisional "working DRG" within 24–48 h for case management, with a target LOS. Programs aim for about 80% working-to-final agreement. But the final DRG can encode events tied to stay length (e.g. groups defined by ventilator support over 96 hours) |
| **APR Severity of Illness / Risk of Mortality** | **No** | Built from *all* secondary diagnoses at discharge, including complications acquired in hospital. About 7–8% of secondary diagnoses aren't present on admission, and they cluster in long, complicated stays. **Reverse causation:** a longer stay creates more chances for complications, which raise severity. In mortality prediction, admission-only (POA) severity lost 0.01–0.13 in c-statistic (3M Florida data: 0.965 → 0.953; Romano & Chan, heart-attack patients: 0.87 → 0.74). The gap for LOS is probably larger but unpublished |

### What hospitals actually do

- Real capacity tools predict **at admission or within 24 h and update daily** from live EHR data. These include Epic's expected-discharge-date models, Qventus, LeanTaaS, and the Hartford HealthCare/MIT tool. Most also predict the discharge destination and barriers to discharge.
- Case managers compare each stay against the working DRG's target length of stay. That supports the user's intuition that hospitals have something like a DRG early on. That provisional DRG is predictive for *groups* of patients but not reliable for individuals.
- **3M offers an "Admission APR-DRG"** that keeps only diagnoses present on admission, giving admission-time severity and mortality risk. The vendor itself treats admission-time and discharge-time severity as different quantities. It needs licensed software and the restricted SPARCS file, which very likely contains present-on-admission flags and an admitting-diagnosis field; that is unconfirmed. The public file has neither.

### Benchmarks for an honest admission-time model

| Study | Outcome | Admission | Later |
| :--- | :--- | :--- | :--- |
| [Rajkomar et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550175/), full EHR | LOS ≥ 7 days (same target) | AUROC 0.80–0.81 | 0.85–0.86 at 24 h |
| [Zeleke et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426288/), triage data only | LOS > 6 days | AUC ≈ 0.75 (gradient boosting and logistic regression nearly tied) | — |

A SPARCS model with full severity and DRG codes will probably score above these. Part of that excess is hindsight, not skill.

### What no admission-time model can know

- **Waiting for a discharge destination:** in a 16-hospital survey, patients medically ready to leave for ≥ 7 days accounted for 21–35% of general-medicine bed-days. 82% were waiting on resources outside the hospital, and safety-net hospitals had about 3× the odds.
- **In-hospital complications:** infections due to medical care add about 9.6 days; postoperative sepsis about 11 (Zhan & Miller 2003).

Payer and safety-net status will partly proxy for discharge delays. This matters for the fairness analysis: a Medicaid effect may reflect placement barriers rather than clinical need.

### Recommended design

**Prediction time = admission.** The admission-plausible rung (M1) is the primary "deployable" model; higher rungs are reported as upper bounds.

| Rung | Adds | Label |
| :--- | :--- | :--- |
| M0 | Demographics, payer, admission type, ED indicator, facility/region | Admission-time |
| M1 | + CCSR diagnosis body-system prefix (≈ 22 levels) or MDC | Admission-plausible (**primary**) |
| M2 | + full CCSR principal diagnosis | Discharge-coded, close to the working diagnosis |
| M3 | + CCSR procedure, APR-DRG | Discharge-coded, partly reflects treatment |
| M4 | + APR Severity of Illness, Risk of Mortality | Oracle / upper bound |

- **Fit and reporting:** fit every model family on every rung, using the same test set. Report AUROC, AUPRC, Brier score and calibration slope and intercept per rung, with the change from the previous rung and bootstrap CIs.
- **Fairness on both ends:** run the payer and race calibration on **M1 and M4**. Measurement-error theory (Luijken et al. 2019) predicts the mismatch between discharge and admission information shows up mainly in **calibration**, less in AUC.
- **Sensitivity check A, diagnosis noise:**
  - Train M2 on clean data.
  - At test time, replace the CCSR category with probability p ∈ {0.12, 0.25, 0.45}. Draw most replacements from the same body system.
  - Plot AUROC and calibration against p, with M1 as a reference line.
  - If M2 at p ≈ 0.25 falls to M1's level, the coarse model is the honest choice.
  - Real discordance concentrates in sicker patients, so random noise *understates* the damage. State this as a limitation.
- **Sensitivity check B, severity:** read the gain from M3 to M4 as mostly hindsight, and cite Romano & Chan. Optionally, shift severity by ±1 level at random on the test set.

### Framing for the paper

- **Model names:** use "admission-time model" vs. "discharge-information (upper-bound) model", and call the difference the *hindsight gap*.
- **Leakage taxonomy:** cite Kapoor & Narayanan 2023 (leakage type L2, "features not legitimately available at prediction time") and Kaufman et al. 2012 (learn–predict separation).
- **Closest precedent:** Ramadan et al., *JAMA Netw Open* 2025. 40% of published same-admission prediction models used final ICD codes as features, which they call label leakage.
- **Defensible internal uses of the upper-bound model:** population-level capacity planning, case-mix benchmarking, and expected LOS given a known or planned DRG (e.g. elective surgery). Not patient-level admission-day triage.

### Key sources

- **Admission vs. discharge severity:**
  - [Romano & Chan 2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC1975668) (APR-DRG risk of mortality, all diagnoses vs. present-on-admission only)
  - [3M/AHRQ admission APR-DRG pilot](https://hcup-us.ahrq.gov/datainnovations/clinicaldata/3MSummaryResultsReportFinal.pdf)
- **Admission vs. discharge diagnosis:**
  - [Hautz 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6505221) (ED vs. discharge diagnosis discrepancy and LOS)
  - [Johnson 2009](https://shmpublications.onlinelibrary.wiley.com/doi/10.1002/jhm.453)
- **Admission-time prediction models:**
  - [Rajkomar 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550175/)
  - [Zeleke 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426288/)
- **Leakage and measurement error:**
  - [Ramadan 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12743287/) (diagnostic codes and label leakage)
  - [Kapoor & Narayanan 2023](https://reproducible.cs.princeton.edu/) (leakage taxonomy)
  - [Luijken 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6619392/) (predictor measurement heterogeneity)
- **Discharge delays and complications:**
  - [Bann et al. 2023, SHM abstract](https://shmabstracts.org/abstract/prolonged-discharge-delays-across-us-hospitals-prevalence-cause-and-impact/) (discharge delays across 16 hospitals)
  - [Zhan & Miller 2003 (PSNet summary)](https://psnet.ahrq.gov/issue/excess-length-stay-charges-and-mortality-attributable-medical-injuries-during-hospitalization)

## Decisions this raises

- [x] ~~Feature ladder M0–M4~~ replaced by a main model + alternate model (see Summary)
- [x] Exclude newborns
- [x] Flag the 10 specialty facilities in EDA; probable exclusion
- [ ] Payer groups for fairness: Medicaid vs. commercial (Private + BCBS); where "Managed Care, Unspecified" goes
- [ ] Race grouping: combine race and ethnicity; how to handle the large "Other Race" group

## Sources

- [2024 Socrata metadata](https://health.data.ny.gov/api/views/sf4k-39ay.json)
- [2023 Socrata metadata](https://health.data.ny.gov/api/views/46xm-urtu.json)
- [2024 Data Dictionary PDF](https://health.data.ny.gov/api/views/sf4k-39ay/files/a768e7f9-db7a-42ce-bedc-e2c88186c687?download=true&filename=NYSDOH_SPARCS_De-Identified_Data_Dictionary_2024.pdf)
- [2024 Overview PDF](https://health.data.ny.gov/api/views/sf4k-39ay/files/4569ea58-f227-4251-8d10-869f070c0444?download=true&filename=NYSDOH_SPARCS_De-Identified_Overview_2024.pdf)
- [Release notes (xlsx)](https://health.data.ny.gov/api/views/sf4k-39ay/files/51682576-a3d1-405b-a988-4d82b3a3a354)
- [OPEN-NY Terms of Use (PDF)](https://data.ny.gov/download/77gx-ii52/application/pdf)
- [SPARCS data access tiers](https://healthweb-back.health.ny.gov/statistics/sparcs/access/)
- [Reilly et al. 2018, SPARCS race data quality (NYC DOHMH)](https://www.nyc.gov/assets/doh/downloads/pdf/survey/epiresearch-sparcs-imputation.pdf)
