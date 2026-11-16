# Outline

Early draft, due Mon 16 Nov 2026. Up to 6 pages of main text in NeurIPS format. Acknowledgements, references and appendices don't count toward the limit.

`[key]` marks a citation key in `template/references.bib`. `[CITE: …]` marks a claim that still needs a source. Numbers come from saved training-set results only; the test set has not been opened. Paper text is in `../references/markdown/` (length-of-stay papers, committed) and `../references/other/markdown/` (everything else, local only).

---

## Decisions needed before writing

1. **Title:** pick one below (§1).
2. **Headline:** calibration (topic 20) or the model comparison (topic 23)? This sets the title and the order of the abstract.
3. **Error bars in the CV tables:** SD or SE? The code and Table 1 use the **SD across folds**. The baseline checkpoint (`baseline.txt`) reported the SE (SD/√5). Pick one and say which in every caption. Recommendation: SD.
4. **Split wording:** the proposal says 80/20, but the code does **72/8/20** (train/recalibration/test). Recommendation: "80/20, with a 10% slice of the training portion held back for recalibration."
5. **XGBoost edge check:** the best setting sits at the low edge of both `min_child_weight` (1) and `subsample` (0.8). Accept it and say so, or add one more grid step before the 5-fold refit?
6. **Test-set bootstrap:** planned but not specified. Proposed spec in §5.4.
7. **Final model list:** baseline, lasso, tree and XGBoost are certain. Random forest and MLP are in only if time allows. Stacking: cut?
8. **Schedule:** `project_requirements.md` and the analysis plan give 6 Nov for the early draft, while `topic_guide.md` gives 16 Nov. This outline follows the topic guide (16 Nov draft, 30 Nov analysis complete, 4 Dec paper and video draft).

---

## 1. Title and abstract

### Title
The current title, from `proposal_v2.txt` and `baseline.txt`, is "Predicting long unplanned hospital stays in New York, fairly across groups".

| # | Option | Tradeoff |
| :-- | :-- | :-- |
| 1 | Predicting Long Unplanned Hospital Stays in New York, Fairly Across Groups | Continuity with the proposal, but "fairly" claims a result we don't have yet, and it doesn't mention calibration |
| 1b | Predicting Long Unplanned Hospital Stays in New York: Accuracy and Calibration Across Payer and Race | Option 1 without the overclaim |
| 2 | Do the Probabilities Mean What They Say? Calibrated Prediction of Prolonged Hospital Stays Across Payer and Race | Calibration first, echoing topic 20. Long, and a little informal |
| 3 | Predicting Stays Over a Week at Admission: Logistic Regression vs. Tree Ensembles on New York Inpatient Data | States the comparison (topic 23), but drops calibration and fairness |
| 4 | **Calibrated Admission-Time Prediction of Prolonged Length of Stay in New York Hospitals** | Uses the field's terms. Fairness appears only in the abstract |
| 5 | Comparing Course Models for Predicting Hospital Stays Over Seven Days: Accuracy, Calibration and Group Reliability | Covers all three, but generic |

Recommendation: **4**, or **1b** to stay close to the proposal.

### Abstract (one paragraph; bullets in order)
- **Question.** For unplanned (emergency, urgent or trauma) admissions, how well can course methods predict, from information available near admission, a stay of more than 7 days? And are the predicted probabilities equally reliable across payer (Medicaid vs. commercial) and race?
- **Data.** NYSDOH SPARCS de-identified inpatient discharges, 2023–24 [sparcs2023, sparcs2024]: 3,192,140 unplanned stays (23.1% long), a 500,000-stay random sample, and 17 categorical admission-time features. Discharge-time fields are excluded as leakage.
- **Methods.** Unpenalized logistic regression (the baseline) vs. lasso, a pruned decision tree and gradient-boosted trees (XGBoost); [random forest, MLP if run]. Each is tuned by stratified 5-fold CV on 360k training stays, then recalibrated (Platt, isotonic) on a separate 40k set and scored once on a 100k test set.
- **Metrics.** AUC, Brier score and calibration curves, overall and by payer and race.
- **Preliminary result (CV).** Baseline AUC 0.773 ± 0.002 and Brier 0.145. The lasso is tied. The tree is lower (0.760). XGBoost tuning points to about 0.79 [refit pending]. All are in the range published for admission-time models (0.75–0.81) [zeleke2023, rajkomar2018].
- **Final sentence (placeholder).** "[Best model] reaches test AUC X [CI]. After [Platt/isotonic] … Calibration by payer/race [gap / no gap]."

---

## 2. Introduction

**P1. Problem and question**
- Length of stay (LOS) is central to planning hospital resources and controlling costs [stone2022 abstract; zeleke2023 §1.1; jain2024 abstract].
- "The LoS is usually unknown at the time a patient is admitted" [jain2024 p2]. This matters most for unplanned admissions, which arrive without notice, unlike scheduled elective stays.
- State the question once, in full (as in the abstract).

**P2. Why it matters (applications, each with a citation)**
- **Bed management and capacity planning:** "Hospital length of stay of patients is a crucial factor for the effective planning and management of hospital resources" [stone2022 p1]. Capacity planning [jain2024 p1, p3]. Bed capacity during COVID-19 [jain2024 p2].
- **Discharge planning:** "Proactively managing discharge from as early in the admission as possible…" [stone2022 §1 p2]. Early identification of likely long stays lets case managers arrange post-acute care (home health, skilled nursing, rehab) sooner.
- **Patient outcomes and cost:** long stays raise the risk of hospital-acquired complications [zeleke2023 §1.1; stone2022 §1]. LOS also measures resource use [rajkomar2018].
- **Optional:** staffing or ED boarding. [CITE: none found yet; drop unless a source turns up.]
- **Paraphrase rather than quote**, and cite each claim. The verbatim quotes are only for checking against the source.

**P3. Why admission-time features, and why leakage matters**
- A planning model may only use what is known when the decision is made.
- Prior SPARCS LOS work uses discharge-time fields (APR-DRG, severity, procedure, disposition) [jain2024]. Using finalized codes as features inflates performance [ramadan2025], and is "illegitimate-feature" leakage, L2 in the taxonomy of [kapoor2023].
- We exclude them. Our principal diagnosis, body-system group (MDC) and medical/surgical flag are coded at discharge, as stand-ins for the working diagnosis. Discharge and admission diagnoses differ in about 12% of ED admissions, and those patients stay longer [hautz2019]. So our AUC is probably somewhat optimistic; this goes under Limitations.
- Benchmarks:
  - LOS ≥ 6 days from triage data: AUC 0.754 [zeleke2023].
  - LOS ≥ 7 days from the full EHR, at admission: 0.80–0.81 [rajkomar2018].

**P4. What we compare**
- A course baseline (logistic regression) against penalized logistic, a decision tree and gradient boosting [chen2016]; random forest and MLP if run. All use the same features, the same CV folds and comparable tuning budgets, with one test set used once.
- Zeleke found logistic regression within 0.002 AUC of boosting [zeleke2023], a precedent for "flexible models may not help".

**P5. Why calibration, and why by payer and race**
- AUC measures only ranking. Planning uses the probabilities themselves (the expected number of long stays), so they need to be calibrated.
- Boosted trees distort probabilities, while logistic regression is usually already calibrated. Platt scaling and isotonic regression correct this [niculescu2005; platt1999; zadrozny2002].
- If risks are systematically too low for one group, that group's needs are under-planned. Wang et al. audited calibration by race and income on statewide all-payer discharge data [wang2024].
- Calibration within groups and equal error rates can't both hold when base rates differ [kleinberg2017; chouldechova2017], so the fairness check is calibration within groups.
- [obermeyer2019 is in the bib but hasn't been read. Read it before citing, or drop it.]

**P6. Contributions (bullets in the paper)**
- A leakage-aware admission-time cohort from public SPARCS 2023–24 data (3.19M unplanned stays), with the exclusions documented.
- An equal-protocol comparison of course models on AUC, Brier and calibration, with error bars.
- Platt vs. isotonic recalibration before and after, for every model.
- Test-set AUC and calibration by payer and race, with null gaps reported honestly.
- Unlike most LOS work, which is single-site [stone2022 §6.1], the data cover a whole state.

---

## 3. Data

**Source** [sparcs2023, sparcs2024; data dictionary: sparcsdictionary2024]
- NYSDOH SPARCS *Hospital Inpatient Discharges (De-Identified)*, 2023 (https://health.data.ny.gov/d/46xm-urtu) and 2024 (https://health.data.ny.gov/d/sf4k-39ay), on Health Data NY under the OPEN-NY terms [openny].
- One row is one stay. There is no patient ID, so repeat stays can't be linked; say "stays", not "patients".
- De-identification: 5 age bands, 3-digit ZIP, no dates beyond the year, 4 race values, and enhanced redaction of some records.
- Not using 2021–22 (COVID-era noise); 2025 is not yet published.

**Size and shape**
- 2023: 2,125,754 stays. 2024: 2,196,737. Pooled: **4,322,491 × 33 columns**. 20.0% long before filtering.

**Cohort flow** (CONSORT-style figure or a short table; pooled counts)

| Step | Removed | Remaining |
| :-- | --: | --: |
| Pooled 2023 + 2024 | — | 4,322,491 |
| Keep Emergency, Urgent and Trauma admissions (drop Elective, Newborn, Not Available) | 1,111,319 | 3,211,172 |
| Drop newborns admitted under other types (MDC 15) | 10,250 | 3,200,922 |
| Drop publisher-redacted records (blank facility; mostly pregnancy-related, MDC 14) | 8,606 | 3,192,316 |
| Drop the publisher's 10 specialty facilities (rehab, life support) | 176 | **3,192,140** |

- Final cohort: 1,562,648 stays in 2023 and 1,629,492 in 2024; **23.09% long**.
- One clause each on why: elective stays are largely planned; newborns follow a different mechanism and would distort the payer comparison; a deployed model always knows its own facility; and the specialty list is the publisher's.
- **Kept on purpose:**
  - **Patients who died:** dropping them would condition on the future.
  - **184 exact-duplicate rows:** without IDs, errors can't be told apart from look-alike stays.
  - **Facilities with many long stays that aren't on the publisher's list:** dropping them would select on the outcome.
- TODO (optional): per-year counts at every step. Rerun the `common.py` filters and print `discharge_year` counts.

**Outcome**
- y = 1 if SPARCS LOS ≥ 8. SPARCS counts LOS as (discharge − admission) + 1 [sparcsdictionary2024], so this means discharge 7 or more days after admission, the same target as [rajkomar2018] (23.9% long there, 23.1% here).

**Sample and split**
- A simple random sample of 500,000 stays (seed 42). It keeps the slower models tractable while leaving the subgroups large, and matches the cohort mix within about 0.15 percentage points.
- Two stratified splits on y (seed 42):

  | Set | Stays | Share | Used for |
  | :-- | --: | --: | :-- |
  | Train | 360,000 | 72% | Fitting and stratified 5-fold CV |
  | Recalibration | 40,000 | 8% | Fitting Platt and isotonic only |
  | Test | 100,000 | 20% | Scored once, the same rows for every model |

- Why random rather than temporal: rows are near-independent stays with no within-year dates. Caveat: one patient's repeat stays can land in both train and test.

**Features: 17, all categorical**
- Age band, gender, race, ethnicity, 3-digit patient ZIP, primary/secondary/tertiary payer, admission type, ED indicator (the bill has an ED revenue code), facility, hospital county, health service area, MDC*, medical/surgical*, CCSR principal diagnosis*, discharge year. (*coded at discharge; stand-ins for the working diagnosis)
- **Encoding:** one-hot, with levels seen in fewer than 100 training stays pooled into one "Other" column per feature, and unseen levels mapped to Other. That gives 632 columns; the baseline drops one level per feature (615). The encoder is fit inside each CV fold. Blank ZIP or payer becomes an "Unknown" level.
- Full per-feature table (levels, levels pooled, whether known at admission) → appendix.

**Comparison with other papers** (Data section, or a short appendix table; from refs digest §H)
- **[zeleke2023]:** triage-time data only: age, sex, mode of arrival, triage acuity, chief complaint and admitting specialty. No diagnosis codes and no payer.
- **[rajkomar2018]:** the entire EHR up to admission, including labs, vitals, medications, notes and prior visits. Their logistic baseline used age, gender, condition categories, admission source, service and labs.
- **[jain2024]:** the same SPARCS file family (2016), all admission types. It used APR-DRG, severity, risk of mortality, procedure and disposition (all discharge-time) and dropped only charges and costs.
- **Ours:** stricter than prior SPARCS work, but poorer in information than EHR or triage data: no triage acuity, chief complaint, labs or prior admissions.
- Age, sex and principal diagnosis are the most common LOS predictors in the literature [stone2022 §4.4].

**Excluded variables (leakage logic)**
- Rule: a feature is allowed only if it is (approximately) known near admission [kapoor2023 L2].

| Excluded | Why |
| :-- | :-- |
| Total charges, total costs | Grow with each day, so they are effectively the target in dollars (even [jain2024] drops them) |
| Patient disposition | Known only at discharge, and tied to long stays (e.g. a nursing-home placement) |
| APR severity of illness, risk of mortality | Computed from all discharge diagnoses, including in-hospital complications |
| APR-DRG | Built from diagnosis plus procedures. Some groups are defined by prolonged ventilation or tracheostomy |
| Principal procedure | For unplanned stays, often a response to how the stay went |
| Birth weight | Not leakage; almost always blank once newborns are removed |
| Facility name, operating certificate, description columns | Duplicate the codes already included |

---

## 4. Method

**How Method differs from Experimental setup.**
- **Method** says what differs by model: what each model is, its search space and the setting chosen. That's where the chosen hyperparameters go.
- **Experimental setup** says what every model shares: folds, metrics, threshold, recalibration protocol, error bars, test protocol and subgroups.
- The full CV comparison is a result, so it goes in §6.

**Shared input.**
- Same one-hot features for every model.
- No scaling, since every column is 0/1.
- No class reweighting, because it would distort the risks.

**Selection rule (all models).**
- Highest mean 5-fold CV AUC, with no one-standard-error rule.
- The budgets are comparable rather than equal: tens of settings × 5 folds per model. Say so.

**Models and status**

| Model | Course module | Status |
| :-- | :-- | :-- |
| Logistic regression, unpenalized (**baseline**) | Classification | Done: CV, OOF, final fit, recalibrators |
| Penalized logistic (ridge, elastic net, lasso as one family; **lasso** reported) | Regularization | Done: CV, OOF, final fit, recalibrators |
| Decision tree (cost-complexity pruned) | Trees | Done: CV, OOF, final fit, recalibrators |
| Gradient-boosted trees (XGBoost) [chen2016] | Trees and ensembles | **Tuned only.** TODO: 5-fold refit, OOF, final fit, recalibrators |
| Random forest | Trees and ensembles | **Not started** (small grid) |
| MLP | Neural networks | **Not started**; first to cut |
| Stacking | Ensembles | **Optional**; suggest cutting |

**Why logistic regression is the baseline.**
- The topic guide asks for a course baseline taught before the draft.
- It is the standard clinical reference model.
- It was roughly as good as boosting in [zeleke2023].

**Hyperparameters** (paper: one compact table; full grids → appendix)

| Model | Search space | Chosen | CV AUC (mean ± SD) |
| :-- | :-- | :-- | :-- |
| Baseline | none (`C = ∞`, lbfgs) | — | 0.7734 ± 0.0017 |
| Penalized logistic | `l1_ratio` ∈ {0, 0.5, 1} × `C` ∈ 10^[-5, 0] (11 values); saga; 165 fits | lasso, C = 1 | 0.7737 ± 0.0017 |
| Decision tree | `ccp_alpha` ∈ {0, 1e-5, 3e-5, 1e-4, 3e-4} × `min_samples_leaf` ∈ {1, 3, 6, 12, 25, 50, 100, 200, 400, 800}; 250 fits | α = 1e-5, leaf 25 (697 leaves, depth 45 on all of train) | 0.7602 ± 0.0012 |
| XGBoost | `max_depth` ∈ {4, 6, 8} × `min_child_weight` ∈ {1, 10} × `subsample` = `colsample_bytree` ∈ {0.8, 1.0}; trees 1–2,000 chosen from fold-averaged curves; lr 0.1; 60 fits | depth 6, mcw 1, subsample 0.8, 1,198 trees (provisional) | ≈0.790 ± 0.002 (from tuning; optimistic, refit pending) |

- **Penalized logistic:** ridge, elastic net and lasso all peak at C = 1, the edge of the grid, and are tied to the 5th decimal. The curve is flat from C = 0.1 up, and larger C only approaches the baseline, so the grid wasn't extended.
- **Decision tree:** the unpruned tree has AUC ≈ 0.62, and pruning matters most [probst2019 on tuning leaf size].
- **XGBoost:** all 12 settings fall within 0.0023 AUC of each other, so the choice is close to a tie. The chosen setting sits at the low grid edge for `min_child_weight` and `subsample` (decision 5).

**Recalibrators** (the same function for every model)
- **Platt:** a one-variable logistic regression on the log-odds of the raw risk.
- **Isotonic:** regression on the raw risk, clipped at the ends.

---

## 5. Experimental setup

### 5.1 Validation
- Stratified 5-fold CV on the 360k training set, `StratifiedKFold(5, shuffle=True, random_state=42)`, with **the same folds for every model**.
- Everything learned from data is fit inside each fold: the encoder, and XGBoost's tree count.
- One seed (42) throughout, with no repeated seeds; say so.
- Final models are refit on all 360k stays with the chosen settings.

### 5.2 Metrics
- **Core:** AUC (ranking) and Brier score (probability accuracy; a model that predicts the base rate for everyone scores ≈ 0.177), plus calibration curves (decile bins).
- **AUC alone chooses hyperparameters.** In every grid the AUC-best setting also had the lowest, or nearly the lowest, Brier.
- **For awareness only:** precision and recall at **threshold 0.2306**, the training long-stay rate. It flags any stay riskier than average. A cutoff of 0.5 would flag only ~9% of stays.
- **Not used:** accuracy, since always predicting "short" scores ~77%.
- **Fit with the literature:**
  - Rajkomar headlines AUROC with 1,000-resample bootstrap CIs, and puts calibration in the supplement [rajkomar2018].
  - Zeleke reports AUC, Brier and calibration curves together and picks its model on AUC and Brier [zeleke2023].
  - So AUC is the standard headline, and we add Brier and calibration as Zeleke does. Don't claim "others ranked by AUC alone".

### 5.3 Recalibration protocol
- Platt and isotonic are fit on the 40k recalibration set, using each final model's predictions.
- They are judged **only on the test set**, comparing raw, Platt and isotonic on the same rows.
- Platt can't change AUC. Isotonic can shift it slightly through ties, so report the change.
- With 40k points, isotonic shouldn't be data-starved: with 1,000 or more calibration points, isotonic matches or beats Platt [niculescu2005].

### 5.4 Error bars
- **CV (done):** mean ± SD across the 5 folds (decision 3). For model vs. baseline, also report the paired fold differences: the lasso beats the baseline on all 5 folds, but by only 0.0002–0.0003.
- **Test (planned, in the proposal and the 11-06 note):** bootstrap resamples of the test set for AUC and each calibration bin, overall and by group. **Not yet specified.** Proposal, to record in a note before the test set is opened:
  - **Paired bootstrap:** resample the 100k test rows B = 1,000 times, as [rajkomar2018] did, and score every model on the same resampled rows. Report 95% percentile CIs.
  - **Metrics:** CIs for AUC and Brier per model, plus CIs for each model's difference from the baseline. A difference whose CI includes 0 is a tie, which is how the topic guide asks for comparisons to be judged.
  - **Subgroups:** the same, resampling within each group. Report gaps such as Medicaid − commercial from the same resamples.

### 5.5 Test-set protocol
- A single comparison notebook, run once, is the only code that loads the test set.
- It scores raw, Platt and isotonic risks for every model on the same 100k rows.
- Test precision and recall use the 0.2306 cutoff on recalibrated risks. Open question: Platt or isotonic?

### 5.6 Fairness evaluation (reporting only; training is unchanged)
- **Payer:** Medicaid; commercial (Private + Blue Cross/Blue Shield); Medicare for context.
- **Race:** White; Black/African American; Other Race.
- **Table:** AUC and Brier with counts and CIs for all groups. Multi-racial (~1,300 test rows) appears in the table only.
- **Calibration curves:** for the main 3 payer and 3 race groups.
- **Caveats:**
  - SPARCS race corresponds poorly to Medicaid and vital-statistics records [reilly2018].
  - "Other Race" lumps several groups together.
  - A payer gap may reflect barriers to discharge, so report it as an association.
- **Open:** where to put "Managed Care, Unspecified".

### 5.7 Software
- Python 3.13, scikit-learn [pedregosa2011], XGBoost [chen2016], uv; 8 cores.
- Fit times: penalized ~300 CPU-min, tree ~117 CPU-min, XGBoost ~93 min.

---

## 6. Preliminary results

**Table 1** (`figures/tab_cv_metrics.tex`, generated by `make_figures.py`)

| Model | AUC ↑ | Brier ↓ | Precision | Recall |
| :-- | :-: | :-: | :-: | :-: |
| Logistic regression (baseline) | 0.7734 ± 0.0017 | 0.1447 ± 0.0003 | 0.4065 ± 0.0019 | 0.7018 ± 0.0030 |
| Lasso logistic regression | 0.7737 ± 0.0017 | 0.1446 ± 0.0003 | 0.4065 ± 0.0016 | 0.7017 ± 0.0026 |
| Decision tree | 0.7602 ± 0.0012 | 0.1465 ± 0.0003 | 0.4060 ± 0.0029 | 0.6581 ± 0.0063 |

*Caption draft:* Five-fold cross-validated performance on the training set (n = 360,000 unplanned admissions, 23.1% longer than 7 days); mean ± SD across folds, raw (unrecalibrated) risks. Precision and recall flag risk ≥ 0.2306, the training long-stay rate. A base-rate predictor has Brier ≈ 0.177. Lasso C = 1; tree α = 10⁻⁵, ≥ 25 stays per leaf. The test set is unused.

**Figure 1** (`figures/roc_oof.pdf`): ROC curves from pooled out-of-fold risks, one per finished model; the legend gives mean ± SD AUC across folds.
*Caption draft:* ROC curves from 5-fold out-of-fold predictions on the training set. Baseline and lasso overlap almost exactly. XGBoost [and RF, MLP] will be added when their CV is complete.

**Observations** (CV only; worded cautiously)
- The baseline's AUC of 0.773 falls within the published admission-time range (0.754 [zeleke2023] to 0.80–0.81 [rajkomar2018]). It may be optimistic because the diagnosis fields are coded at discharge.
- **Lasso vs. baseline is a tie:** +0.0003 AUC, well under the fold SD. With ~288k fitting stays per fold and ~600 columns, a penalty has little variance to remove.
- **The tree is worse:** about 0.013 lower AUC, slightly worse Brier, and lower recall at the same precision.
- **All models beat the base-rate Brier.** At the threshold, about 41% of flagged stays are long and about 70% of long stays are flagged.
- **Fold SDs** reflect only CV-split noise; the final comparison uses test bootstrap CIs.
- **Optional, caveated:** XGBoost tuning suggests AUC ≈ 0.790. That figure comes from the same folds used to choose its setting, so it isn't in Table 1 until the refit.
- **Calibration plots are planned:** out-of-fold calibration of raw risks for all models, then test-set panels for raw vs. Platt vs. isotonic. They wait until every model is fitted and recalibrated.

---

## 7. Remaining sections (outline only)

- **Test results:**
  - AUC and Brier for every model, raw / Platt / isotonic, with bootstrap CIs and paired differences from the baseline.
  - Test ROC.
  - Precision and recall for context.
  - How the recommended model is chosen: decide before the test set is opened.
- **Calibration:**
  - Out-of-fold calibration figure (train).
  - Test panels per model (raw / Platt / isotonic); the full figure goes in the appendix if space is short.
  - Which models needed recalibration most. Boosting is expected to gain the most [niculescu2005]; report whatever we find.
- **Fairness by payer and race:**
  - Subgroup table with counts and CIs.
  - Calibration curves by group for the recommended model.
  - The impossibility framing [kleinberg2017].
  - Caveats (§5.6).
- **Discussion and limitations:**
  - What the comparison shows, including a null "LR is as good" result.
  - Diagnosis fields coded at discharge.
  - No patient ID.
  - Deaths shorten stays.
  - One random sample and one seed.
  - New York only, 2023–24.
- **Conclusion:** 2–3 sentences.
- **Acknowledgements:**
  - NYSDOH SPARCS data credit (no State logos).
  - **AI-use statement (required):** which assistants were used, and for what (code, literature search and summaries, LaTeX, editing). State that every reference and result was checked.
- **References:** from `template/references.bib`.
- **Appendix:**
  - Cohort flow.
  - Full feature table.
  - Excluded columns.
  - Tuning grids and curves.
  - Full calibration panels and subgroup tables.
  - Top of the decision tree and feature importance.

---

## 8. Experiments before Analysis Complete (Mon 30 Nov)

Each step starts with a dated note, per CLAUDE.md.

- [ ] **Tue 17 – Wed 18 Nov — XGBoost:**
  - Settle the edge check (decision 5).
  - Run the 5-fold refit, then the final model, recalibration predictions, recalibrators and importance.
  - Rerun `make_figures.py`.
- [ ] **Thu 19 – Sun 22 Nov — random forest:** small grid, the same outputs as the other models.
- [ ] **Mon 23 Nov — MLP:** go or cut. Default: cut unless the random forest is done.
- [ ] **By Tue 24 Nov — comparison note and notebook, up to the test set:**
  - Write down the bootstrap spec, the recommended-model rule and the placement of the subgroups.
  - Make the out-of-fold calibration figure.
- [ ] **Wed 25 – Thu 26 Nov — open the test set once:** run every metric, CI, calibration panel and subgroup result.
- [ ] **Fri 27 – Mon 30 Nov — buffer:** freeze the results. No new models after the test set is opened.

**Out of scope unless time is left over:**
- Stacking.
- Optional extras: a hindsight feature set, a 2023→2024 temporal check, the psychiatric per-diem appendix, an occlusion study, interactions, permutation importance.
- A baseline rerun without the three diagnosis features. It was promised in the checkpoint's next steps and is the most useful extra for the discharge-coding caveat. Keep or drop?

---

## Reference notes

- **In the bib, PDF and Markdown read:**
  - zeleke2023, rajkomar2018, jain2024, stone2022
  - hautz2019, kapoor2023, reilly2018
  - niculescu2005, wang2024, kleinberg2017, chouldechova2017
  - probst2019, chen2016
  - SPARCS dictionary and overview
- **In the bib, read from the PMC text only:** ramadan2025.
- **In the bib, not read:** obermeyer2019, platt1999, zadrozny2002, pedregosa2011. Read before citing; standard method citations are fine.
- **Correction to the 2026-10-15 dataset note:** it says "Recalibration reduced disparities" for [wang2024]. The paper's claim is narrower: retrained models showed the lowest bias in Maryland, but not consistently in Florida.
- **Not added:** Levin et al. 2021 (download blocked); Reddad 2026, Romano & Chan, Luijken, Johnson 2009 and others from the notes. Ask if any are wanted.
