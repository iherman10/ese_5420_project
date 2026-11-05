**Question**

For unplanned (emergency, urgent or trauma) admissions to New York hospitals, how well can course methods predict a stay longer than 7 days from the information available near admission, and are the predicted probabilities equally reliable for Medicaid and commercially insured patients and across racial groups? 

**Dataset**

New York State's SPARCS Hospital Inpatient Discharges (de-identified), published by the NYS Department of Health: one row per inpatient stay, with 33 columns, for [2023](https://health.data.ny.gov/d/46xm-urtu) (2.13M stays) and [2024](https://health.data.ny.gov/d/sf4k-39ay) (2.20M stays), pooled. Restricting to unplanned admissions leaves about 3.3M stays, about 20% of which last more than 7 days. The data is de-identified by the publisher: there is no patient ID, age is in five bands, ZIP codes are cut to 3 digits, no dates are given beyond the discharge year, and abortion-related records have their location fields blanked. The public tier needs no data use agreement and is released under the [OPEN-NY Terms of Use](https://data.ny.gov/download/77gx-ii52/application/pdf), which permit reuse and redistribution of derived figures without pre-approval. I will credit NYSDOH SPARCS.


**Methods**

Binary classification of whether a stay lasts more than 7 days. The features are roughly what staff know near admission: age group, gender, race, ethnicity, 3-digit ZIP, payer, admission type, ED indicator, facility and region, and the body-system and principal-diagnosis categories. The data is split 80/20 into train and test. Hyperparameters are tuned with K-fold cross-validation (resampling module), and a slice of the training set is held back for recalibration.

- **Models:**
  - Unpenalized logistic regression (classification), as the baseline.
  - Regularized logistic regression: lasso, ridge and elastic net (regularization).
  - A single decision tree, random forest and gradient-boosted trees (trees & ensembles).
  - An MLP (neural networks).
  - A stacking ensemble that combines the models above.
- **Evaluation, on the test set:**
  - AUC, for how well each model ranks long stays above short ones.
  - Calibration curves, for whether predicted probabilities match observed rates. Each model is recalibrated with Platt scaling, and the curves are compared before and after.
- **Fairness:** calibration curves and AUC repeated separately by payer (Medicaid vs. commercial) and by race, to see whether the model over- or under-predicts for particular groups.

**Baseline & Results**

The baseline is unpenalized logistic regression on the same features. For reference, published admission-time models for similar targets (stays of 6+ or 7+ days) reach AUC ≈ 0.75 from triage data alone ([Zeleke et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10426288/)) and 0.80–0.81 from full electronic health records ([Rajkomar et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6550175/)). A result would be a table of test-set AUC for each model alongside calibration curves before and after Platt scaling, overall and by payer and race.

**Group Members**

I will be working alone. 