# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline model
#
# Predicting whether an unplanned New York hospital admission lasts more than
# 7 days. Decisions are in `docs/notes/2026-11-05-data-cohort-split.md`,
# `docs/notes/2026-11-06-cleaning-encoding-baseline.md`,
# `docs/notes/2026-11-07-saving-results.md` and
# `docs/notes/2026-11-12-recalibration-and-comparison.md`.
#
# The steps that every model shares live in `analysis/common.py`. This notebook walks
# through them with a look at the data before each one, then fits the baseline.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

from analysis import common

# %% [markdown]
# ## Step 1: Get the data
#
# Download the 2023 and 2024 SPARCS files unless they're already there, then
# stop with an error if either file isn't the expected size.

# %%
raw = common.load_raw()
for year, df in raw.items():
    print(f"{year}: {df.shape[0]:,} rows, {df.shape[1]} columns")

# %% [markdown]
# ## Step 2: Build the cohort and sample
#
# This step only lines up the columns, creates the target and chooses the rows.
# Before each decision, we look at the data.
#
# ### Pool the two years
#
# First, do the two files use the same column names?


# %%
columns = {year: {common.snake(c) for c in df.columns} for year, df in raw.items()}
print("Only in 2023:", sorted(columns[2023] - columns[2024]))
print("Only in 2024:", sorted(columns[2024] - columns[2023]))
print("Year column present:", "discharge_year" in columns[2023] & columns[2024])

# %% [markdown]
# Two columns have different names, so we rename them. Without this, each would
# split into two half-empty columns. The files already have a year column, so we
# don't need to add one.

# %%
print(common.RENAMES)

cohort = common.pool(raw)
del raw
print(f"{len(cohort):,} stays")

# %% [markdown]
# ### Create the target
#
# Length of stay is stored as text. Which values aren't plain whole numbers?

# %%
is_number = cohort["length_of_stay"].str.fullmatch(r"\d+")
print(cohort.loc[~is_number, "length_of_stay"].value_counts().to_string())

# %% [markdown]
# The only exception is the top value, spelled "120 +" in 2023 and "120+" in
# 2024. We treat both as 120. A long stay is more than 7 days, so $y = 1$ if
# length of stay is at least 8.

# %%
cohort = common.add_target(cohort)

print(f"Length of stay ranges from {cohort['los'].min()} to {cohort['los'].max()}")
print(f"Long stays: {cohort['y'].mean():.1%} of all stays")


# %%
def long_stay_table(df, by):
    """Number of stays and long-stay rate for each level of a column."""
    table = df.groupby(by, dropna=False)["y"].agg(stays="size", long_stay_rate="mean")
    return table.sort_values("stays", ascending=False).round(3)


# %% [markdown]
# ### Admission type
#
# Which kinds of admission are in the data?

# %%
print(long_stay_table(cohort, "type_of_admission").to_string())

# %% [markdown]
# We're predicting long stays for unplanned admissions, so we keep emergency,
# urgent and trauma. Elective stays are largely planned, newborn stays work
# differently, and "Not Available" doesn't tell us the type.

# %%
print(common.KEEP_ADMISSION_TYPES)

cohort = common.keep_unplanned(cohort)
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Newborns
#
# The admission type filter removes stays labeled "Newborn". Are any newborns
# left under other admission types? APR MDC code 15 is the newborn diagnostic
# group.

# %%
newborns = cohort[cohort["apr_mdc_code"] == common.NEWBORN_MDC]
print(f"{len(newborns):,} stays, {newborns['y'].mean():.1%} long")
print()
print(long_stay_table(newborns, "type_of_admission").to_string())
print()
print("Top diagnoses:")
diagnoses = newborns["ccsr_diagnosis_description"].str.capitalize()
print(diagnoses.value_counts().head(5).to_string())

# %% [markdown]
# Some are, though they're a small group. Our plan leaves newborns out because
# their stays work differently, so we drop this group too.

# %%
cohort = common.drop_newborns(cohort)
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Redacted records
#
# There's no column flagging redacted records, but `facility_name` says
# "Redacted" on some rows. What's blank on those rows?

# %%
LOCATION_COLUMNS = [
    "permanent_facility_id",
    "hospital_county",
    "health_service_area",
    "zip_code",
]

is_redacted = cohort["facility_name"].str.contains("Redacted", na=False)
print(
    f"{is_redacted.sum():,} redacted stays, {cohort.loc[is_redacted, 'y'].mean():.1%} long"
)
print(f"Blank facility ID: {cohort['permanent_facility_id'].isna().sum():,} stays")
print()
print("Share of rows blank:")
blank = cohort[LOCATION_COLUMNS].isna().groupby(is_redacted).mean().round(3)
print(blank.rename_axis("redacted").to_string())
print()
print("Top diagnostic groups in redacted stays:")
print(cohort.loc[is_redacted, "apr_mdc_code"].value_counts().head(3).to_string())

# %% [markdown]
# The publisher blanks the location on these rows for privacy, and its
# documentation describes this as extra de-identification for abortion records.
# Every redacted row has no facility, county, service area or ZIP. A real hospital
# always knows which facility it is, so these rows don't fit how the model would
# be used, and we drop them.
#
# This removes mostly pregnancy-related stays (MDC 14), so the paper needs to say
# so.

# %%
cohort = common.drop_redacted(cohort)
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Specialty facilities
#
# We want a model for typical hospitals. The publisher lists 10 facilities whose
# discharges are mostly rehab or life support. How many of their stays are left?

# %%
is_specialty = cohort["permanent_facility_id"].isin(common.SPECIALTY_FACILITIES)
print(f"{is_specialty.sum():,} stays, {cohort.loc[is_specialty, 'y'].mean():.1%} long")
print()
print(long_stay_table(cohort[is_specialty], "permanent_facility_id").to_string())

# %% [markdown]
# Very few, and only five of the 10 facilities have any stays left. We drop them
# so that the rule is simple: the publisher's list defines a specialty facility.

# %%
cohort = common.drop_specialty(cohort)
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Kept on purpose
#
# These groups might look like candidates for removal, but we keep all of them.

# %%
died = cohort["patient_disposition"] == "Expired"
print(
    f"Patients who died: {died.sum():,} stays, {cohort.loc[died, 'y'].mean():.1%} long"
)
print(f"Exact duplicate rows: {cohort.duplicated().sum():,} extra rows")

is_blank_zip = cohort["zip_code"].isna()
print(f"Blank ZIP code: {is_blank_zip.sum():,} stays")
print()

is_behavioral = cohort["apr_mdc_code"].isin(["19", "20"])
facilities = cohort.assign(behavioral=is_behavioral).groupby("permanent_facility_id")
facilities = facilities.agg(
    stays=("y", "size"),
    long_stay_rate=("y", "mean"),
    behavioral_share=("behavioral", "mean"),
    name=("facility_name", "first"),
)
print("Highest long-stay rates among facilities with at least 200 stays:")
highest = facilities[facilities["stays"] >= 200]
highest = highest.sort_values("long_stay_rate", ascending=False).head(6).round(3)
print(highest.to_string())

# %% [markdown]
# - **Patients who died:** death isn't known at admission, so removing them would
#   use future information. Their stays are cut short, which the paper should say.
# - **Exact duplicate rows:** with no patient ID, we can't tell whether these are
#   errors or separate stays that look alike.
# - **Blank ZIP codes:** ZIP is the patient's home area, not the hospital's
#   location, so we handle these blanks during cleaning.
# - **Facilities with many long stays:** these aren't on the publisher's list.
#   `behavioral_share` is the share of stays for mental health or substance use
#   (MDC 19 and 20). Most of these sites are nearly all behavioral, but not all of
#   them. Choosing sites by long-stay rate would select on the outcome we're
#   predicting, so they stay in and the paper notes it.

# %% [markdown]
# ### Check the cohort
#
# Stop with an error if the cohort isn't the expected size.

# %%
common.check_cohort(cohort)

print(f"Cohort: {len(cohort):,} stays, {cohort['y'].mean():.2%} long")
print(cohort["discharge_year"].value_counts().to_string())

# %% [markdown]
# ### Draw the sample
#
# To keep the slower models practical to train, all models use one simple random
# sample of 500,000 stays from the roughly 3.2 million in the cohort.

# %%
sample = common.draw_sample(cohort)
print(f"Sample: {len(sample):,} stays, {sample['y'].mean():.2%} long")

# %% [markdown]
# Does the sample look like the cohort? The two columns below are the percentage
# of stays at each level.

# %%
for column in ["discharge_year", "type_of_admission", "payment_typology_1", "race"]:
    shares = pd.DataFrame(
        {
            "cohort": cohort[column].value_counts(normalize=True, dropna=False),
            "sample": sample[column].value_counts(normalize=True, dropna=False),
        }
    )
    print((shares * 100).round(2).to_string())
    print()

# %% [markdown]
# ## Step 3: Split
#
# We split the sample into three sets, keeping the long-stay rate the same in each:
#
# - **Train (360,000):** fitting, and K-fold CV for tuning. CV does the job of a
#   validation set, so there isn't a separate one.
# - **Recalibration (40,000):** fitting Platt scaling and isotonic regression only.
# - **Test (100,000):** final scoring, once, on the same rows for every model. It
#   stays untouched until the end, so the baseline checkpoint score comes from CV
#   on the training set.
#
# We take the test set off first, then take the recalibration set from what's left.

# %%
train, recal, test = common.split(sample)
splits = {"train": train, "recalibration": recal, "test": test}

# %% [markdown]
# Is the long-stay rate the same in each set?

# %%
for name, df in splits.items():
    print(f"{name}: {len(df):,} stays, {df['y'].mean():.2%} long")

# %% [markdown]
# How many test stays will each fairness group have? These are counts only, so
# they don't use any test-set outcomes.

# %%
for column in ["payment_typology_1", "race"]:
    print(test[column].value_counts(dropna=False).to_string())
    print()

# %% [markdown]
# ## Step 4: Clean
#
# Everything we look at in this step comes from the training set. The fixes go in
# one function, `common.clean`, so the recalibration and test sets can get the same
# fixes, unchanged, when we use them.
#
# ### Labels that differ by year
#
# These are the 17 features the model will use. Which of their levels appear in
# only one year?

# %%
FEATURES = common.FEATURES

for column in FEATURES:
    if column == "discharge_year":
        continue
    in_2023 = set(train.loc[train["discharge_year"] == "2023", column].dropna())
    in_2024 = set(train.loc[train["discharge_year"] == "2024", column].dropna())
    if in_2023 != in_2024:
        print(column)
        print("  Only in 2023:", sorted(in_2023 - in_2024))
        print("  Only in 2024:", sorted(in_2024 - in_2023))

# %% [markdown]
# Age, county and service area are the same levels spelled differently, so without
# a fix each one would split in two and partly encode the year. We use one spelling
# for each.
#
# Facility and diagnosis are codes rather than spellings, so we leave them alone.
# Any that are rare get grouped in Step 5.

# %%
for column, fixes in common.LABEL_FIXES.items():
    print(column, fixes)

# %% [markdown]
# ### Codes, not descriptions
#
# Three features have a code column and a description column. Does each code have
# one description?

# %%
CODE_DESCRIPTIONS = {
    "ccsr_diagnosis_code": "ccsr_diagnosis_description",
    "apr_mdc_code": "apr_mdc_description",
    "permanent_facility_id": "facility_name",
}

for code, description in CODE_DESCRIPTIONS.items():
    within_year = train.groupby(["discharge_year", code])[description].nunique()
    across_years = train.groupby(code)[description].unique()
    shared = train.groupby(description)[code].nunique()
    print(code)
    print(f"  Codes: {train[code].nunique()}")
    print(f"  Descriptions: {train[description].nunique()}")
    print(f"  Most descriptions for a code within a year: {within_year.max()}")
    print(
        f"  Most descriptions for a code across years: {across_years.str.len().max()}"
    )
    print(f"  Descriptions used by more than one code: {(shared > 1).sum()}")
    print(f"  Example: {list(across_years[across_years.str.len() > 1].iloc[0])}")

# %% [markdown]
# Within a year, each code has one description. Across years, most codes have two,
# and the examples differ only in how they're written. One-hot encoding the
# descriptions would split each level by year, so we encode the codes and use the
# descriptions only to label results.
#
# A few facility names are used by more than one ID. We keep the IDs as they are,
# because year-to-year facility changes are out of scope.
#
# ### Blanks
#
# Which features have blanks, and do those stays look different?

# %%
blank_counts = train[FEATURES].isna().sum()
BLANK_COLUMNS = blank_counts[blank_counts > 0].index.tolist()
assert BLANK_COLUMNS == common.BLANK_COLUMNS, BLANK_COLUMNS

blanks = pd.DataFrame(
    {
        "blank_stays": blank_counts[BLANK_COLUMNS],
        "blank_share": train[BLANK_COLUMNS].isna().mean(),
        "long_rate_if_blank": [
            train.loc[train[c].isna(), "y"].mean() for c in BLANK_COLUMNS
        ],
        "long_rate_if_filled": [
            train.loc[train[c].notna(), "y"].mean() for c in BLANK_COLUMNS
        ],
    }
)
print(blanks.round(3).to_string())

# %% [markdown]
# Only ZIP and the three payer columns have blanks, and the long-stay rate differs
# between blank and filled rows, so a blank tells us something. We fill blanks with
# "Unknown" so that each becomes its own level. A one-hot "Unknown" column already
# marks these rows, so we don't add a separate missingness indicator.
#
# A blank payer 2 or 3 probably means there's no backup payer, not that a value is
# missing. Payer 1 has so few blanks that its "Unknown" level will go into "Other"
# in Step 5.
#
# ### Apply the fixes
#
# Stop with an error if the cleaned training set doesn't look as expected.


# %%
train = common.clean(train)

levels = train[FEATURES].nunique()
print(levels.to_string())
print(f"Total: {levels.sum()}")

assert train[FEATURES].notna().all().all()
assert levels["age_group"] == 5, levels["age_group"]
assert levels["health_service_area"] == 8, levels["health_service_area"]
assert levels["hospital_county"] == 57, levels["hospital_county"]
assert levels.sum() == 838, levels.sum()

# %% [markdown]
# ## Step 5: Encode
#
# Every feature is a category with no natural order, even when its levels look like
# numbers, so we one-hot encode all 17. Each level gets its own 0/1 column.
#
# ### Rare levels
#
# A level with very few stays would get a noisy coefficient. We group any level with
# fewer than 100 training stays into "Other". No standard sets this number, so it's
# our judgment. We chose it before seeing any model results, and it uses counts, not
# outcomes.
#
# How many levels and stays does this affect in each feature?

# %%
MIN_STAYS = common.MIN_STAYS

rare =pd.DataFrame(index=FEATURES, columns=["levels", "rare_levels", "rare_stays"])
for column in FEATURES:
    counts = train[column].value_counts()
    is_rare = counts < MIN_STAYS
    rare.loc[column] = [len(counts), is_rare.sum(), counts[is_rare].sum()]
print(rare.to_string())
print(f"Total rare levels: {rare['rare_levels'].sum()}")

# %% [markdown]
# Diagnosis and facility are affected most, with many levels but few stays. Where a
# feature has one rare level, such as gender "U", that level is just relabeled.
#
# ### Fit the encoder
#
# This fits the encoder once on all of train, so we can check what it does. In Step
# 6, it's refit inside each CV fold, so the validation fold can't affect the grouping.
#
# `drop="first"` leaves out one level from each feature as the reference. Which level
# does it leave out, and how many columns do we end up with?

# %%
encoder = common.make_encoder(drop="first")
encoded = encoder.fit_transform(train[FEATURES])

names = pd.Series(encoder.get_feature_names_out())
summary = pd.DataFrame(
    {
        "reference": [
            categories[index]
            for categories, index in zip(encoder.categories_, encoder.drop_idx_)
        ],
        "columns": [names.str.startswith(f"{c}_").sum() for c in FEATURES],
    },
    index=FEATURES,
)
print(summary.to_string())
print()
print(f"Encoded training set: {encoded.shape[0]:,} rows, {encoded.shape[1]} columns")
print(f"Share of values that aren't zero: {encoded.mean():.1%}")

assert encoded.shape == (common.TRAIN_SIZE, 615), encoded.shape

# %% [markdown]
# The reference is the alphabetically first level that isn't rare. For example, MDC
# code "00" is rare, so the reference is "01". We won't interpret these coefficients,
# so which level is the reference doesn't matter here.
#
# The result is sparse: each row has at most 17 nonzero values, so it fits easily in
# memory even with 615 columns.
#
# ### New levels
#
# A validation fold or later set can contain a level that the encoder never saw. What
# happens to one?

# %%
new_facility = train[FEATURES].iloc[[0]].assign(permanent_facility_id="NEW")
new_zip = train[FEATURES].iloc[[0]].assign(zip_code="NEW")


def active_columns(row, feature):
    """Names of a feature's columns that are 1 for this row."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = encoder.transform(row).toarray()[0]
    return [n for n, v in zip(names, values) if v and n.startswith(f"{feature}_")]


print("New facility:", active_columns(new_facility, "permanent_facility_id"))
print("New ZIP:", active_columns(new_zip, "zip_code"))

# %% [markdown]
# A new facility goes to "Other", which scikit-learn labels `infrequent_sklearn`. A new
# ZIP gets no column at all, because ZIP has no rare levels and so no "Other". All
# zeros means the model treats that stay like the reference level.
#
# That second case affects only features with no rare levels, which have many stays at
# every level, so a new level there is unlikely. We'll check for it when we use the
# other sets.

# %% [markdown]
# ## Step 6: Fit and evaluate the baseline
#
# The baseline is unpenalized logistic regression, so there are no hyperparameters to
# tune. We use stratified 5-fold CV on the training set. The recalibration set is only
# used in Step 7, and the test set isn't used yet.
#
# ### Save slow results
#
# Slow results only need to run once. `common.load_or_compute` loads saved files if
# they exist, and otherwise computes the results and saves them. To recompute, delete
# the files. Do that whenever something earlier in the notebook changes. Decisions are
# in `docs/notes/2026-11-07-saving-results.md`.

# %%
RESULTS_DIR = common.RESULTS_DIR

# %% [markdown]
# ### Run cross-validation
#
# The pipeline refits the encoder inside each fold, so the validation fold can't affect
# the grouping. `C=np.inf` turns the penalty off. New levels in a validation fold make
# the encoder warn, as expected from Step 5, so we silence that one warning.


# %%
MAX_ITERATIONS = 5_000

baseline = make_pipeline(
    clone(encoder), LogisticRegression(C=np.inf, max_iter=MAX_ITERATIONS)
)


def run_baseline_cv():
    X, y = train[FEATURES], train["y"]
    folds = common.FOLDS
    predictions = pd.DataFrame(index=train.index, columns=["fold", "risk"], dtype=float)
    scores = []

    for fold, (fit_rows, validation_rows) in enumerate(folds.split(X, y), start=1):
        model = clone(baseline)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            model.fit(X.iloc[fit_rows], y.iloc[fit_rows])
            risk = model.predict_proba(X.iloc[validation_rows])[:, 1]
        seconds = time.perf_counter() - start

        iterations = model[-1].n_iter_[0]
        predictions.iloc[validation_rows] = np.column_stack(
            [np.full(len(risk), fold), risk]
        )
        scores.append(
            {
                "fold": fold,
                "auc": roc_auc_score(y.iloc[validation_rows], risk),
                "brier": brier_score_loss(y.iloc[validation_rows], risk),
                "columns": model[-1].coef_.shape[1],
                "iterations": iterations,
                "converged": iterations < MAX_ITERATIONS,
                "seconds": seconds,
            }
        )

    return {"scores": pd.DataFrame(scores), "predictions": predictions}


baseline_cv = common.load_or_compute(
    {
        "scores": RESULTS_DIR / "baseline_logistic" / "cv_scores.csv",
        "predictions": RESULTS_DIR / "baseline_logistic" / "oof_predictions.parquet",
    },
    run_baseline_cv,
)
scores = baseline_cv["scores"]
predictions = baseline_cv["predictions"]

common.check_rows(predictions, train)
assert scores["converged"].all(), "A fold didn't converge"

print(scores.round(4).to_string(index=False))

# %% [markdown]
# ### Results
#
# This table has one row per model. AUC and Brier are the mean and SD across the 5
# validation folds. Precision and recall use all the out-of-fold predictions at the
# default 0.5 threshold. They're for awareness only, and we don't use them to choose
# models.


# %%
def results_row(model, scores, predictions):
    """One row of the results table for a model's CV results."""
    y = train.loc[predictions.index, "y"]
    flagged = predictions["risk"] >= 0.5
    return {
        "model": model,
        "auc": scores["auc"].mean(),
        "auc_sd": scores["auc"].std(),
        "brier": scores["brier"].mean(),
        "brier_sd": scores["brier"].std(),
        "precision": precision_score(y, flagged),
        "recall": recall_score(y, flagged),
    }


results = pd.DataFrame([results_row("Baseline logistic", scores, predictions)])
print(results.round(4).to_string(index=False))

# %% [markdown]
# AUC is within the published range of 0.75 to 0.81, and the folds barely differ. The
# Brier score beats the roughly 0.177 from giving everyone the overall long-stay rate.
# Diagnosis is recorded at discharge, so these scores may be somewhat better than a true
# admission-day model would get.
#
# Recall is low because few patients clear the 0.5 threshold. These two numbers depend
# on the threshold, so low recall here doesn't mean the model ranks patients poorly.
#
# ### Calibration
#
# Each stay has one out-of-fold prediction, from a model that didn't train on it. We sort
# the stays into 10 equal-sized groups by predicted risk, then compare each group's
# average predicted risk with the share who actually stayed long.

# %%
observed, predicted = calibration_curve(
    train["y"], predictions["risk"], n_bins=10, strategy="quantile"
)
calibration = pd.DataFrame({"predicted": predicted, "observed": observed})
calibration["gap"] = calibration["observed"] - calibration["predicted"]
print(calibration.round(3).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(6, 5))
limit = max(predicted.max(), observed.max()) * 1.05
ax.plot([0, limit], [0, limit], color="gray", linestyle="--", label="Perfect calibration")
ax.plot(predicted, observed, marker="o", label="Baseline")
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_xlabel("Average predicted risk")
ax.set_ylabel("Share who stayed more than 7 days")
ax.set_title("Baseline calibration (out-of-fold, training set)")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "baseline_logistic" / "calibration.png", dpi=150)

# %% [markdown]
# The curve sits on the diagonal, with every group within about 1 percentage point.
# That's what we'd expect here: logistic regression tends to be well calibrated on data
# like what it was fit on, and these folds come from the same sample. This doesn't test
# calibration by group, which comes at the final stage.
#
# ### PRG026
#
# This diagnosis has no long stays in train, so its coefficient has no finite best value.
# What risk do these stays get?

# %%
is_prg026 = train["ccsr_diagnosis_code"] == "PRG026"
print(f"Stays: {is_prg026.sum():,}")
print(f"Long stays: {train.loc[is_prg026, 'y'].sum()}")
print(f"Average predicted risk: {predictions.loc[is_prg026, 'risk'].mean():.3%}")
print(f"Highest predicted risk: {predictions.loc[is_prg026, 'risk'].max():.3%}")

# %% [markdown]
# These stays get a very low risk, which matches the data, and every fold converged, so
# we keep this level as it is. Its coefficient doesn't mean anything, and the paper will
# say so.

# %% [markdown]
# ## Step 7: Final model and recalibration
#
# Decisions are in `docs/notes/2026-11-12-recalibration-and-comparison.md`. The
# baseline is refit on all of train, then predicts the recalibration set. Platt
# scaling and isotonic regression are fit on those risks with
# `common.fit_recalibrators`. Both are judged only on the test set, in the comparison
# notebook.
#
# The recalibration set gets the same cleaning as train. To refit the model, delete
# `model.joblib`. The predictions and recalibrators take seconds, so they rerun every
# time and overwrite the saved files.

# %%
recal = common.clean(recal)
OUT_DIR = RESULTS_DIR / "baseline_logistic"


def fit_final_model():
    model = clone(baseline)
    model.fit(train[FEATURES], train["y"])
    return {"model": model}


final_model = common.load_or_compute(
    {"model": OUT_DIR / "model.joblib"}, fit_final_model
)["model"]
assert final_model[-1].n_iter_[0] < MAX_ITERATIONS, "The final model didn't converge"

# %% [markdown]
# Recalibration-set stays with a level the encoder never saw, in a feature with no
# infrequent column, get all zeros for that feature, as in Step 5. How many are there?

# %%
print(common.unseen_levels(final_model[0], recal).to_string())

# %%
recal_predictions = common.predict_risk(final_model, recal)
common.check_rows(recal_predictions, recal)
recal_predictions.to_parquet(OUT_DIR / "recal_predictions.parquet")

recalibrators = common.fit_recalibrators(recal_predictions["risk"], recal["y"])
joblib.dump(recalibrators, OUT_DIR / "recalibrators.joblib")
print(
    f"Platt: slope {recalibrators['platt'].coef_[0, 0]:.3f}, "
    f"intercept {recalibrators['platt'].intercept_[0]:.3f}"
)
