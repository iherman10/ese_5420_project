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
#     display_name: ese-5420-project (3.13.x)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline model
#
# Predicting whether an unplanned New York hospital admission lasts more than
# 7 days. Decisions are in `docs/notes/2026-11-05-data-cohort-split.md`.

# %%
import re
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 42

# %% [markdown]
# ## Step 1: Get the data
#
# Download the 2023 and 2024 SPARCS files unless they're already there, then
# stop with an error if either file isn't the expected size.

# %%
RAW_DIR = Path("data/raw")
URL = "https://health.data.ny.gov/api/views/{id}/rows.csv?accessType=DOWNLOAD"
DATASETS = {2023: "46xm-urtu", 2024: "sf4k-39ay"}
EXPECTED_ROWS = {2023: 2_125_754, 2024: 2_196_737}
EXPECTED_COLUMNS = 33

raw = {}
for year, dataset_id in DATASETS.items():
    path = RAW_DIR / f"sparcs_{year}.csv"
    if not path.exists():
        print(f"Downloading {year}...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(id=dataset_id), path)

    raw[year] = pd.read_csv(path, dtype=str, engine="pyarrow")
    assert raw[year].shape == (EXPECTED_ROWS[year], EXPECTED_COLUMNS), (
        f"{path} has shape {raw[year].shape}. If the download was interrupted, "
        "delete the file and run this again."
    )
    print(f"{year}: {raw[year].shape[0]:,} rows, {raw[year].shape[1]} columns")

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
def snake(name):
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


columns = {year: {snake(c) for c in df.columns} for year, df in raw.items()}
print("Only in 2023:", sorted(columns[2023] - columns[2024]))
print("Only in 2024:", sorted(columns[2024] - columns[2023]))
print("Year column present:", "discharge_year" in columns[2023] & columns[2024])

# %% [markdown]
# Two columns have different names, so we rename them. Without this, each would
# split into two half-empty columns. The files already have a year column, so we
# don't need to add one.

# %%
RENAMES = {
    "hospital_service_area": "health_service_area",
    "zip_code_3_digits": "zip_code",
}

frames = [df.rename(columns=snake).rename(columns=RENAMES) for df in raw.values()]
assert set(frames[0].columns) == set(frames[1].columns)

cohort = pd.concat(frames, ignore_index=True)
del raw, frames
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
los_text = cohort["length_of_stay"].str.replace(" ", "").replace("120+", "120")
cohort["los"] = pd.to_numeric(los_text)
cohort["y"] = (cohort["los"] >= 8).astype(int)

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
KEEP_ADMISSION_TYPES = ["Emergency", "Urgent", "Trauma"]

cohort = cohort[cohort["type_of_admission"].isin(KEEP_ADMISSION_TYPES)]
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Newborns
#
# The admission type filter removes stays labeled "Newborn". Are any newborns
# left under other admission types? APR MDC code 15 is the newborn diagnostic
# group.

# %%
newborns = cohort[cohort["apr_mdc_code"] == "15"]
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
cohort = cohort[cohort["apr_mdc_code"] != "15"]
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
cohort = cohort[cohort["permanent_facility_id"].notna()]
print(f"{len(cohort):,} stays remain")

# %% [markdown]
# ### Specialty facilities
#
# We want a model for typical hospitals. The publisher lists 10 facilities whose
# discharges are mostly rehab or life support. How many of their stays are left?

# %%
SPECIALTY_FACILITIES = [
    "000340",
    "000775",
    "000831",
    "000968",
    "001046",
    "001138",
    "001460",
    "001486",
    "009431",
    "010223",
]

is_specialty = cohort["permanent_facility_id"].isin(SPECIALTY_FACILITIES)
print(f"{is_specialty.sum():,} stays, {cohort.loc[is_specialty, 'y'].mean():.1%} long")
print()
print(long_stay_table(cohort[is_specialty], "permanent_facility_id").to_string())

# %% [markdown]
# Very few, and only five of the 10 facilities have any stays left. We drop them
# so that the rule is simple: the publisher's list defines a specialty facility.

# %%
cohort = cohort[~is_specialty]
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
EXPECTED_COHORT = {"2023": 1_562_648, "2024": 1_629_492}

stays_by_year = cohort["discharge_year"].value_counts().to_dict()
assert stays_by_year == EXPECTED_COHORT, stays_by_year
assert len(cohort) == 3_192_140, len(cohort)
assert cohort["los"].notna().all()
assert round(cohort["y"].mean(), 4) == 0.2309, cohort["y"].mean()

print(f"Cohort: {len(cohort):,} stays, {cohort['y'].mean():.2%} long")
print(pd.Series(stays_by_year).to_string())

# %% [markdown]
# ### Draw the sample
#
# To keep the slower models practical to train, all models use one simple random
# sample of 500,000 stays from the roughly 3.2 million in the cohort.

# %%
SAMPLE_SIZE = 500_000

sample = cohort.sample(n=SAMPLE_SIZE, random_state=SEED)
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
