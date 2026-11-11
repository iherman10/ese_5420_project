"""Shared data, cohort, split, cleaning and encoding steps for every model notebook.

Decisions are in `docs/notes/2026-11-05-data-cohort-split.md`,
`docs/notes/2026-11-06-cleaning-encoding-baseline.md` and
`docs/notes/2026-11-10-model-comparison-plan.md`. `analysis/baseline_model.py` walks
through each step with a look at the data first.

Run notebooks from the repo root, so the relative paths below resolve, and import this
module with `from analysis import common`.
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

# Step 1: Get the data

URL = "https://health.data.ny.gov/api/views/{id}/rows.csv?accessType=DOWNLOAD"
DATASETS = {2023: "46xm-urtu", 2024: "sf4k-39ay"}
EXPECTED_ROWS = {2023: 2_125_754, 2024: 2_196_737}
EXPECTED_COLUMNS = 33


def load_raw():
    """Download the SPARCS files if needed and load them as text, keyed by year."""
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
    return raw


# Step 2: Build the cohort

RENAMES = {
    "hospital_service_area": "health_service_area",
    "zip_code_3_digits": "zip_code",
}
KEEP_ADMISSION_TYPES = ["Emergency", "Urgent", "Trauma"]
NEWBORN_MDC = "15"
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
EXPECTED_COHORT = {"2023": 1_562_648, "2024": 1_629_492}


def snake(name):
    """Convert a column name to snake_case."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def pool(raw):
    """Stack the two years with matching snake_case column names."""
    frames = [df.rename(columns=snake).rename(columns=RENAMES) for df in raw.values()]
    assert set(frames[0].columns) == set(frames[1].columns)
    return pd.concat(frames, ignore_index=True)


def add_target(df):
    """Add numeric length of stay (`los`) and the target `y` = 1 if it's at least 8."""
    los_text = df["length_of_stay"].str.replace(" ", "").replace("120+", "120")
    df["los"] = pd.to_numeric(los_text)
    df["y"] = (df["los"] >= 8).astype(int)
    return df


def keep_unplanned(df):
    """Keep emergency, urgent and trauma admissions."""
    return df[df["type_of_admission"].isin(KEEP_ADMISSION_TYPES)]


def drop_newborns(df):
    """Drop newborns admitted under the other admission types."""
    return df[df["apr_mdc_code"] != NEWBORN_MDC]


def drop_redacted(df):
    """Drop records whose facility and location the publisher redacted."""
    return df[df["permanent_facility_id"].notna()]


def drop_specialty(df):
    """Drop stays at the publisher's specialty (rehab and life-support) facilities."""
    return df[~df["permanent_facility_id"].isin(SPECIALTY_FACILITIES)]


def check_cohort(cohort):
    """Stop with an error if the cohort isn't the expected size."""
    stays_by_year = cohort["discharge_year"].value_counts().to_dict()
    assert stays_by_year == EXPECTED_COHORT, stays_by_year
    assert len(cohort) == 3_192_140, len(cohort)
    assert cohort["los"].notna().all()
    assert round(cohort["y"].mean(), 4) == 0.2309, cohort["y"].mean()


def build_cohort(raw):
    """Pool the years, add the target, remove out-of-scope stays and check the result."""
    cohort = add_target(pool(raw))
    cohort = drop_specialty(drop_redacted(drop_newborns(keep_unplanned(cohort))))
    check_cohort(cohort)
    return cohort


# Sample and step 3: Split

SAMPLE_SIZE = 500_000
TRAIN_SIZE = 360_000
RECAL_SIZE = 40_000
TEST_SIZE = 100_000


def draw_sample(cohort):
    """The one simple random sample of 500,000 stays that every model uses."""
    return cohort.sample(n=SAMPLE_SIZE, random_state=SEED)


def split(sample):
    """Split the sample into train, recalibration and test, stratified on `y`."""
    train_recal, test = train_test_split(
        sample, test_size=TEST_SIZE, stratify=sample["y"], random_state=SEED
    )
    train, recal = train_test_split(
        train_recal, test_size=RECAL_SIZE, stratify=train_recal["y"], random_state=SEED
    )
    assert [len(df) for df in (train, recal, test)] == [TRAIN_SIZE, RECAL_SIZE, TEST_SIZE]
    assert train.index.union(recal.index).union(test.index).size == SAMPLE_SIZE
    return train, recal, test


# Step 4: Clean

FEATURES = [
    "age_group",
    "gender",
    "race",
    "ethnicity",
    "zip_code",
    "payment_typology_1",
    "payment_typology_2",
    "payment_typology_3",
    "type_of_admission",
    "emergency_department_indicator",
    "permanent_facility_id",
    "hospital_county",
    "health_service_area",
    "apr_mdc_code",
    "apr_medical_surgical_description",
    "ccsr_diagnosis_code",
    "discharge_year",
]
LABEL_FIXES = {
    "age_group": {
        "0 to 17": "0-17",
        "18 to 29": "18-29",
        "30 to 49": "30-49",
        "50 to 69": "50-69",
    },
    "health_service_area": {"Capital/Adirond": "Capital/Adirondacks"},
    "hospital_county": {"Manhattan": "New York", "St Lawrence": "Saint Lawrence"},
}
BLANK_COLUMNS = [
    "zip_code",
    "payment_typology_1",
    "payment_typology_2",
    "payment_typology_3",
]


def clean(df):
    """Apply the label fixes and fill blanks. Learns nothing from the data."""
    return df.replace(LABEL_FIXES).fillna({c: "Unknown" for c in BLANK_COLUMNS})


def load_splits():
    """Run steps 1-4 and return the cleaned train, recalibration and test sets."""
    sample = draw_sample(build_cohort(load_raw()))
    return tuple(clean(df) for df in split(sample))


# Step 5: Encode

MIN_STAYS = 100


def make_encoder(drop=None):
    """One-hot encoder with rare levels grouped into "Other".

    The baseline uses `drop="first"`. Every other model uses the default, full one-hot.
    """
    return OneHotEncoder(
        min_frequency=MIN_STAYS, handle_unknown="infrequent_if_exist", drop=drop
    )


# Cross-validation and saved results

FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def load_or_compute(paths, compute):
    """Load saved tables if they all exist. Otherwise compute and save them."""
    if all(path.exists() for path in paths.values()):
        print("Loaded saved results")
        return {
            name: pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            for name, path in paths.items()
        }

    results = compute()
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            results[name].to_csv(path, index=False)
        else:
            results[name].to_parquet(path)
    print("Computed and saved results")
    return results


def check_rows(predictions, df):
    """Stop with an error if saved predictions don't match the current rows."""
    assert predictions.index.equals(df.index), "Saved results are out of date"
    assert predictions["risk"].notna().all()


# Fairness groups

GROUPS = {
    "payment_typology_1": {
        "Medicaid": ["Medicaid"],
        "Commercial": ["Private Health Insurance", "Blue Cross/Blue Shield"],
        "Medicare": ["Medicare"],
    },
    "race": {
        "White": ["White"],
        "Black/African American": ["Black/African American"],
        "Other Race": ["Other Race"],
    },
}
