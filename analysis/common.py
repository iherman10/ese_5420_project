"""Shared data, cohort, split, cleaning and encoding steps for every model notebook.

Decisions are in `docs/notes/2026-11-05-data-cohort-split.md`,
`docs/notes/2026-11-06-cleaning-encoding-baseline.md`,
`docs/notes/2026-11-07-saving-results.md`,
`docs/notes/2026-11-10-model-comparison-plan.md` and
`docs/notes/2026-11-12-recalibration-and-comparison.md`. `analysis/baseline_model.py` walks
through each step with a look at the data first.

Run notebooks from the repo root, so the relative paths below resolve, and import this
module with `from analysis import common`.
"""

import re
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

# Data

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


# Cohort

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


# Sample and split

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


# Cleaning

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
    """Load the data, build the cohort, sample, split and clean. Returns the cleaned
    train, recalibration and test sets."""
    sample = draw_sample(build_cohort(load_raw()))
    return tuple(clean(df) for df in split(sample))


# Encoding

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
    """Load saved results if they all exist. Otherwise compute and save them.

    `.csv` files are saved without the index, `.parquet` files with it (the row IDs),
    and `.joblib` files hold any Python object, such as a fitted model.
    """
    if all(path.exists() for path in paths.values()):
        print("Loaded saved results")
        return {name: _load(path) for name, path in paths.items()}

    results = compute()
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        _save(results[name], path)
    print("Computed and saved results")
    return results


def _load(path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".joblib":
        return joblib.load(path)
    raise ValueError(f"Unknown file type: {path}")


def _save(result, path):
    if path.suffix == ".csv":
        result.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        result.to_parquet(path)
    elif path.suffix == ".joblib":
        joblib.dump(result, path)
    else:
        raise ValueError(f"Unknown file type: {path}")


def check_rows(predictions, df):
    """Stop with an error if saved predictions don't match the current rows."""
    assert predictions.index.equals(df.index), "Saved results are out of date"
    assert predictions["risk"].notna().all()


# Model notebooks

UNKNOWN_LEVELS = "Found unknown categories"


def encode_folds(X, y, make_encoder=make_encoder):
    """Encode each CV fold once, fitting the encoder on the fold's training part only.

    New levels in a validation fold make the encoder warn, as expected, so that one
    warning is silenced.
    """
    folds = []
    for fold, (fit_rows, validation_rows) in enumerate(FOLDS.split(X, y), start=1):
        encoder = make_encoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=UNKNOWN_LEVELS)
            X_fit = encoder.fit_transform(X.iloc[fit_rows])
            X_validation = encoder.transform(X.iloc[validation_rows])
        folds.append(
            {
                "fold": fold,
                "validation_rows": validation_rows,
                "X_fit": X_fit,
                "y_fit": y.iloc[fit_rows].to_numpy(),
                "X_validation": X_validation,
                "y_validation": y.iloc[validation_rows].to_numpy(),
            }
        )
    return folds


# The training set's long-stay rate (83,026 of 360,000 stays). Flagging at it marks any
# stay the model rates as riskier than average. See
# docs/notes/2026-11-12-base-rate-threshold.md.
THRESHOLD = 0.2306


def check_threshold(train):
    """Stop with an error if `THRESHOLD` isn't the training set's long-stay rate."""
    rate = train["y"].mean()
    assert abs(rate - THRESHOLD) < 5e-5, f"THRESHOLD {THRESHOLD} != base rate {rate:.4f}"


def fold_scores(fold, y_true, risk, seconds, **details):
    """One row of a CV scores table: the fold, its validation AUC, Brier, precision and
    recall, any model-specific details, then the fit time in seconds.

    Precision and recall flag stays with risk at or above `THRESHOLD`. Precision is NaN
    if no stay is flagged.
    """
    flagged = np.asarray(risk) >= THRESHOLD
    return {
        "fold": fold,
        "auc": roc_auc_score(y_true, risk),
        "brier": brier_score_loss(y_true, risk),
        "precision": precision_score(y_true, flagged, zero_division=np.nan),
        "recall": recall_score(y_true, flagged),
        **details,
        "seconds": seconds,
    }


def logistic_details(model):
    """Fit details of a fitted LogisticRegression, for `fold_scores`: the share of
    encoded columns with a nonzero coefficient, the solver's iterations and whether it
    stopped before `max_iter`."""
    iterations = int(model.n_iter_[0])
    return {
        "nonzero_coef_share": float((model.coef_ != 0).mean()),
        "iterations": iterations,
        "converged": iterations < model.max_iter,
    }


def show_scores(scores):
    """Print a scores table with 4 significant figures."""
    print(scores.to_string(index=False, float_format="{:.4g}".format))


RESULT_SCORES = ["auc", "brier", "precision", "recall"]


def cv_results(model, scores):
    """A one-row results table: the mean and SD of each score across a model's folds."""
    row = {"model": model}
    for score in RESULT_SCORES:
        row[score] = scores[score].mean()
        row[f"{score}_sd"] = scores[score].std()
    return pd.DataFrame([row])


def oof_frame(index, folds, risks):
    """One out-of-fold risk per training stay, with its fold, indexed by row ID."""
    predictions = pd.DataFrame(index=index, columns=["fold", "risk"], dtype=float)
    for fold, risk in zip(folds, risks):
        predictions.iloc[fold["validation_rows"]] = np.column_stack(
            [np.full(len(risk), fold["fold"]), risk]
        )
    return predictions


def predict_risk(model, df):
    """A fitted pipeline's risk for each stay in `df`, indexed by row ID."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=UNKNOWN_LEVELS)
        risk = model.predict_proba(df[FEATURES])[:, 1]
    return pd.DataFrame({"risk": risk}, index=df.index)


# Recalibration

CLIP = 1e-6


def _log_odds(risk):
    p = np.clip(np.asarray(risk, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_recalibrators(risk, y):
    """Fit Platt scaling and isotonic regression on a model's recalibration-set risks.

    Platt is an unpenalized logistic regression on the log-odds of the risk. Isotonic
    gives risks outside the fitted range the nearest fitted value.
    """
    platt = LogisticRegression(C=np.inf).fit(_log_odds(risk), y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(np.asarray(risk), y)
    return {"platt": platt, "isotonic": isotonic}


def recalibrate(recalibrators, risk):
    """Apply both recalibrators to raw risks. Returns {"platt": ..., "isotonic": ...}."""
    return {
        "platt": recalibrators["platt"].predict_proba(_log_odds(risk))[:, 1],
        "isotonic": recalibrators["isotonic"].predict(np.asarray(risk)),
    }


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
