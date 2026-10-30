"""Verify the data claims in docs/proposal_v2.md.

Downloads the 2023 and 2024 SPARCS inpatient discharge CSVs into data/raw/
(if not already there) and prints row/column counts, admission-type mix,
long-stay prevalence, missingness and feature cardinality.

    uv run python analysis/verify_data.py
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

DATASETS = {
    2023: "46xm-urtu",
    2024: "sf4k-39ay",
}
URL = "https://health.data.ny.gov/api/views/{id}/rows.csv?accessType=DOWNLOAD"

KEEP_ADMISSION_TYPES = {"Emergency", "Urgent", "Trauma"}

# 2023 and 2024 name these columns differently.
RENAMES = {
    "hospital_service_area": "health_service_area",
    "zip_code_3_digits": "zip_code",
}


def snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def download(year: int) -> Path:
    path = RAW / f"sparcs_{year}.csv"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {year} to {path} ...")
        urllib.request.urlretrieve(URL.format(id=DATASETS[year]), path)
    return path


def load(year: int) -> pd.DataFrame:
    df = pd.read_csv(download(year), dtype=str, engine="pyarrow")
    df.columns = [snake(c) for c in df.columns]
    return df.rename(columns=RENAMES)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    frames = {}
    for year in DATASETS:
        df = load(year)
        section(f"{year}: shape")
        print(f"{len(df):,} rows x {df.shape[1]} columns")
        frames[year] = df

    cols = {y: set(df.columns) for y, df in frames.items()}
    if cols[2023] != cols[2024]:
        section("Column mismatch after renames")
        print("Only in 2023:", sorted(cols[2023] - cols[2024]))
        print("Only in 2024:", sorted(cols[2024] - cols[2023]))

    df = pd.concat(
        [f.assign(discharge_year=y) for y, f in frames.items()], ignore_index=True
    )

    section("Admission type (all stays)")
    print(df["type_of_admission"].value_counts(dropna=False).to_string())

    df = df[df["type_of_admission"].isin(KEEP_ADMISSION_TYPES)].copy()
    section("Kept stays (emergency, urgent, trauma)")
    print(f"{len(df):,} stays")
    print(df["discharge_year"].value_counts().sort_index().to_string())

    # Top value is "120 +" in 2023 and "120+" in 2024.
    los = df["length_of_stay"].str.replace(" ", "").replace("120+", "120")
    df["los"] = pd.to_numeric(los, errors="coerce")
    df["y"] = (df["los"] >= 8).astype(int)

    section("Length of stay")
    print(df["los"].describe().to_string())
    print(f"Unparseable LOS values: {df['los'].isna().sum():,}")
    print(f"120+ stays: {(df['los'] == 120).sum():,}")

    section("Prevalence of y = 1 (LOS >= 8)")
    print(f"Overall: {df['y'].mean():.3f}")
    print(df.groupby("discharge_year")["y"].mean().round(3).to_string())

    section("Missingness on kept stays (% missing, nonzero only)")
    # Treat SPARCS placeholder strings as missing too.
    blanks = df.isna() | df.isin(["", "Not Available", "Unknown"])
    miss = (blanks.mean() * 100).round(2).sort_values(ascending=False)
    print(miss[miss > 0].to_string())

    section("Cardinality (distinct levels) on kept stays")
    print(df.nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()


# Output of `uv run python analysis/verify_data.py`, run 2026-10-30:
#
# === 2023: shape ===
# 2,125,754 rows x 33 columns
#
# === 2024: shape ===
# 2,196,737 rows x 33 columns
#
# === Admission type (all stays) ===
# type_of_admission
# Emergency        2870379
# Elective          712698
# Newborn           392468
# Urgent            322676
# Trauma             18117
# Not Available       6153
#
# === Kept stays (emergency, urgent, trauma) ===
# 3,211,172 stays
# discharge_year
# 2023    1572202
# 2024    1638970
#
# === Length of stay ===
# count    3.211172e+06
# mean     6.292115e+00
# std      8.923033e+00
# min      1.000000e+00
# 25%      2.000000e+00
# 50%      4.000000e+00
# 75%      7.000000e+00
# max      1.200000e+02
# Unparseable LOS values: 0
# 120+ stays: 3,423
#
# === Prevalence of y = 1 (LOS >= 8) ===
# Overall: 0.230
# discharge_year
# 2023    0.231
# 2024    0.229
#
# === Missingness on kept stays (% missing, nonzero only) ===
# birth_weight                           99.45
# payment_typology_3                     86.50
# payment_typology_2                     49.57
# ccsr_procedure_description             33.83
# ccsr_procedure_code                    33.83
# ethnicity                              11.98
# zip_code                                1.86
# hospital_county                         0.27
# health_service_area                     0.27
# operating_certificate_number            0.27
# permanent_facility_id                   0.27
# apr_risk_of_mortality                   0.01
# apr_severity_of_illness_description     0.01
#
# === Cardinality (distinct levels) on kept stays ===
# total_charges                          2965974
# total_costs                            2633642
# ccsr_diagnosis_description                 952
# ccsr_diagnosis_code                        483
# facility_name                              392
# apr_drg_description                        338
# apr_drg_code                               336
# ccsr_procedure_description                 320
# ccsr_procedure_code                        320
# permanent_facility_id                      203
# operating_certificate_number               157
# birth_weight                               121
# length_of_stay                             121
# los                                        120
# hospital_county                             59
# zip_code                                    50
# apr_mdc_description                         48
# apr_mdc_code                                26
# patient_disposition                         20
# payment_typology_2                           9
# payment_typology_1                           9
# payment_typology_3                           9
# health_service_area                          9
# age_group                                    9
# apr_severity_of_illness_description          5
# apr_risk_of_mortality                        5
# apr_severity_of_illness_code                 5
# ethnicity                                    4
# race                                         4
# apr_medical_surgical_description             3
# type_of_admission                            3
# gender                                       3
# discharge_year                               2
# emergency_department_indicator               2
# y                                            2
