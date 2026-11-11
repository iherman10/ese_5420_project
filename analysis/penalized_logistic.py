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
# # Penalized logistic regression
#
# Ridge, lasso and elastic net logistic regression for predicting whether an unplanned
# New York hospital admission lasts more than 7 days. Decisions are in
# `docs/notes/2026-11-10-model-comparison-plan.md` and
# `docs/notes/2026-11-11-penalized-logistic.md`. The data, cohort, split, cleaning and
# encoder come from `analysis/common.py`, and `analysis/baseline_model.py` walks
# through them.

# %%
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from analysis import common

RESULTS_DIR = common.RESULTS_DIR / "penalized_logistic"

# %% [markdown]
# ## Step 1: Setup
#
# The same training set, features and folds as the baseline. The recalibration and
# test sets aren't used here.

# %%
train, recal, test = common.load_splits()
X, y = train[common.FEATURES], train["y"]
print(f"{len(train):,} training stays, {y.mean():.1%} stayed more than 7 days")
print(f"{len(common.FEATURES)} features")

# %% [markdown]
# ## Step 2: The model
#
# Elastic net logistic regression adds a penalty on the size of the coefficients to the
# usual fit. The penalty is a mix of two kinds, set by `l1_ratio`:
#
# - **`l1_ratio = 0` is ridge.** It shrinks every coefficient toward zero, but never
#   all the way.
# - **`l1_ratio = 1` is lasso.** It can set coefficients exactly to zero, which drops
#   those levels from the model.
# - **In between is elastic net,** which does some of both.
#
# `C` sets how much the penalty matters. Small `C` is a strong penalty. Large `C` is a
# weak one, and as `C` grows the model approaches the unpenalized baseline. The fit
# adds up the error over every row but the penalty doesn't grow with the data, so with
# 288,000 rows per fold even `C = 1` is a very weak penalty.
#
# One grid covers all three penalties:
#
# - `l1_ratio` in {0, 0.5, 1}
# - 11 values of `C` from 1e-5 to 1, evenly spaced on a log scale
#
# That's 33 settings × 5 folds = 165 fits. Every setting uses full one-hot encoding with
# no level dropped, and no scaling, since every column is a 0/1 dummy. `saga` is the
# only scikit-learn solver that handles every `l1_ratio`.

# %%
L1_RATIOS = [0.0, 0.5, 1.0]
PENALTY_NAMES = {0.0: "Ridge", 0.5: "Elastic net", 1.0: "Lasso"}
CS = np.logspace(-5, 0, 11)
MAX_ITERATIONS = 1_000


def make_model(l1_ratio, C):
    return LogisticRegression(
        solver="saga",
        l1_ratio=l1_ratio,
        C=C,
        max_iter=MAX_ITERATIONS,
        random_state=common.SEED,
    )


# %% [markdown]
# ## Step 3: Tune
#
# The encoder is fit on each fold's training part only. It doesn't depend on `C` or
# `l1_ratio`, so each fold is encoded once and reused for all 33 settings. New levels
# in a validation fold make the encoder warn, as expected, so we silence that one
# warning.
#
# The 165 fits run in parallel. For each one we save the validation AUC and Brier, the
# number of iterations, whether it converged, how many coefficients are nonzero and the
# fit time. To recompute, delete `results/penalized_logistic/tuning.csv`.


# %%
def encode_folds():
    """Encode each fold, fitting the encoder on the fold's training part only."""
    folds = []
    for fold, (fit_rows, validation_rows) in enumerate(
        common.FOLDS.split(X, y), start=1
    ):
        encoder = common.make_encoder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found unknown categories")
            X_fit = encoder.fit_transform(X.iloc[fit_rows])
            X_validation = encoder.transform(X.iloc[validation_rows])
        folds.append(
            {
                "fold": fold,
                "X_fit": X_fit,
                "y_fit": y.iloc[fit_rows].to_numpy(),
                "X_validation": X_validation,
                "y_validation": y.iloc[validation_rows].to_numpy(),
            }
        )
    return folds


def fit_one(fold, l1_ratio, C):
    """Fit one setting on one fold and score it on the validation part."""
    model = make_model(l1_ratio, C)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(fold["X_fit"], fold["y_fit"])
    seconds = time.perf_counter() - start
    risk = model.predict_proba(fold["X_validation"])[:, 1]
    iterations = model.n_iter_[0]
    return {
        "l1_ratio": l1_ratio,
        "C": C,
        "fold": fold["fold"],
        "auc": roc_auc_score(fold["y_validation"], risk),
        "brier": brier_score_loss(fold["y_validation"], risk),
        "nonzero": int((model.coef_ != 0).sum()),
        "columns": model.coef_.shape[1],
        "iterations": iterations,
        "converged": iterations < MAX_ITERATIONS,
        "seconds": seconds,
    }


def run_tuning():
    folds = encode_folds()
    rows = Parallel(n_jobs=-1, verbose=5)(
        delayed(fit_one)(fold, l1_ratio, C)
        for l1_ratio in L1_RATIOS
        for C in CS
        for fold in folds
    )
    return {"tuning": pd.DataFrame(rows)}


tuning = common.load_or_compute(
    {"tuning": RESULTS_DIR / "tuning.csv"}, run_tuning
)["tuning"]

assert len(tuning) == len(L1_RATIOS) * len(CS) * common.FOLDS.get_n_splits()
print(f"{tuning['seconds'].sum() / 60:.0f} minutes of fitting in total")
print(f"{(~tuning['converged']).sum()} of {len(tuning)} fits didn't converge")
print(tuning[~tuning["converged"]].to_string(index=False))

# %% [markdown]
# ## Step 4: Tuning curve
#
# Mean and SD of each score across the 5 folds, for every setting. The baseline's CV
# AUC and Brier, from `results/baseline_logistic/cv_scores.csv`, are the reference.

# %%
summary = (
    tuning.groupby(["l1_ratio", "C"])
    .agg(
        auc=("auc", "mean"),
        auc_sd=("auc", "std"),
        brier=("brier", "mean"),
        brier_sd=("brier", "std"),
        nonzero=("nonzero", "mean"),
        converged=("converged", "all"),
        seconds=("seconds", "mean"),
    )
    .reset_index()
)
summary.insert(0, "penalty", summary["l1_ratio"].map(PENALTY_NAMES))

baseline_scores = pd.read_csv(common.RESULTS_DIR / "baseline_logistic" / "cv_scores.csv")
baseline_auc = baseline_scores["auc"].mean()
baseline_brier = baseline_scores["brier"].mean()
print(f"Baseline: AUC {baseline_auc:.4f}, Brier {baseline_brier:.4f}")

print(summary.drop(columns="l1_ratio").to_string(index=False, float_format="{:.4g}".format))

# %%
COLORS = {0.0: "#2a78d6", 0.5: "#eb6834", 1.0: "#1baf7a"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for ax, score, baseline_value, label in [
    (axes[0], "auc", baseline_auc, "CV AUC (higher is better)"),
    (axes[1], "brier", baseline_brier, "CV Brier score (lower is better)"),
]:
    ax.axhline(baseline_value, color="gray", linestyle="--", linewidth=1, label="Baseline")
    for l1_ratio in L1_RATIOS:
        rows = summary[summary["l1_ratio"] == l1_ratio]
        ax.errorbar(
            rows["C"],
            rows[score],
            yerr=rows[f"{score}_sd"],
            color=COLORS[l1_ratio],
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=PENALTY_NAMES[l1_ratio],
        )
    ax.set_xscale("log")
    ax.set_xlabel("C (smaller is a stronger penalty)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend(loc="lower right")
fig.suptitle("Penalized logistic regression: 5-fold CV scores by penalty strength")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "tuning.png", dpi=150)

# %% [markdown]
# ### Best setting for each penalty
#
# The highest mean CV AUC within each penalty. If the best `C` is at either end of the
# grid, the grid should be extended that way.

# %%
best = summary.loc[summary.groupby("l1_ratio")["auc"].idxmax()].copy()
best["at_edge"] = best["C"].isin([CS.min(), CS.max()])
best["auc_vs_baseline"] = best["auc"] - baseline_auc
print(
    best[
        [
            "penalty",
            "C",
            "auc",
            "auc_sd",
            "auc_vs_baseline",
            "brier",
            "nonzero",
            "at_edge",
        ]
    ].to_string(index=False, float_format="{:.4g}".format)
)
