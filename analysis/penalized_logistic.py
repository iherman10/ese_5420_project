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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from analysis import common

RESULTS_DIR = common.RESULTS_DIR / "penalized_logistic"

# %% [markdown]
# ## Setup
#
# The same training set, features and folds as the baseline. The recalibration set is
# only used in Recalibration, where the final model predicts on it. The test set isn't used
# yet.

# %%
train, recal, test = common.load_splits()
X, y = train[common.FEATURES], train["y"]
print(f"{len(train):,} training stays, {y.mean():.1%} stayed more than 7 days")
print(f"{len(common.FEATURES)} features")

# %% [markdown]
# ## Tuning
#
# ### The model
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
# ### Fits
#
# `common.encode_folds` fits the encoder on each fold's training part only. The encoder
# doesn't depend on `C` or `l1_ratio`, so each fold is encoded once and reused for all
# 33 settings.
#
# The 165 fits run in parallel. For each one we save the validation AUC and Brier, the
# share of coefficients that are nonzero, the number of iterations, whether it converged
# and the fit time. To recompute, delete `results/penalized_logistic/tuning.csv`.
#
# The saved grid was fit before precision and recall were added to the fold scores, so
# the tuning tables show AUC and Brier only. Refitting the grid, about 300 CPU-minutes,
# would add them. The chosen setting's tables below have all four.


# %%
def fit_one(fold, l1_ratio, C, return_risk=False):
    """Fit one setting on one fold and score it on the validation part."""
    model = make_model(l1_ratio, C)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(fold["X_fit"], fold["y_fit"])
    seconds = time.perf_counter() - start
    risk = model.predict_proba(fold["X_validation"])[:, 1]
    scores = {
        "l1_ratio": l1_ratio,
        "C": C,
        **common.fold_scores(
            fold["fold"],
            fold["y_validation"],
            risk,
            seconds,
            **common.logistic_details(model),
        ),
    }
    return (scores, risk) if return_risk else scores


def with_penalty(scores):
    """Name each row's penalty in place of its `l1_ratio`, for display."""
    named = scores.drop(columns="l1_ratio")
    named.insert(0, "penalty", scores["l1_ratio"].map(PENALTY_NAMES))
    return named


def run_tuning():
    folds = common.encode_folds(X, y)
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
common.show_scores(with_penalty(tuning[~tuning["converged"]]))

# %% [markdown]
# ### Tuning curve
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
        nonzero_coef_share=("nonzero_coef_share", "mean"),
        converged=("converged", "all"),
        seconds=("seconds", "mean"),
    )
    .reset_index()
)

baseline_scores = pd.read_csv(common.RESULTS_DIR / "baseline_logistic" / "cv_scores.csv")
baseline_auc = baseline_scores["auc"].mean()
baseline_brier = baseline_scores["brier"].mean()
print(f"Baseline: AUC {baseline_auc:.4f}, Brier {baseline_brier:.4f}")

common.show_scores(with_penalty(summary))

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
# The highest mean CV AUC within each penalty, with each score's difference from the
# baseline. All three best `C` values are at the top end of the grid, and the Findings
# explain why the grid isn't extended.
#
# `nonzero_coef_share` is the share of encoded columns with a nonzero coefficient,
# averaged over the folds.

# %%
best = with_penalty(summary.loc[summary.groupby("l1_ratio")["auc"].idxmax()])
best["auc_vs_baseline"] = best["auc"] - baseline_auc
best["brier_vs_baseline"] = best["brier"] - baseline_brier
common.show_scores(
    best[
        [
            "penalty",
            "C",
            "auc",
            "auc_sd",
            "auc_vs_baseline",
            "brier",
            "brier_sd",
            "brier_vs_baseline",
            "nonzero_coef_share",
        ]
    ],
    highlight=True,
)

# %% [markdown]
# ### Chosen model
#
# Lasso at `C = 1`, the best mean CV AUC. It also has the lowest Brier, and it drops
# about 9% of the encoded columns. Ridge and elastic net trail by only 0.00001–0.00002.
#
# #### Out-of-fold predictions
#
# The tuning step only saved scores, so we refit the chosen setting on the same 5 folds
# to get one out-of-fold risk for each training stay. The fits are deterministic, so
# the scores must match the tuning rows. To recompute, delete `cv_scores.csv` and
# `oof_predictions.parquet`.

# %%
CHOSEN = {"l1_ratio": 1.0, "C": 1.0}


def run_chosen_cv():
    folds = common.encode_folds(X, y)
    fits = Parallel(n_jobs=-1)(
        delayed(fit_one)(fold, **CHOSEN, return_risk=True) for fold in folds
    )
    return {
        "scores": pd.DataFrame([scores for scores, _ in fits]),
        "predictions": common.oof_frame(train.index, folds, [risk for _, risk in fits]),
    }


chosen_cv = common.load_or_compute(
    {
        "scores": RESULTS_DIR / "cv_scores.csv",
        "predictions": RESULTS_DIR / "oof_predictions.parquet",
    },
    run_chosen_cv,
)
cv_scores = chosen_cv["scores"]
oof_predictions = chosen_cv["predictions"]

common.check_rows(oof_predictions, train)
assert cv_scores["converged"].all(), "A fold didn't converge"
tuned = tuning[(tuning["l1_ratio"] == CHOSEN["l1_ratio"]) & (tuning["C"] == CHOSEN["C"])]
for column in ["auc", "nonzero_coef_share"]:
    assert np.allclose(cv_scores[column], tuned.sort_values("fold")[column])
common.show_scores(with_penalty(cv_scores))

# %% [markdown]
# #### Results
#
# The same results table as the baseline's. Each score is the mean and SD across the 5
# validation folds. Precision and recall use the same base-rate threshold as the
# baseline (`common.THRESHOLD`). They're for awareness only, and we don't use them to
# choose models.

# %%
common.check_threshold(train)
results = common.cv_results("Lasso logistic", cv_scores)
common.show_scores(results)

# %% [markdown]
# ## Recalibration
#
# ### Final model
#
# The encoder and lasso refit on all of train. To refit, delete `model.joblib`.

# %%
def fit_final_model():
    model = make_pipeline(common.make_encoder(), make_model(**CHOSEN))
    model.fit(X, y)
    return {"model": model}


final_model = common.load_or_compute(
    {"model": RESULTS_DIR / "model.joblib"}, fit_final_model
)["model"]
lasso = final_model[-1]
assert (lasso.l1_ratio, lasso.C) == (CHOSEN["l1_ratio"], CHOSEN["C"]), "Saved model is out of date"
assert lasso.n_iter_[0] < MAX_ITERATIONS, "The final model didn't converge"

# %% [markdown]
# ### Platt scaling and isotonic regression
#
# The final model predicts the recalibration set, and Platt scaling and isotonic
# regression are fit on those risks with `common.fit_recalibrators`. Both are judged
# only on the test set, in the comparison notebook. These steps take seconds, so they
# rerun every time and overwrite the saved files.

# %%
recal_predictions = common.predict_risk(final_model, recal)
common.check_rows(recal_predictions, recal)
recal_predictions.to_parquet(RESULTS_DIR / "recal_predictions.parquet")

recalibrators = common.fit_recalibrators(recal_predictions["risk"], recal["y"])
joblib.dump(recalibrators, RESULTS_DIR / "recalibrators.joblib")


# %% [markdown]
# ## Findings
#
# Takeaways so far, from the tuning results above. They'll be updated as diagnostics are
# added.
#
# - **Chosen model: lasso at `C = 1`.** It has the best mean CV AUC (0.77367) and the
#   lowest Brier (0.14459) of the three, and it drops about 9% of the encoded columns.
#   The margins over ridge and elastic net are only 0.00001–0.00002, so the three are
#   practically identical. `C` is chosen by best mean CV AUC.
# - **A penalty barely helps.** All three penalties do best at `C = 1`, the weakest
#   penalty in the grid. There they beat the baseline on every fold, by 0.00015–0.00028
#   AUC and with a lower Brier. The gain is consistent but too small to matter: CV AUC
#   0.7737 vs 0.7734. A likely reason is that even a weak penalty steadies the
#   coefficients of rare levels. Every stronger penalty makes both scores worse.
# - **Why: low variance, so the error is bias.** With about 288,000 rows per fold and
#   about 610 columns, the unpenalized coefficients are already precise. A penalty can
#   only add bias. The remaining error is what a linear model of these dummies can't
#   express, such as interactions, plus noise no model can remove. The tree models and
#   the MLP will show how much of it is bias.
# - **Best `C` at the edge of the grid.** The best `C` is at the top end, but the curves
#   are flat from `C = 0.1` up, and larger `C` just approaches the baseline. Extending
#   the grid upward wouldn't change anything.
# - **Strong L1 penalties zero every coefficient.** Lasso at `C` ≤ 1e-4 and elastic net
#   at `C` ≤ 3e-5 predict the base rate for every stay: AUC 0.5 and Brier about 0.178.
#   Ridge never zeros a coefficient, so it still ranks stays at AUC 0.68 even at
#   `C = 1e-5`.
# - **Sparsity costs nothing at `C = 1`.** Lasso keeps 91% of the columns and elastic
#   net 93%, with the same AUC as ridge.
# - **Convergence.** 4 of 165 fits hit `max_iter`: elastic net at `C` ≈ 0.32, folds 2–5.
#   Their scores match the converged neighbors, so they're fine for AUC and Brier.
#   Their `nonzero_coef_share` may be slightly high, since saga zeros coefficients
#   gradually.
# - **Interactions are deferred.** The tree models learn interactions from the same
#   features. If XGBoost clearly beats logistic regression, try a logistic regression
#   with a few interactions as an optional extra.
