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
# # Random forest
#
# A random forest for predicting whether an unplanned New York hospital admission lasts
# more than 7 days. Decisions are in `docs/notes/2026-11-10-model-comparison-plan.md` and
# `docs/notes/2026-11-24-random-forest.md`. The data, cohort, split, cleaning and encoder
# come from `analysis/common.py`, and `analysis/baseline_model.py` walks through them.

# %%
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from analysis import common

RESULTS_DIR = common.RESULTS_DIR / "random_forest"

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
# A random forest grows many deep trees, each on a bootstrap sample of the stays, and
# averages their risks. At each split a tree only considers a random subset of the
# columns, which makes the trees less alike, so averaging them cancels more of their
# noise. A stay's risk is the average over the trees of the share of long stays in the
# leaf it lands in.
#
# The grid:
#
# - **`max_features`** in {`"sqrt"`, 0.1, 0.2, 0.3}: the share of columns each split
#   considers. It counts the one-hot columns, not the 17 features, and most columns are
#   rare levels, so `"sqrt"` (about 25 of about 610 columns) mostly draws columns that
#   can't split much. That's why the grid goes well above it.
# - **`min_samples_leaf`** in {5, 10, 25}: the fewest training stays a leaf may hold.
#   Averaging trees removes variance, so each tree can be grown deeper than the single
#   tree, whose best leaf size was 25. A leaf size of 1, the textbook default, scored
#   worst on fold 1 and is too slow and too big here.
#
# That's 12 settings × 5 folds = 60 fits. Fixed settings:
#
# - **100 trees for tuning,** 300 for the chosen model. On fold 1, 100 trees was within
#   0.001 AUC of 300, which is close enough to rank settings, and more trees never
#   overfit.
# - **Full bootstrap samples, Gini, no depth limit,** scikit-learn's defaults.
# - **No `class_weight`,** since reweighting the classes would distort the risks.

# %%
MAX_FEATURES = ["sqrt", 0.1, 0.2, 0.3]
MIN_SAMPLES_LEAF = [5, 10, 25]
TUNING_TREES = 100
SETTINGS = ["max_features", "min_samples_leaf"]


def make_model(max_features, min_samples_leaf, n_estimators=TUNING_TREES):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=8,
        random_state=common.SEED,
    )


# %% [markdown]
# ### Fits
#
# `common.encode_folds` fits the encoder on each fold's training part only, and each fold
# is encoded once and reused for all 12 settings. The input stays sparse, which was twice
# as fast as dense on fold 1, with identical results.
#
# The fits run one at a time, each on all 8 cores, so only one forest is in memory at a
# time. For each fit we save the validation AUC, Brier, precision and recall, the trees'
# mean number of leaves, and the fit time. The whole grid takes about 2.5–3 hours, most
# of it at `max_features = 0.3`. To recompute, delete `results/random_forest/tuning.csv`.


# %%
def fit_one(fold, max_features, min_samples_leaf, n_estimators=TUNING_TREES):
    """Fit one setting on one fold and score it on the validation part."""
    model = make_model(max_features, min_samples_leaf, n_estimators)
    start = time.perf_counter()
    model.fit(fold["X_fit"], fold["y_fit"])
    seconds = time.perf_counter() - start
    risk = model.predict_proba(fold["X_validation"])[:, 1]
    scores = {
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        **common.fold_scores(
            fold["fold"],
            fold["y_validation"],
            risk,
            seconds,
            mean_leaves=np.mean([tree.get_n_leaves() for tree in model.estimators_]),
        ),
    }
    return model, risk, scores


def run_tuning():
    folds = common.encode_folds(X, y)
    rows = []
    for max_features in MAX_FEATURES:
        for min_samples_leaf in MIN_SAMPLES_LEAF:
            for fold in folds:
                _, _, scores = fit_one(fold, max_features, min_samples_leaf)
                rows.append(scores)
                print(
                    f"max_features {max_features}, leaf {min_samples_leaf}, "
                    f"fold {fold['fold']}: AUC {scores['auc']:.4f}, "
                    f"{scores['seconds']:.0f}s",
                    flush=True,
                )
    return {"tuning": pd.DataFrame(rows)}


tuning = common.load_or_compute(
    {"tuning": RESULTS_DIR / "tuning.csv"}, run_tuning
)["tuning"]

assert len(tuning) == len(MAX_FEATURES) * len(MIN_SAMPLES_LEAF) * common.FOLDS.get_n_splits()
print(f"{tuning['seconds'].sum() / 60:.0f} minutes of fitting in total")
