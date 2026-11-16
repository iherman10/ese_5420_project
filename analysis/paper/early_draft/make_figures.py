"""Preliminary table and figure for the early draft, from saved cross-validation results.

Reads only training-set results: each model's `cv_scores.csv` (one row per CV fold) and
`oof_predictions.parquet` (one out-of-fold risk per training stay). The test set is never
used. The training labels for the ROC curves come from the same data, cohort, sample and
split steps as every notebook (`analysis/common.py`); only the training rows are kept.

Writes to `analysis/paper/early_draft/figures/`:

- `tab_cv_metrics.tex`: mean ± SD across the 5 folds of AUC, Brier, precision and
  recall, as `common.cv_results` reports them in the notebooks. Precision and recall flag
  stays at or above `common.THRESHOLD`, the training set's long-stay rate.
- `roc_oof.pdf`: ROC curves from the pooled out-of-fold risks, one per finished model.

Models without saved out-of-fold results (such as XGBoost before its final CV refit) are
skipped and listed.

Run from anywhere in the repo:

    uv run python analysis/paper/early_draft/make_figures.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)  # common.RESULTS_DIR and RAW_DIR are relative to the repo root
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from analysis import common  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures"

# Every planned model, in the paper's order: (results folder, label in table and figure).
MODELS = [
    ("baseline_logistic", "Logistic regression (baseline)"),
    ("penalized_logistic", "Lasso logistic regression"),
    ("decision_tree", "Decision tree"),
    ("xgboost", "XGBoost"),
    ("random_forest", "Random forest"),
    ("mlp", "MLP"),
]

# Okabe-Ito colors with distinct line styles, so curves don't rely on color alone.
STYLES = [
    ("#0072B2", "-"),
    ("#D55E00", "--"),
    ("#009E73", "-."),
    ("#CC79A7", ":"),
    ("#E69F00", (0, (5, 1, 1, 1))),
    ("#56B4E9", (0, (3, 1))),
]


def finished_models():
    """Models with both saved CV files. Prints the ones skipped."""
    done, skipped = [], []
    for folder, label in MODELS:
        path = common.RESULTS_DIR / folder
        files = [path / "cv_scores.csv", path / "oof_predictions.parquet"]
        (done if all(f.exists() for f in files) else skipped).append((folder, label))
    for folder, _ in skipped:
        print(f"Skipped {folder}: no cv_scores.csv and oof_predictions.parquet yet")
    return done


def training_labels():
    """The training set's `y`, indexed by row ID. The recalibration and test sets are
    split off and discarded unused."""
    sample = common.draw_sample(common.build_cohort(common.load_raw()))
    train, _, _ = common.split(sample)
    common.check_threshold(train)
    return train["y"]


def check_oof(folder, oof, y, scores):
    """The saved out-of-fold risks are this training set's, and they reproduce the
    saved per-fold AUC and Brier."""
    common.check_rows(oof, y.to_frame())
    assert sorted(oof["fold"].unique()) == [1, 2, 3, 4, 5], folder
    for row in scores.itertuples():
        in_fold = oof["fold"] == row.fold
        auc = roc_auc_score(y[in_fold], oof.loc[in_fold, "risk"])
        brier = np.mean((y[in_fold] - oof.loc[in_fold, "risk"]) ** 2)
        assert abs(auc - row.auc) < 1e-9 and abs(brier - row.brier) < 1e-9, folder


def fmt(mean, sd):
    return f"${mean:.4f} \\pm {sd:.4f}$"


def write_table(results, path):
    """A booktabs table of mean ± SD across folds."""
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & AUC $\uparrow$ & Brier $\downarrow$ & Precision & Recall \\",
        r"\midrule",
    ]
    for row in results.itertuples():
        cells = [fmt(getattr(row, s), getattr(row, f"{s}_sd")) for s in common.RESULT_SCORES]
        lines.append(f"{row.model} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def plot_roc(curves, path):
    """One ROC curve per model from its pooled out-of-fold risks, with the mean fold AUC
    in the legend to match the table."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.8, linestyle=":", label="Chance")
    for (label, fpr, tpr, auc, auc_sd), (color, style) in zip(curves, STYLES):
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=style,
            linewidth=1.4,
            label=f"{label}\n  AUC {auc:.3f} ± {auc_sd:.3f}",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(color="0.9", linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1, handlelength=2.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    models = finished_models()
    y = training_labels()

    results, curves = [], []
    for folder, label in models:
        path = common.RESULTS_DIR / folder
        scores = pd.read_csv(path / "cv_scores.csv")
        oof = pd.read_parquet(path / "oof_predictions.parquet")
        check_oof(folder, oof, y, scores)

        row = common.cv_results(label, scores)
        results.append(row)
        fpr, tpr, _ = roc_curve(y, oof["risk"])
        curves.append((label, fpr, tpr, row["auc"].iloc[0], row["auc_sd"].iloc[0]))
        pooled = roc_auc_score(y, oof["risk"])
        print(f"{folder}: pooled out-of-fold AUC {pooled:.4f}")

    results = pd.concat(results, ignore_index=True)
    common.show_scores(results)

    OUT_DIR.mkdir(exist_ok=True)
    write_table(results, OUT_DIR / "tab_cv_metrics.tex")
    plot_roc(curves, OUT_DIR / "roc_oof.pdf")
    print(f"Wrote {OUT_DIR / 'tab_cv_metrics.tex'} and {OUT_DIR / 'roc_oof.pdf'}")


if __name__ == "__main__":
    main()
