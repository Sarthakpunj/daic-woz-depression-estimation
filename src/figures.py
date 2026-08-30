"""
figures.py — generate dissertation visuals for the Phase 1 baseline.

Produces (into outputs/figures/):
  1. phq8_distribution.png      - PHQ-8 score histogram (shows label skew)
  2. class_balance.png          - binary depressed/not balance, train vs dev
  3. mae_vs_meanpredictor.png   - each model's MAE vs the mean-predictor line
  4. modality_comparison.png    - audio vs visual vs fusion (tests RQ1)
  5. mae_by_score_band.png      - MAE across score bands (severe-case failure)
  6. plain_vs_weighted.png      - skew-mitigation effect on severe-case MAE
  7. overfit_gap.png            - train vs test MAE per model (overfitting)
  8. predicted_vs_actual.png    - regression scatter (models cluster at mean)

Run AFTER train_regression.py and build_dataset.py:
    python figures.py
Reads outputs/regression_results.csv, outputs/regression_metrics.json,
and the feature parquet files. Degrades gracefully if something's missing.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict

import config

FIG_DIR = config.OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# consistent colours
C_NOT = "#4C72B0"
C_DEP = "#C44E52"
C_MEAN = "#555555"
C_BARS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]


def _load_results():
    path = config.OUT_DIR / "regression_results.csv"
    if not path.exists():
        print(f"  (skip) {path} not found — run train_regression.py first.")
        return None
    return pd.read_csv(path)


def _load_features():
    tr = config.OUT_DIR / "train_features.parquet"
    dv = config.OUT_DIR / "dev_features.parquet"
    if not tr.exists() or not dv.exists():
        return None
    return pd.concat([pd.read_parquet(tr), pd.read_parquet(dv)],
                     ignore_index=True)


def fig_phq8_distribution(df):
    if df is None or config.SCORE_COL not in df.columns:
        print("  (skip) PHQ-8 distribution: no score data"); return
    scores = df[config.SCORE_COL].values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=range(0, 26), color=C_DEP, edgecolor="white", alpha=0.85)
    ax.axvline(10, color="black", ls="--", lw=1.2, label="PHQ-8 = 10 (depression cut-off)")
    ax.axvline(scores.mean(), color=C_MEAN, ls=":", lw=1.5,
               label=f"mean = {scores.mean():.1f}")
    ax.set_xlabel("PHQ-8 score"); ax.set_ylabel("Number of participants")
    ax.set_title("PHQ-8 score distribution (right-skewed: few severe cases)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "phq8_distribution.png", dpi=150)
    plt.close(fig); print("  saved phq8_distribution.png")


def fig_class_balance(df):
    if df is None or config.LABEL_COL not in df.columns:
        print("  (skip) class balance"); return
    counts = df[config.LABEL_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Not depressed", "Depressed"], counts.values, color=[C_NOT, C_DEP])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    ax.set_ylabel("Participants")
    ax.set_title(f"Class balance (n={len(df)}, "
                 f"{100*counts.get(1,0)/len(df):.0f}% depressed)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "class_balance.png", dpi=150)
    plt.close(fig); print("  saved class_balance.png")


def fig_mae_vs_mean(res):
    if res is None: return
    plain = res[res.get("weighting", "plain").eq("plain")] if "weighting" in res else res
    models = plain[plain["model"] != "mean_predictor"].copy()
    mean_row = res[res["model"] == "mean_predictor"]
    mean_mae = mean_row["mae"].iloc[0] if len(mean_row) else None
    labels = models["model"] + "\n" + models["features"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(models)), models["mae"], color=C_NOT)
    if mean_mae:
        ax.axhline(mean_mae, color=C_DEP, ls="--", lw=1.5,
                   label=f"mean predictor (MAE={mean_mae:.2f})")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("MAE (lower = better)")
    ax.set_title("Model MAE vs mean-predictor baseline (plain)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "mae_vs_meanpredictor.png", dpi=150)
    plt.close(fig); print("  saved mae_vs_meanpredictor.png")


def fig_modality(res):
    if res is None: return
    plain = res[res.get("weighting", "plain").eq("plain")] if "weighting" in res else res
    plain = plain[plain["model"] != "mean_predictor"]
    pivot = plain.pivot_table(index="model", columns="features", values="mae")
    order = ["audio", "visual", "fusion"]
    pivot = pivot[[c for c in order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot.plot(kind="bar", ax=ax, color=C_BARS[:len(pivot.columns)])
    ax.set_ylabel("MAE"); ax.set_xlabel("")
    ax.set_title("MAE by modality (does fusion beat single modality? — RQ1)")
    ax.legend(title="features"); plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "modality_comparison.png", dpi=150)
    plt.close(fig); print("  saved modality_comparison.png")


def fig_score_band(metrics):
    if metrics is None: return
    # find any model's per-band breakdown (use rf fusion plain if present)
    key = None
    for cand in ["rf__fusion__plain", "rf__audio__plain", "rf__fusion"]:
        if cand in metrics: key = cand; break
    if key is None:
        print("  (skip) score band: no per-band data"); return
    bands = metrics[key].get("mae_by_score_band", {})
    if not bands:
        print("  (skip) score band: empty"); return
    labels = list(bands.keys())
    maes = [bands[b]["mae"] for b in labels]
    ns = [bands[b]["n"] for b in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, maes, color=C_DEP)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"n={n}", ha="center", fontsize=8)
    ax.set_ylabel("MAE"); ax.set_title(f"MAE by PHQ-8 score band ({key})\n"
                                       "high MAE on severe cases = skew failure")
    plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(FIG_DIR / "mae_by_score_band.png", dpi=150)
    plt.close(fig); print("  saved mae_by_score_band.png")


def fig_plain_vs_weighted(res):
    if res is None or "weighting" not in res.columns or "mae_severe" not in res.columns:
        print("  (skip) plain vs weighted: need weighting + mae_severe columns"); return
    sub = res[res["model"] != "mean_predictor"].copy()
    sub["cfg"] = sub["model"] + "-" + sub["features"]
    piv = sub.pivot_table(index="cfg", columns="weighting", values="mae_severe")
    if "plain" not in piv or "weighted" not in piv:
        print("  (skip) plain vs weighted: missing a condition"); return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w/2, piv["plain"], w, label="plain", color=C_NOT)
    ax.bar(x + w/2, piv["weighted"], w, label="weighted", color=C_DEP)
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=30, fontsize=8, ha="right")
    ax.set_ylabel("Severe-case MAE (PHQ-8 15–24)")
    ax.set_title("Skew mitigation: severe-case MAE, plain vs weighted")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "plain_vs_weighted.png", dpi=150)
    plt.close(fig); print("  saved plain_vs_weighted.png")


def fig_overfit(res):
    if res is None or "train_mae" not in res.columns:
        print("  (skip) overfit: no train_mae column"); return
    plain = res[res.get("weighting", "plain").eq("plain")] if "weighting" in res else res
    plain = plain[plain["model"] != "mean_predictor"].copy()
    plain["cfg"] = plain["model"] + "-" + plain["features"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(plain)); w = 0.38
    ax.bar(x - w/2, plain["train_mae"], w, label="train MAE", color=C_NOT)
    ax.bar(x + w/2, plain["mae"], w, label="test MAE (CV)", color=C_DEP)
    ax.set_xticks(x); ax.set_xticklabels(plain["cfg"], rotation=30, fontsize=8, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("Train vs test MAE — large gap = overfitting")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "overfit_gap.png", dpi=150)
    plt.close(fig); print("  saved overfit_gap.png")


def fig_predicted_vs_actual(df):
    """Refit one model via CV-predict to show the scatter. Illustrative only —
    uses the same leak-free CV protocol for the predictions."""
    if df is None or config.SCORE_COL not in df.columns:
        print("  (skip) pred-vs-actual: no data"); return
    audio = [c for c in df.columns if c.startswith(("covarep_", "formant_"))]
    visual = [c for c in df.columns if c.startswith(("gaze_", "pose_", "au_"))]
    cols = audio + visual
    X = df[cols].values; y = df[config.SCORE_COL].values.astype(float)
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", RandomForestRegressor(n_estimators=300,
                                                   random_state=config.RANDOM_STATE,
                                                   n_jobs=-1))])
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=cv)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, pred, alpha=0.6, color=C_NOT, edgecolor="white")
    lim = [0, 24]
    ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
    ax.axhline(y.mean(), color=C_DEP, ls=":", lw=1.5,
               label=f"mean predictor ({y.mean():.1f})")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual PHQ-8"); ax.set_ylabel("Predicted PHQ-8")
    ax.set_title("Predicted vs actual (RF fusion, CV)\n"
                 "predictions cluster near the mean")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close(fig); print("  saved predicted_vs_actual.png")


def main():
    print(f"Writing figures to {FIG_DIR}")
    res = _load_results()
    df = _load_features()
    metrics = None
    mpath = config.OUT_DIR / "regression_metrics.json"
    if mpath.exists():
        metrics = json.load(open(mpath))

    fig_phq8_distribution(df)
    fig_class_balance(df)
    fig_mae_vs_mean(res)
    fig_modality(res)
    fig_score_band(metrics)
    fig_plain_vs_weighted(res)
    fig_overfit(res)
    fig_predicted_vs_actual(df)
    print("Done.")


if __name__ == "__main__":
    main()