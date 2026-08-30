"""
plot_text_results.py — figures for the Phase 2 text modality.

Reads text_results.csv (and text_results_pca.csv if present) and the Phase 1
baseline numbers, and produces comparison figures. It plots WHATEVER models are
in the CSV, so it works whether you ran 2 or 3 encoders, with or without PCA.

    python3 plot_text_results.py

Figures saved to OUT_DIR/figures_text/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import daic_woz_pipeline.src.config as config

FIGDIR = config.OUT_DIR / "figures_text"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Phase 1 baseline reference points (from your verified baseline results)
BASELINE_BEST_MAE = 4.76     # best audio/visual config
BASELINE_R2 = -0.06          # negative (worse than mean) — adjust to your exact
MEAN_PRED_MAE = 4.758


def load(path):
    p = config.OUT_DIR / path
    return pd.read_csv(p) if p.exists() else None


def label(row):
    return f"{row['text_model']}-{row['model']}"


def plot_mae(df):
    d = df[df["model"] != "mean_predictor"].copy()
    names = [label(r) for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, d["mae"], yerr=d.get("mae_std"),
                  capsize=3, color="#4C72B0")
    ax.axhline(MEAN_PRED_MAE, ls="--", color="crimson",
               label=f"mean predictor ({MEAN_PRED_MAE:.2f})")
    ax.axhline(BASELINE_BEST_MAE, ls=":", color="gray",
               label=f"baseline best audio/visual ({BASELINE_BEST_MAE})")
    ax.axhline(4.0, ls="-.", color="green", alpha=0.6, label="target MAE = 4.0")
    ax.set_ylabel("MAE (lower = better)")
    ax.set_title("Text modality: MAE vs mean predictor and baseline")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIGDIR / "text_mae.png", dpi=150); plt.close()


def plot_r2(df):
    d = df[df["model"] != "mean_predictor"].copy()
    if "r2_pooled" not in d.columns:
        return
    names = [label(r) for _, r in d.iterrows()]
    colors = ["#55A868" if v > 0 else "#C44E52" for v in d["r2_pooled"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, d["r2_pooled"], color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.axhline(BASELINE_R2, ls=":", color="gray",
               label=f"baseline R² ({BASELINE_R2}, negative)")
    ax.set_ylabel("R² (pooled)  — positive = explains real variance")
    ax.set_title("Text modality: R² (text positive vs baseline negative)")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIGDIR / "text_r2.png", dpi=150); plt.close()


def plot_f1(df):
    d = df[df["model"] != "mean_predictor"].copy()
    if "f1_pooled" not in d.columns:
        return
    names = [label(r) for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, d["f1_pooled"], color="#8172B3")
    ax.axhline(0.0, color="gray", ls=":", label="baseline F1 ≈ 0")
    ax.set_ylabel("F1 (pooled, PHQ-8 ≥ 10)")
    ax.set_title("Text modality: depression-detection F1")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIGDIR / "text_f1.png", dpi=150); plt.close()


def plot_phase_compare(df):
    """Headline: baseline vs best text, MAE and R² side by side."""
    d = df[df["model"] != "mean_predictor"].copy()
    best = d.loc[d["mae"].idxmin()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar(["baseline\n(audio/visual)", f"text\n({label(best)})"],
           [BASELINE_BEST_MAE, best["mae"]], color=["gray", "#4C72B0"])
    a1.axhline(MEAN_PRED_MAE, ls="--", color="crimson", label="mean predictor")
    a1.axhline(4.0, ls="-.", color="green", alpha=0.6, label="target")
    a1.set_ylabel("MAE"); a1.set_title("Best MAE"); a1.legend(fontsize=7)
    if "r2_pooled" in d.columns:
        a2.bar(["baseline\n(audio/visual)", f"text\n({label(best)})"],
               [BASELINE_R2, best["r2_pooled"]],
               color=["#C44E52", "#55A868"])
        a2.axhline(0, color="black", lw=1)
        a2.set_ylabel("R² (pooled)"); a2.set_title("Best R²")
    plt.suptitle("Phase 1 baseline vs Phase 2 text")
    plt.tight_layout(); plt.savefig(FIGDIR / "phase1_vs_phase2.png", dpi=150)
    plt.close()


def plot_pca_compare(df, dfp):
    """If PCA results exist, compare MAE with vs without PCA per config."""
    if dfp is None:
        return
    d = df[df["model"] != "mean_predictor"].copy()
    dp = dfp[dfp["model"] != "mean_predictor"].copy()
    d["key"] = d.apply(label, axis=1); dp["key"] = dp.apply(label, axis=1)
    merged = d.merge(dp, on="key", suffixes=("_nopca", "_pca"))
    x = np.arange(len(merged)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, merged["mae_nopca"], w, label="no PCA", color="#4C72B0")
    ax.bar(x + w/2, merged["mae_pca"], w, label="PCA", color="#DD8452")
    ax.axhline(MEAN_PRED_MAE, ls="--", color="crimson")
    ax.set_xticks(x); ax.set_xticklabels(merged["key"], rotation=45, ha="right")
    ax.set_ylabel("MAE"); ax.set_title("Effect of PCA-inside-folds on MAE")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGDIR / "pca_compare.png", dpi=150); plt.close()


def main():
    df = load("text_results.csv")
    if df is None:
        print("text_results.csv not found. Run run_text_regression.py first.")
        return
    dfp = load("text_results_pca.csv")

    plot_mae(df)
    plot_r2(df)
    plot_f1(df)
    plot_phase_compare(df)
    plot_pca_compare(df, dfp)
    plot_overfit(df)
    plot_severe(df)
    plot_encoder_summary(df)

    made = sorted(p.name for p in FIGDIR.glob("*.png"))
    print(f"Saved {len(made)} figures to {FIGDIR}:")
    for m in made:
        print("  ", m)




# ---- additional figures for a complete analysis -------------------------

def plot_overfit(df):
    d = df[df["model"] != "mean_predictor"].copy()
    if "overfit_gap" not in d.columns:
        return
    names = [label(r) for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, d["overfit_gap"], color="#CCB974")
    ax.set_ylabel("Overfit gap (test MAE − train MAE)")
    ax.set_title("Text modality: overfitting gap per configuration")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIGDIR / "text_overfit_gap.png", dpi=150); plt.close()


def plot_severe(df):
    d = df[df["model"] != "mean_predictor"].copy()
    if "mae_severe" not in d.columns:
        return
    names = [label(r) for _, r in d.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, d["mae_severe"], color="#C44E52")
    ax.axhline(11.08, ls="--", color="gray",
               label="baseline severe MAE (11.08, mean-reversion)")
    ax.set_ylabel("MAE on severe cases (PHQ-8 15–24)")
    ax.set_title("Text modality: severe-case MAE vs baseline")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(FIGDIR / "text_severe_mae.png", dpi=150); plt.close()


def plot_encoder_summary(df):
    """Best MAE and R^2 per encoder — makes the encoder ranking explicit."""
    d = df[df["model"] != "mean_predictor"].copy()
    if "text_model" not in d.columns:
        return
    rows = []
    for enc, g in d.groupby("text_model"):
        best = g.loc[g["mae"].idxmin()]
        rows.append((enc, best["mae"], best.get("r2_pooled", float("nan"))))
    rows.sort(key=lambda r: r[1])
    encs = [r[0] for r in rows]
    maes = [r[1] for r in rows]
    r2s = [r[2] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar(encs, maes, color="#4C72B0")
    a1.axhline(MEAN_PRED_MAE, ls="--", color="crimson", label="mean predictor")
    a1.axhline(4.0, ls="-.", color="green", alpha=0.6, label="target")
    a1.set_ylabel("Best MAE"); a1.set_title("Best MAE per encoder")
    a1.legend(fontsize=7)
    a2.bar(encs, r2s, color="#55A868")
    a2.axhline(0, color="black", lw=1)
    a2.set_ylabel("R² (pooled)"); a2.set_title("Best R² per encoder")
    plt.suptitle("Encoder comparison (best config each)")
    plt.tight_layout(); plt.savefig(FIGDIR / "encoder_summary.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()