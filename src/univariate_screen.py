"""
univariate_screen.py — descriptive check of how much signal the features carry.

For each feature, computes its Spearman correlation with the PHQ-8 score, then
applies Benjamini-Hochberg FDR correction across all features. Reports the
strongest correlation and how many features survive correction.

Interpretation:
  - Tiny max |rho| and few/zero features surviving FDR -> quantitative evidence
    that the classical functionals carry little PHQ-8 signal, explaining why no
    model beat the mean predictor.

IMPORTANT: this is a DESCRIPTIVE analysis for the write-up only. Do NOT use it
to select features for the models — feature selection must happen inside the CV
folds, or it creates the exact leakage this project critiques.

Run AFTER build_dataset.py:
    python univariate_screen.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import daic_woz_pipeline.src.config as config
from daic_woz_pipeline.src.train_regression import load_combined, feature_columns

FIG_DIR = config.OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    df = load_combined()
    audio, visual = feature_columns(df)
    cols = audio + visual
    y = df[config.SCORE_COL].values.astype(float)

    rho, pvals, names = [], [], []
    for c in cols:
        v = df[c].values.astype(float)
        m = ~np.isnan(v)
        if m.sum() < 10 or np.nanstd(v[m]) == 0:
            continue  # skip empty/constant columns
        r, p = spearmanr(v[m], y[m])
        if np.isnan(r):
            continue
        rho.append(r); pvals.append(p); names.append(c)

    rho = np.array(rho); pvals = np.array(pvals)

    # Benjamini-Hochberg FDR. Use statsmodels if available, else a manual BH.
    try:
        from statsmodels.stats.multitest import multipletests
        rej, p_adj, *_ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    except Exception:
        order = np.argsort(pvals)
        n = len(pvals)
        p_adj = np.empty(n)
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            k = n - rank
            prev = min(prev, pvals[i] * n / k)
            p_adj[i] = prev
        rej = p_adj < 0.05

    n_sig = int(rej.sum())
    abs_rho = np.abs(rho)
    top = np.argsort(abs_rho)[::-1][:10]

    print(f"Features tested: {len(names)}")
    print(f"Max |Spearman rho| with PHQ-8: {abs_rho.max():.3f} "
          f"(feature: {names[int(abs_rho.argmax())]})")
    print(f"Features surviving FDR (q<0.05): {n_sig} / {len(names)}")
    print("\nTop 10 by |rho|:")
    for i in top:
        flag = "*" if rej[i] else " "
        print(f"  {flag} {names[i]:<32} rho={rho[i]:+.3f}  p_adj={p_adj[i]:.3f}")

    # histogram of |rho|
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(abs_rho, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(abs_rho.max(), color="#C44E52", ls="--",
               label=f"max |rho| = {abs_rho.max():.3f}")
    ax.set_xlabel("|Spearman correlation with PHQ-8|")
    ax.set_ylabel("Number of features")
    ax.set_title(f"Univariate feature–PHQ-8 association\n"
                 f"{n_sig}/{len(names)} survive FDR (q<0.05)")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "univariate_screen.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    print("Descriptive only — do NOT select features with this outside CV folds.")


if __name__ == "__main__":
    main()