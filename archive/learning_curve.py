"""
learning_curve.py — does more data help, or is the limit the features?

Subsamples participants at several fractions, runs the same leak-free nested-CV
evaluation, and plots MAE vs sample size with the mean-predictor line.

Interpretation:
  - Flat line at ~mean-predictor MAE across all fractions -> the limit is the
    FEATURES (more data won't help; richer features needed = Phase 2).
  - Line still falling at 100% -> a SAMPLE-SIZE story (more data might help).

Run AFTER build_dataset.py:
    python learning_curve.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import daic_woz_pipeline.src.config as config
from daic_woz_pipeline.src.train_regression import (load_combined, feature_columns,
                               nested_cv_evaluate, mean_predictor_baseline)

FIG_DIR = config.OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
REPS = 10              # repeats per fraction (small subsamples are noisy)
MODEL = "svr"          # single representative model keeps runtime sane


def main():
    df = load_combined()
    audio, visual = feature_columns(df)
    cols = audio + visual
    X = df[cols].values
    y = df[config.SCORE_COL].values.astype(float)
    rng = np.random.default_rng(config.RANDOM_STATE)

    mean_mae = mean_predictor_baseline(X, y)["mae_mean"]

    means, stds = [], []
    for f in FRACTIONS:
        size = max(20, int(f * len(df)))  # floor so folds remain viable
        reps = []
        for _ in range(REPS):
            idx = rng.choice(len(df), size=size, replace=False)
            res = nested_cv_evaluate(X[idx], y[idx], None, MODEL)
            reps.append(res["mae_mean"])
        means.append(np.mean(reps)); stds.append(np.std(reps))
        print(f"  fraction={f:.2f} (n={size}): "
              f"MAE={np.mean(reps):.3f} +/- {np.std(reps):.3f}")

    sizes = [max(20, int(f * len(df))) for f in FRACTIONS]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4,
                color="#4C72B0", label=f"{MODEL.upper()} (audio+visual)")
    ax.axhline(mean_mae, color="#C44E52", ls="--",
               label=f"mean predictor ({mean_mae:.2f})")
    ax.set_xlabel("Number of participants (subsampled)")
    ax.set_ylabel("MAE (nested CV)")
    ax.set_title("Learning curve: does more data help?\n"
                 "flat at mean-predictor line = feature ceiling")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "learning_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    print("Read: flat near the red line across sizes => limit is the features, "
          "not the sample size (supports the Phase-2 richer-features argument).")


if __name__ == "__main__":
    main()