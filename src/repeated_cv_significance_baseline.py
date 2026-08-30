"""
repeated_cv_significance_baseline.py — protocol-symmetric significance test for
the BASELINE, matching the Nadeau-Bengio repeated-CV test now used in Phase 2.

Applies the same 5x5 repeated CV + corrected resampled t-test to the baseline's
best configuration (SVR on audio features), so both phases use identical
significance machinery. Expected to confirm the null (non-significant) — which
strengthens it: "non-significant even under a higher-powered 25-fold repeated
test" is stronger than the single 5-fold Wilcoxon p = 0.81.

Run:
    python3 repeated_cv_significance_baseline.py train_features.parquet dev_features.parquet
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from scipy import stats

import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

N_REPEATS = 5
N_FOLDS = 5


def corrected_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test.
       diffs: per-fold (mean_predictor_MAE - model_MAE); positive => model better."""
    k = len(diffs)
    mean = np.mean(diffs); var = np.var(diffs, ddof=1)
    if var == 0:
        return np.inf, 0.0
    t = mean / np.sqrt(var * (1.0/k + n_test/n_train))
    p = 2 * stats.t.sf(abs(t), k - 1)
    return t, p


def load(paths):
    frames = [pd.read_parquet(p if "/" in p else str(config.OUT_DIR / p)) for p in paths]
    feats = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if config.SCORE_COL in feats.columns:
        df = feats
    else:
        rows = []
        for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
            d = pd.read_csv(config.DATA_ROOT / fn); d.columns=[c.strip() for c in d.columns]
            rows.append(d)
        lab = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
        df = feats.merge(lab, on=config.ID_COL, how="inner")
    gcol = getattr(config, "GENDER_COL", "Gender")
    excl = {config.ID_COL, gcol, config.SCORE_COL, "PHQ8_Binary"}
    # baseline "best config" = SVR on AUDIO features. Select audio columns if named,
    # else use all non-label features (still SVR, still the baseline pipeline).
    audio_cols = [c for c in df.columns if c.startswith(("covarep_", "formant_"))]
    if audio_cols:
        cols = audio_cols
        note = f"audio features only ({len(cols)})"
    else:
        cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
        note = f"all features ({len(cols)}) — audio-prefixed columns not found"
    return df[cols].values.astype(float), df[config.SCORE_COL].values.astype(float), note


def main():
    paths = sys.argv[1:] or ["train_features.parquet", "dev_features.parquet"]
    X, y, note = load(paths)
    print(f"Baseline SVR significance — {note}")
    print(f"n={len(y)}; repeated CV {N_REPEATS}x{N_FOLDS} = {N_REPEATS*N_FOLDS} folds\n")

    diffs, model_maes, mean_maes = [], [], []
    for rep in range(N_REPEATS):
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=rep).split(X):
            pipe = Pipeline([("imp",SimpleImputer(strategy="median")),
                             ("sc",StandardScaler()),("clf",SVR())])
            gs = GridSearchCV(pipe, {"clf__C":[1.0,10.0]},
                              scoring="neg_mean_absolute_error",
                              cv=KFold(3,shuffle=True,random_state=config.RANDOM_STATE), n_jobs=-1)
            gs.fit(X[tr], y[tr])
            mm = mean_absolute_error(y[te], gs.best_estimator_.predict(X[te]))
            pm = mean_absolute_error(y[te], np.full(len(te), y[tr].mean()))
            diffs.append(pm - mm); model_maes.append(mm); mean_maes.append(pm)

    diffs = np.array(diffs)
    n = len(y); n_test = n // N_FOLDS; n_train = n - n_test
    t, p = corrected_ttest(diffs, n_train, n_test)
    print(f"Model  MAE (mean over folds): {np.mean(model_maes):.3f}")
    print(f"Mean-predictor MAE:          {np.mean(mean_maes):.3f}")
    print(f"Mean difference (pred-model): {np.mean(diffs):+.3f}")
    print(f"Folds where model beats mean: {(diffs>0).sum()}/{len(diffs)}")
    print(f"\nNadeau-Bengio corrected t-test: t={t:.3f}, p={p:.4f}")
    if p < 0.05:
        print("  -> SIGNIFICANT (unexpected for the baseline; investigate).")
    else:
        print("  -> NOT significant: the baseline does not beat the mean predictor")
        print("     even under the higher-powered 25-fold repeated test. This")
        print("     strengthens the null and matches the Phase 2 protocol.")


if __name__ == "__main__":
    main()