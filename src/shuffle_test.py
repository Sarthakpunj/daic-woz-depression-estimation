"""
shuffle_test.py — label-shuffle smoke test (a.k.a. permutation sanity check).

Idea: if we randomly SHUFFLE the PHQ-8 labels, any real feature->target
relationship is destroyed. A correct, leakage-free pipeline must then perform
NO BETTER than the mean predictor (R^2 ~ 0, MAE ~ the mean-predictor MAE).

  * If shuffled-label performance stays near the mean predictor  -> PIPELINE CLEAN.
    The real (unshuffled) result therefore reflects genuine signal, not leakage
    or a bug.
  * If shuffled-label performance is still "good" (positive R^2, low MAE)
    -> RED FLAG: leakage or a bug is letting the model cheat.

We run this on whatever feature set you point it at (text embeddings or the
audio/visual baseline parquet), comparing:
    - REAL labels   (should beat the mean -> shows signal exists)
    - SHUFFLED labels x N repeats (should collapse to the mean -> proves clean)

Usage:
    python3 shuffle_test.py text_features_mpnet.parquet
    python3 shuffle_test.py train_features.parquet dev_features.parquet
(the second form concatenates a baseline audio/visual feature set)
"""

import sys
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

import daic_woz_pipeline.src.config as config

N_SHUFFLES = 10
N_OUTER, N_INNER = 5, 3
RNG = np.random.RandomState(config.RANDOM_STATE)


def evaluate_svr(X, y):
    """One nested-CV pass with SVR; returns pooled MAE and pooled R^2."""
    outer = KFold(n_splits=N_OUTER, shuffle=True,
                  random_state=config.RANDOM_STATE)
    all_true, all_pred = [], []
    for tr, te in outer.split(X):
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", SVR())])
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=config.RANDOM_STATE)
        gs = GridSearchCV(pipe, {"clf__C": [1.0, 10.0]},
                          scoring="neg_mean_absolute_error", cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        pred = gs.best_estimator_.predict(X[te])
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())
    at, ap = np.array(all_true), np.array(all_pred)
    return mean_absolute_error(at, ap), r2_score(at, ap)


def mean_predictor(X, y):
    outer = KFold(n_splits=N_OUTER, shuffle=True,
                  random_state=config.RANDOM_STATE)
    at, ap = [], []
    for tr, te in outer.split(X):
        pred = np.full(len(te), y[tr].mean())
        at.extend(y[te].tolist()); ap.extend(pred.tolist())
    at, ap = np.array(at), np.array(ap)
    return mean_absolute_error(at, ap), r2_score(at, ap)


def load_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        df = pd.read_csv(config.DATA_ROOT / fn)
        df.columns = [c.strip() for c in df.columns]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]


def load_features(paths):
    frames = [pd.read_parquet(p) for p in paths]
    feats = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # If the parquet already carries the PHQ-8 score (baseline format), use it
    # directly. Otherwise (text-embedding format), merge labels from the splits.
    if config.SCORE_COL in feats.columns:
        df = feats
    else:
        df = feats.merge(load_labels(), on=config.ID_COL, how="inner")

    # feature columns = numeric, excluding ID, gender, and any label columns
    gender_col = getattr(config, "GENDER_COL", "Gender")
    exclude = {config.ID_COL, gender_col, config.SCORE_COL, "PHQ8_Binary"}
    cols = [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    X = df[cols].values.astype(float)
    y = df[config.SCORE_COL].values.astype(float)
    return X, y, len(cols)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python3 shuffle_test.py <features.parquet> [more.parquet]")
        print("Looking in OUT_DIR for text_features_*.parquet ...")
        args = sorted(glob.glob(str(config.OUT_DIR / "text_features_*.parquet")))
        if not args:
            return
        args = [args[0]]  # default: first text feature set

    paths = []
    for a in args:
        p = a if "/" in a else str(config.OUT_DIR / a)
        paths.append(p)
    print(f"Feature file(s): {[p.split('/')[-1] for p in paths]}")

    X, y, nfeat = load_features(paths)
    print(f"n={len(y)} participants, {nfeat} features\n")

    mp_mae, mp_r2 = mean_predictor(X, y)
    print(f"Mean predictor:        MAE={mp_mae:.3f}  R2={mp_r2:.3f}")

    real_mae, real_r2 = evaluate_svr(X, y)
    print(f"REAL labels (SVR):     MAE={real_mae:.3f}  R2={real_r2:.3f}  "
          f"<- should BEAT mean (shows signal)\n")

    print(f"Shuffled labels x{N_SHUFFLES} (should COLLAPSE to mean):")
    sh_maes, sh_r2s = [], []
    for i in range(N_SHUFFLES):
        y_shuf = y.copy()
        RNG.shuffle(y_shuf)
        m, r = evaluate_svr(X, y_shuf)
        sh_maes.append(m); sh_r2s.append(r)
        print(f"  shuffle {i+1:2d}: MAE={m:.3f}  R2={r:.3f}")

    sh_mae = np.mean(sh_maes); sh_r2 = np.mean(sh_r2s)
    print(f"\n  shuffled mean:       MAE={sh_mae:.3f}  R2={sh_r2:.3f}")

    print("\n=== VERDICT ===")
    print(f"Real R2          : {real_r2:+.3f}")
    print(f"Shuffled R2 (avg): {sh_r2:+.3f}  (should be ~0 or negative)")
    print(f"Mean-pred R2     : {mp_r2:+.3f}")
    clean = sh_r2 < 0.05 and abs(sh_mae - mp_mae) < 0.5
    signal = real_r2 > sh_r2 + 0.05
    if clean:
        print("PASS: shuffled labels collapse to ~mean-predictor performance.")
        print("      -> No leakage/bug: the pipeline only finds REAL signal.")
    else:
        print("WARNING: shuffled labels still perform well -> possible leakage/bug.")
    if signal:
        print("CONFIRMED: real labels beat shuffled -> genuine signal in features.")
    else:
        print("NOTE: real labels do NOT beat shuffled -> features carry little/no "
              "signal (expected for the audio/visual baseline; not expected for text).")


if __name__ == "__main__":
    main()