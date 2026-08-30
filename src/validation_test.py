"""
validation_tests.py — validation / robustness tests for the text modality.

Runs several independent checks, each as its own clearly-labelled function so the
results can be cited separately in the report. Point it at a text-feature parquet.

    python3 validation_tests.py text_features_mpnet.parquet

Tests included:
  1. permutation_test   — shuffle labels; clean pipeline must collapse to mean.
  2. significance_test  — paired Wilcoxon: are per-fold errors of the model vs
                          the mean predictor significantly different?
  3. ablation_test      — embeddings-only vs embeddings+linguistic features.
  4. seed_stability     — re-run across several CV seeds; is the result robust?
  5. heldout_eval       — train on all train+dev, evaluate ONCE on the held-out
                          test set. REQUIRES the test-set features + labels
                          (not available until the 47 test participants are
                          downloaded); skipped automatically if absent.

Notes:
  - Uses SVR for the permutation/significance/seed tests (fast, representative).
  - All preprocessing is fitted inside folds; no leakage.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import wilcoxon

import daic_woz_pipeline.src.config as config

N_OUTER, N_INNER = 5, 3
N_SHUFFLES = 100


# ---------- shared helpers --------------------------------------------------

def _pipe(est):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", est)])


def _est(name):
    if name == "svr":
        return SVR(), {"clf__C": [1.0, 10.0]}
    if name == "ridge":
        return Ridge(random_state=config.RANDOM_STATE), {"clf__alpha": [1.0, 10.0, 100.0]}
    raise ValueError(name)


def _per_fold_errors(X, y, model="svr", seed=config.RANDOM_STATE):
    """Return per-fold mean-abs-errors for the model and the mean predictor,
    plus pooled MAE and pooled R^2 for the model."""
    outer = KFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    fold_model, fold_mean = [], []
    at, ap = [], []
    for tr, te in outer.split(X):
        est, grid = _est(model)
        inner = KFold(n_splits=N_INNER, shuffle=True, random_state=seed)
        gs = GridSearchCV(_pipe(est), grid, scoring="neg_mean_absolute_error",
                          cv=inner, n_jobs=-1)
        gs.fit(X[tr], y[tr])
        pred = gs.best_estimator_.predict(X[te])
        mpred = np.full(len(te), y[tr].mean())
        fold_model.append(mean_absolute_error(y[te], pred))
        fold_mean.append(mean_absolute_error(y[te], mpred))
        at.extend(y[te]); ap.extend(pred)
    at, ap = np.array(at), np.array(ap)
    return (np.array(fold_model), np.array(fold_mean),
            mean_absolute_error(at, ap), r2_score(at, ap))


def load_features(paths, want_linguistic=True):
    frames = [pd.read_parquet(p) for p in paths]
    feats = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if config.SCORE_COL in feats.columns:
        df = feats
    else:
        rows = []
        for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
            d = pd.read_csv(config.DATA_ROOT / fn)
            d.columns = [c.strip() for c in d.columns]
            rows.append(d)
        labels = pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]
        df = feats.merge(labels, on=config.ID_COL, how="inner")
    gender_col = getattr(config, "GENDER_COL", "Gender")
    label_cols = {config.ID_COL, gender_col, config.SCORE_COL, "PHQ8_Binary"}
    emb = [c for c in df.columns if c.startswith("emb_")]
    ling = [c for c in df.columns if c.startswith("ling_")]
    if not emb:  # baseline-format: everything numeric that isn't a label
        emb = [c for c in df.columns if c not in label_cols
               and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("ling_")]
    cols = emb + (ling if want_linguistic else [])
    X = df[cols].values.astype(float)
    y = df[config.SCORE_COL].values.astype(float)
    return X, y, emb, ling


# ---------- 1. permutation test --------------------------------------------

def permutation_test(X, y):
    print("\n[1] PERMUTATION TEST (shuffle labels; should collapse to mean)")
    rng = np.random.RandomState(config.RANDOM_STATE)
    fm, fmean, real_mae, real_r2 = _per_fold_errors(X, y)
    mean_mae = fmean.mean()
    print(f"  mean predictor MAE = {mean_mae:.3f}")
    print(f"  real labels:  MAE={real_mae:.3f}  R2={real_r2:+.3f}")
    sh_r2 = []
    for _ in range(N_SHUFFLES):
        ys = y.copy(); rng.shuffle(ys)
        _, _, _, r2 = _per_fold_errors(X, ys)
        sh_r2.append(r2)
    sh = np.mean(sh_r2)
    print(f"  shuffled R2 (avg of {N_SHUFFLES}): {sh:+.3f}")
    verdict = "CLEAN (no leakage)" if sh < 0.05 else "WARNING: possible leakage"
    signal = ("signal CONFIRMED" if (real_r2 > sh + 0.05 and real_r2 > 0.02)
              else "little/no signal")
    print(f"  -> {verdict}; {signal}")
    return {"real_r2": real_r2, "shuffled_r2": sh}
    


# ---------- 2. significance test -------------------------------------------

def significance_test(X, y, model="svr"):
    print("\n[2] SIGNIFICANCE TEST (paired Wilcoxon: model vs mean predictor)")
    fm, fmean, mae, r2 = _per_fold_errors(X, y, model)
    print(f"  per-fold model MAE: {np.round(fm,3)}")
    print(f"  per-fold mean MAE : {np.round(fmean,3)}")
    try:
        stat, p = wilcoxon(fm, fmean)
        print(f"  Wilcoxon p = {p:.4f}  (model {'<' if fm.mean()<fmean.mean() else '>='} mean)")
        print(f"  -> {'significant' if p < 0.05 else 'NOT significant'} at alpha=0.05 "
              f"(note: only {N_OUTER} folds, so power is limited)")
    except ValueError as e:
        print(f"  Wilcoxon could not run: {e}")
    return {"model_mae": float(fm.mean()), "mean_mae": float(fmean.mean())}


# ---------- 3. ablation: embeddings vs +linguistic -------------------------

def ablation_test(paths, model="svr"):
    print("\n[3] ABLATION (embeddings-only vs embeddings+linguistic)")
    Xe, y, emb, ling = load_features(paths, want_linguistic=False)
    Xl, _, _, _ = load_features(paths, want_linguistic=True)
    if not ling:
        print("  no ling_ features present; skipping.")
        return None
    _, _, mae_e, r2_e = _per_fold_errors(Xe, y, model)
    _, _, mae_l, r2_l = _per_fold_errors(Xl, y, model)
    print(f"  embeddings only      : MAE={mae_e:.3f}  R2={r2_e:+.3f}  ({len(emb)} feat)")
    print(f"  embeddings+linguistic: MAE={mae_l:.3f}  R2={r2_l:+.3f}  ({len(emb)+len(ling)} feat)")
    delta = mae_e - mae_l
    print(f"  -> linguistic features {'help' if delta>0 else 'do NOT help'} "
          f"(ΔMAE={delta:+.3f})")
    return {"mae_emb": mae_e, "mae_emb_ling": mae_l}


# ---------- 4. seed stability ----------------------------------------------

def seed_stability(X, y, model="svr", seeds=(42, 0, 1, 7, 123)):
    print("\n[4] SEED STABILITY (re-run across CV seeds)")
    maes, r2s = [], []
    for s in seeds:
        _, _, mae, r2 = _per_fold_errors(X, y, model, seed=s)
        maes.append(mae); r2s.append(r2)
        print(f"  seed {s:>4}: MAE={mae:.3f}  R2={r2:+.3f}")
    print(f"  -> MAE {np.mean(maes):.3f} ± {np.std(maes):.3f}; "
          f"R2 {np.mean(r2s):+.3f} ± {np.std(r2s):.3f} across seeds")
    return {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes))}


# ---------- 5. held-out test evaluation (needs test data) ------------------

def heldout_eval(train_paths, model="ridge"):
    print("\n[5] HELD-OUT TEST EVALUATION")
    test_feat = config.OUT_DIR / "text_features_mpnet_TEST.parquet"
    test_split = getattr(config, "TEST_SPLIT", None)
    if not test_feat.exists() or test_split is None or not (config.DATA_ROOT / test_split).exists():
        print("  SKIPPED: test-set features/labels not found.")
        print("  To enable: download the 47 test participants, build their text")
        print("  features as text_features_mpnet_TEST.parquet, and set")
        print("  config.TEST_SPLIT to the test split CSV. Then re-run.")
        return None
    # train on all train+dev, evaluate once on test
    Xtr, ytr, emb, ling = load_features(train_paths)
    test = pd.read_parquet(test_feat)
    tlab = pd.read_csv(config.DATA_ROOT / test_split)
    tlab.columns = [c.strip() for c in tlab.columns]
    tdf = test.merge(tlab[[config.ID_COL, config.SCORE_COL]], on=config.ID_COL)
    cols = emb + ling
    Xte = tdf[cols].values.astype(float)
    yte = tdf[config.SCORE_COL].values.astype(float)
    est, grid = _est(model)
    gs = GridSearchCV(_pipe(est), grid, scoring="neg_mean_absolute_error",
                      cv=KFold(N_INNER, shuffle=True, random_state=config.RANDOM_STATE),
                      n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    print(f"  held-out test (n={len(yte)}): MAE={mean_absolute_error(yte,pred):.3f}  "
          f"R2={r2_score(yte,pred):+.3f}")
    return {"test_mae": float(mean_absolute_error(yte, pred)),
            "test_r2": float(r2_score(yte, pred))}


def main():
    args = sys.argv[1:]
    if not args:
        g = sorted(glob.glob(str(config.OUT_DIR / "text_features_mpnet.parquet")))
        if not g:
            print("Usage: python3 validation_tests.py <features.parquet>")
            return
        args = [g[0]]
    paths = [a if "/" in a else str(config.OUT_DIR / a) for a in args]
    print(f"Feature file(s): {[p.split('/')[-1] for p in paths]}")
    X, y, emb, ling = load_features(paths)
    print(f"n={len(y)}, {len(emb)} embedding + {len(ling)} linguistic features")

    permutation_test(X, y)
    significance_test(X, y)
    ablation_test(paths)
    seed_stability(X, y)
    heldout_eval(paths)
    print("\nDone.")


if __name__ == "__main__":
    main()