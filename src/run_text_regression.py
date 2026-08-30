"""
run_text_regression.py — evaluate text-feature sets through leakage-free nested
participant-level CV, with the diagnostics [41] insists on.

Enhancements:
  * R^2 reported per fold AND pooled  -- the goodness-of-fit metric [41] says
    everyone wrongly omits. Positive R^2 = explains real variance (beats the
    mean predictor in the way that matters); negative = worse than the mean.
  * Optional PCA INSIDE each fold (USE_PCA) to test whether reducing 768/384-dim
    embeddings on n=142 stabilises the result. PCA is fit on the TRAIN fold only
    -- never on test -- so there is no leakage.
    
  * Works for any number of text_features_<tag>.parquet (mpnet, minilm,
    mentalbert, ...).

Self-contained: does NOT modify train_regression.py, so the audio+visual
baseline is untouched. Reuses the same KFold/RANDOM_STATE so the protocol
matches the baseline exactly.

    python3 run_text_regression.py
"""

import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             f1_score, r2_score)

import daic_woz_pipeline.src.config as config

PHQ8_CUTOFF = 10
USE_PCA = False        # set True to test PCA-inside-folds
PCA_COMPONENTS = 30
N_OUTER, N_INNER = 5, 3


def model_and_grid(name):
    if name == "rf":
        return RandomForestRegressor(random_state=config.RANDOM_STATE), \
               {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10]}
    if name == "ridge":
        return Ridge(random_state=config.RANDOM_STATE), \
               {"clf__alpha": [1.0, 10.0, 100.0]}
    if name == "svr":
        return SVR(), {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}
    raise ValueError(name)


def build_pipeline(est):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if USE_PCA:
        steps.append(("pca", PCA(n_components=PCA_COMPONENTS,
                                 random_state=config.RANDOM_STATE)))
    steps.append(("clf", est))
    return Pipeline(steps)


def evaluate(X, y, model_name):
    outer = KFold(n_splits=N_OUTER, shuffle=True,
                  random_state=config.RANDOM_STATE)
    maes, rmses, r2s, train_maes = [], [], [], []
    all_true, all_pred = [], []
    for tr, te in outer.split(X):
        est, grid = model_and_grid(model_name)
        pipe = build_pipeline(est)
        inner = KFold(n_splits=N_INNER, shuffle=True,
                      random_state=config.RANDOM_STATE)
        search = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                              cv=inner, n_jobs=-1)
        search.fit(X[tr], y[tr])
        best = search.best_estimator_
        pred = best.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        if len(te) > 1 and np.var(y[te]) > 0:
            r2s.append(r2_score(y[te], pred))
        train_maes.append(mean_absolute_error(y[tr], best.predict(X[tr])))
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())

    at, ap = np.array(all_true), np.array(all_pred)
    return {
        "mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "rmse": float(np.mean(rmses)),
        "r2_perfold_mean": float(np.mean(r2s)) if r2s else float("nan"),
        "r2_pooled": float(r2_score(at, ap)),
        "f1_pooled": float(f1_score(at >= PHQ8_CUTOFF, ap >= PHQ8_CUTOFF,
                                    zero_division=0)),
        "overfit_gap": float(np.mean(maes) - np.mean(train_maes)),
        "mae_severe": _severe_mae(at, ap),
    }


def _severe_mae(at, ap):
    m = (at >= 15) & (at <= 24)
    return float(mean_absolute_error(at[m], ap[m])) if m.sum() else float("nan")


def mean_predictor(X, y):
    outer = KFold(n_splits=N_OUTER, shuffle=True,
                  random_state=config.RANDOM_STATE)
    maes, all_true, all_pred = [], [], []
    for tr, te in outer.split(X):
        pred = np.full(len(te), y[tr].mean())
        maes.append(mean_absolute_error(y[te], pred))
        all_true.extend(y[te].tolist()); all_pred.extend(pred.tolist())
    at, ap = np.array(all_true), np.array(all_pred)
    return {"mae": float(np.mean(maes)), "r2_pooled": float(r2_score(at, ap))}


def load_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        df = pd.read_csv(config.DATA_ROOT / fn)
        df.columns = [c.strip() for c in df.columns]
        rows.append(df)
    lab = pd.concat(rows, ignore_index=True)
    return lab[[config.ID_COL, config.SCORE_COL]]


def main():
    labels = load_labels()
    files = sorted(glob.glob(str(config.OUT_DIR / "text_features_*.parquet")))
    if not files:
        print("No text_features_*.parquet found. Run build_text_features.py.")
        return

    rows = []
    mean_done = False
    for f in files:
        tag = f.split("text_features_")[-1].replace(".parquet", "")
        feats = pd.read_parquet(f)
        df = feats.merge(labels, on=config.ID_COL, how="inner")
        cols = [c for c in feats.columns
                if c.startswith("emb_") or c.startswith("ling_")]
        X = df[cols].values
        y = df[config.SCORE_COL].values.astype(float)

        if not mean_done:
            mp = mean_predictor(X, y)
            mean_mae, mean_r2 = mp["mae"], mp["r2_pooled"]
            print(f"Mean predictor: MAE={mean_mae:.3f}  R2={mean_r2:.3f}  "
                  f"(n={len(df)})  PCA={'on' if USE_PCA else 'off'}\n")
            rows.append({"text_model": "-", "model": "mean_predictor",
                         "mae": round(mean_mae, 3), "r2_pooled": round(mean_r2, 3)})
            mean_done = True

        print(f"=== {tag}  ({len(cols)} features) ===")
        for m in ["rf", "ridge", "svr"]:
            r = evaluate(X, y, m)
            beat_mae = "yes" if r["mae"] < mean_mae else "NO"
            beat_r2 = "yes" if r["r2_pooled"] > 0 else "NO"
            rows.append({
                "text_model": tag, "model": m,
                "mae": round(r["mae"], 3), "mae_std": round(r["mae_std"], 3),
                "r2_pooled": round(r["r2_pooled"], 3),
                "r2_perfold": round(r["r2_perfold_mean"], 3),
                "f1_pooled": round(r["f1_pooled"], 3),
                "mae_severe": round(r["mae_severe"], 2),
                "overfit_gap": round(r["overfit_gap"], 3),
                "beats_mean_mae": beat_mae, "pos_r2": beat_r2,
            })
            print(f"  {m}: MAE={r['mae']:.3f}(+/-{r['mae_std']:.3f}) "
                  f"R2_pooled={r['r2_pooled']:.3f} F1={r['f1_pooled']:.3f} "
                  f"beats_mean={beat_mae} posR2={beat_r2}")

    table = pd.DataFrame(rows)
    tag = "_pca" if USE_PCA else ""
    out = config.OUT_DIR / f"text_results{tag}.csv"
    table.to_csv(out, index=False)
    print(f"\n=== Text modality (nested participant-level CV) ===")
    print(table.to_string(index=False))
    print(f"\nMean predictor MAE={mean_mae:.3f} R2={mean_r2:.3f}. "
          f"Positive R2 = explains real variance (the bar [41] cares about).")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()