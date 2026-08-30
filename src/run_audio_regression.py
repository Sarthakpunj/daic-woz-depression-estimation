"""
run_audio_regression.py — nested-CV evaluation of audio embeddings.

Takes the parquet filename as a COMMAND-LINE ARGUMENT so there is no chance of
misattributing results across the six audio feature sets (2 models x 3 layers).
The filename is echoed in the output and written into the results CSV.

CRITICAL: the audio parquets contain ALL 188 participants (train+dev AND the
held-out test set). This script merges against the train/dev splits, so nested CV
runs on the ~141 train+dev ONLY; test participants are excluded and reserved for
the held-out evaluation.

Run:
    python3 run_audio_regression.py audio_w2v2_participant_layerLAST.parquet
    python3 run_audio_regression.py audio_wavlm_participant_layerMID.parquet

Appends to audio_results_all.csv so you build one complete table across runs.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import daic_woz_pipeline.src.config as config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

PHQ8_CUTOFF = 10
SEVERE_LO = 15
N_OUTER, N_INNER = 5, 3
RESULTS_CSV = "audio_results_all.csv"


def load_traindev_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn)
        d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)[[config.ID_COL, config.SCORE_COL]]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_audio_regression.py <audio_features.parquet>")
        print("\nAvailable audio parquets in outputs/:")
        for p in sorted(config.OUT_DIR.glob("audio_*.parquet")):
            print(f"  {p.name}")
        return

    fname = sys.argv[1]
    path = Path(fname) if "/" in fname else config.OUT_DIR / fname
    if not path.exists():
        print(f"ERROR: {path} not found.")
        return

    print(f"=== FEATURE FILE: {path.name} ===")
    audio = pd.read_parquet(path)
    print(f"  {audio.shape[0]} participants, {audio.shape[1]-1} embedding dims")

    labels = load_traindev_labels()
    df = audio.merge(labels, on=config.ID_COL, how="inner")
    print(f"  train+dev matched: {len(df)}  "
          f"({len(audio)-len(df)} test participants excluded, reserved for held-out)")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[emb_cols].values.astype(float)
    y = df[config.SCORE_COL].values.astype(float)

    outer = KFold(N_OUTER, shuffle=True, random_state=config.RANDOM_STATE)
    at, ap = [], []
    for tr, te in outer.split(X):
        at.extend(y[te]); ap.extend([y[tr].mean()] * len(te))
    mp_mae = mean_absolute_error(at, ap); mp_r2 = r2_score(at, ap)
    print(f"\n  mean predictor: MAE={mp_mae:.3f}  R2={mp_r2:+.3f}\n")

    rows = []
    for name in ["rf", "ridge", "svr"]:
        fold_maes, all_t, all_p, train_maes = [], [], [], []
        for tr, te in outer.split(X):
            if name == "ridge":
                est, grid = Ridge(random_state=config.RANDOM_STATE), {"clf__alpha":[1.0,10.0,100.0]}
            elif name == "svr":
                est, grid = SVR(), {"clf__C":[1.0,10.0]}
            else:
                est, grid = RandomForestRegressor(random_state=config.RANDOM_STATE,
                                                  n_jobs=-1), {"clf__n_estimators":[200]}
            pipe = Pipeline([("imp",SimpleImputer(strategy="median")),
                             ("sc",StandardScaler()),("clf",est)])
            gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                              cv=KFold(N_INNER, shuffle=True,
                                       random_state=config.RANDOM_STATE), n_jobs=-1)
            gs.fit(X[tr], y[tr])
            pred = gs.best_estimator_.predict(X[te])
            fold_maes.append(mean_absolute_error(y[te], pred))
            train_maes.append(mean_absolute_error(y[tr], gs.best_estimator_.predict(X[tr])))
            all_t.extend(y[te]); all_p.extend(pred)

        t, p = np.array(all_t), np.array(all_p)
        mae, r2 = mean_absolute_error(t,p), r2_score(t,p)
        f1 = f1_score(t>=PHQ8_CUTOFF, p>=PHQ8_CUTOFF, zero_division=0)
        sev = t >= SEVERE_LO
        mae_sev = mean_absolute_error(t[sev], p[sev]) if sev.sum() else float("nan")

        # Honest verdict: beating the mean on MAE alone is NOT enough; R2 must be
        # meaningfully positive. A tiny MAE win with R2 ~ 0 is noise.
        meaningful = (mae < mp_mae - 0.1) and (r2 > 0.02)
        print(f"  {name:6s}: MAE={mae:.3f}(+/-{np.std(fold_maes):.3f})  R2={r2:+.3f}  "
              f"F1={f1:.3f}  MAE_sev={mae_sev:.2f}  "
              f"-> {'MEANINGFUL' if meaningful else 'no meaningful signal'}")

        rows.append({"feature_file": path.name, "model": name,
                     "mae": round(mae,3), "r2": round(r2,3),
                     "mae_std": round(np.std(fold_maes),3),
                     "f1": round(f1,3), "mae_severe": round(mae_sev,2),
                     "overfit_gap": round(mae - np.mean(train_maes),3),
                     "mean_pred_mae": round(mp_mae,3),
                     "meaningful_signal": "yes" if meaningful else "no"})

    out = pd.DataFrame(rows)
    csv_path = config.OUT_DIR / RESULTS_CSV
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        prev = prev[prev["feature_file"] != path.name]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(csv_path, index=False)
    print(f"\n  appended to {csv_path.name}")


if __name__ == "__main__":
    main()