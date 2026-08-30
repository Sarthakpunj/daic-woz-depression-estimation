"""
heldout_test_audio.py — held-out evaluation of the audio modality.

Trains on the ~141 train+dev participants and evaluates ONCE on the 47 held-out
test participants (already present in the audio parquets, which contain all 188).

Mirrors the text/baseline held-out design so all three modalities are compared on
the same unseen participants. Also reports the per-group (gender) breakdown for
protocol consistency with the text chapter.

Run:
    python3 heldout_test_audio.py audio_w2v2_participant_layerLAST.parquet
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

PHQ8_CUTOFF = 10


def load_split(fn):
    d = pd.read_csv(config.DATA_ROOT / fn)
    d.columns = [c.strip() for c in d.columns]
    return d


def find_score_col(df):
    if config.SCORE_COL in df.columns:
        return config.SCORE_COL
    for c in df.columns:
        cl = c.lower()
        if "phq" in cl and "score" in cl:
            return c
    raise ValueError(f"no PHQ score column in {list(df.columns)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 heldout_test_audio.py <audio_features.parquet>")
        print("\nAvailable:")
        for p in sorted(config.OUT_DIR.glob("audio_*participant*.parquet")):
            print(f"  {p.name}")
        return

    fname = sys.argv[1]
    path = Path(fname) if "/" in fname else config.OUT_DIR / fname
    if not path.exists():
        print(f"ERROR: {path} not found."); return

    print(f"=== AUDIO HELD-OUT TEST: {path.name} ===\n")
    audio = pd.read_parquet(path)
    emb_cols = [c for c in audio.columns if c.startswith("emb_")]

    # train+dev
    tr_lab = pd.concat([load_split(config.TRAIN_SPLIT),
                        load_split(config.DEV_SPLIT)], ignore_index=True)
    tr_lab = tr_lab[[config.ID_COL, config.SCORE_COL]]
    tr = audio.merge(tr_lab, on=config.ID_COL, how="inner")

    # test split
    cands = list(config.DATA_ROOT.glob("*test*split*.csv"))
    if not cands:
        print("ERROR: test split CSV not found."); return
    ts = pd.read_csv(cands[0]); ts.columns = [c.strip() for c in ts.columns]
    scol = find_score_col(ts)
    gcol = "Gender" if "Gender" in ts.columns else None
    keep = [config.ID_COL, scol] + ([gcol] if gcol else [])
    te = audio.merge(ts[keep], on=config.ID_COL, how="inner")

    print(f"  train+dev: {len(tr)}   held-out test: {len(te)}   "
          f"(test split file: {cands[0].name}, score col: {scol})\n")

    Xtr = tr[emb_cols].values.astype(float)
    ytr = tr[config.SCORE_COL].values.astype(float)
    Xte = te[emb_cols].values.astype(float)
    yte = te[scol].values.astype(float)

    for model_name in ["svr", "ridge"]:
        if model_name == "svr":
            est, grid = SVR(), {"clf__C": [1.0, 10.0]}
        else:
            est, grid = Ridge(random_state=config.RANDOM_STATE), {"clf__alpha": [1.0, 10.0, 100.0]}
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("clf", est)])
        gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                          cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE),
                          n_jobs=-1)
        gs.fit(Xtr, ytr)
        pred = gs.best_estimator_.predict(Xte)
        mp = np.full(len(yte), ytr.mean())

        mae = mean_absolute_error(yte, pred); r2 = r2_score(yte, pred)
        f1 = f1_score(yte >= PHQ8_CUTOFF, pred >= PHQ8_CUTOFF, zero_division=0)
        mp_mae = mean_absolute_error(yte, mp); mp_r2 = r2_score(yte, mp)

        print(f"  mean predictor  : MAE={mp_mae:.3f}  R2={mp_r2:+.3f}")
        print(f"  audio {model_name:6s}  : MAE={mae:.3f}  R2={r2:+.3f}  F1={f1:.3f}  {gs.best_params_}")
        meaningful = (mae < mp_mae - 0.1) and (r2 > 0.02)
        print(f"  -> {'MEANINGFUL signal' if meaningful else 'NO meaningful signal (R2 ~ 0 or negative)'}")

        # per-group breakdown (protocol consistency with the text chapter)
        if gcol:
            te2 = te.copy(); te2["pred"] = pred
            gaps = {}
            for g, sub in te2.groupby(gcol):
                yt = sub[scol].values; yp = sub["pred"].values
                gm = mean_absolute_error(yt, yp)
                gmp = mean_absolute_error(yt, np.full(len(yt), ytr.mean()))
                gaps[int(g)] = (gm, gmp)
                print(f"     group {int(g)}: n={len(sub):2d}  model MAE={gm:.3f}  "
                      f"mean-pred MAE={gmp:.3f}")
            if len(gaps) == 2:
                a, b = sorted(gaps)
                print(f"     model MAE gap={abs(gaps[a][0]-gaps[b][0]):.3f}  "
                      f"mean-pred gap={abs(gaps[a][1]-gaps[b][1]):.3f}")
        print()

    print("Compare: text held-out MAE 4.24 R2 +0.31 F1 0.59 (meaningful);")
    print("         baseline held-out MAE 5.34 R2 -0.06 F1 0.00 (null).")


if __name__ == "__main__":
    main()