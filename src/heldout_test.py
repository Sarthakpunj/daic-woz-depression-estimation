"""
heldout_test_both.py — final held-out evaluation for BOTH modalities, so the
baseline-vs-text comparison is fair (same unseen 47 test participants).

For each modality it trains on ALL 142 train+dev participants and evaluates ONCE
on the 47 held-out test participants:
  * baseline : audio+visual functionals (re-uses features.extract_participant)
  * text     : mpnet embeddings        (re-uses build_text_features helpers)

Prerequisites (all local, no GPU):
  - test split CSV in DATA_ROOT (47 participants + PHQ-8 scores); set TEST_SPLIT.
  - test participants downloaded into DATA_ROOT/<id>_P/ with the needed files:
      baseline needs COVAREP/FORMANT/CLNF; text needs TRANSCRIPT.
    The script downloads+trims any missing test participants (keeps all 6 files).

Run:
    python3 heldout_test_both.py            # both modalities
    python3 heldout_test_both.py text       # text only
    python3 heldout_test_both.py baseline   # baseline only
"""

import sys
import subprocess
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

TEST_SPLIT = getattr(config, "TEST_SPLIT", "full_test_split.csv")
MPNET = "sentence-transformers/all-mpnet-base-v2"
PHQ8_CUTOFF = 10
BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"

from daic_woz_pipeline.src.build_text_features import (read_participant_text, chunk_pool_embed,
                                 linguistic_features, ADD_LINGUISTIC)
try:
    from trim_participants import keep_file
except Exception:
    keep_file = None
import features as featmod


# ---------- test split + download ------------------------------------------

def test_ids_scores():
    path = config.DATA_ROOT / TEST_SPLIT
    if not path.exists():
        cands = list(config.DATA_ROOT.glob("*test*split*.csv"))
        if cands:
            path = cands[0]
    if not path.exists():
        raise FileNotFoundError(
            f"Test split not found in {config.DATA_ROOT}. Set TEST_SPLIT.")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Test split may use 'PHQ_Score' while train/dev use 'PHQ8_Score'. Find any
    # column containing 'phq' and 'score' (case-insensitive).
    score_col = None
    if config.SCORE_COL in df.columns:
        score_col = config.SCORE_COL
    else:
        for c in df.columns:
            cl = c.lower()
            if "phq" in cl and "score" in cl:
                score_col = c
                break
    if score_col is None:
        raise ValueError(
            f"No PHQ score column in {path.name}. Columns: {list(df.columns)}")
    ids = [int(x) for x in df[config.ID_COL].tolist()]
    scores = dict(zip(ids, df[score_col].astype(float).tolist()))
    print(f"Test split: {len(ids)} participants from {path.name} (score: {score_col})")
    return ids, scores


def ensure_files(pid):
    folder = config.DATA_ROOT / f"{pid}_P"
    if (folder / f"{pid}_COVAREP.csv").exists() and \
       (folder / f"{pid}_TRANSCRIPT.csv").exists():
        return True
    zip_path = config.DATA_ROOT / f"{pid}_P.zip"
    url = f"{BASE_URL}/{pid}_P.zip"
    print(f"  [{pid}] downloading...", flush=True)
    r = subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(zip_path), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [{pid}] download failed"); 
        if zip_path.exists(): zip_path.unlink()
        return False
    folder.mkdir(exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            if any(n.startswith(f"{pid}_P/") for n in names):
                z.extractall(config.DATA_ROOT)
            else:
                z.extractall(folder)
    except zipfile.BadZipFile:
        print(f"  [{pid}] bad zip"); zip_path.unlink(); return False
    if keep_file is not None:
        for f in folder.iterdir():
            if f.is_file() and not keep_file(f.name):
                f.unlink()
    zip_path.unlink()
    return (folder / f"{pid}_COVAREP.csv").exists()


# ---------- training-set loaders -------------------------------------------

def load_train_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn)
        d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def evaluate(name, Xtr, ytr, Xte, yte, model="ridge"):
    if model == "ridge":
        est, grid = Ridge(random_state=config.RANDOM_STATE), {"clf__alpha":[1.0,10.0,100.0]}
    else:
        est, grid = SVR(), {"clf__C":[1.0,10.0]}
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", est)])
    gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                      cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE),
                      n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    mp = np.full(len(yte), ytr.mean())
    print(f"\n=== {name} (held-out test, n={len(yte)}) ===")
    print(f"  mean predictor : MAE={mean_absolute_error(yte,mp):.3f}  R2={r2_score(yte,mp):+.3f}")
    print(f"  {name:13s}: MAE={mean_absolute_error(yte,pred):.3f}  "
          f"R2={r2_score(yte,pred):+.3f}  "
          f"F1={f1_score(yte>=PHQ8_CUTOFF, pred>=PHQ8_CUTOFF, zero_division=0):.3f}  "
          f"({gs.best_params_})")
    beat = mean_absolute_error(yte,pred) < mean_absolute_error(yte,mp)
    print(f"  -> {'BEATS' if beat else 'does NOT beat'} the mean predictor.")
    return pred


def run_text(ids, scores):
    from sentence_transformers import SentenceTransformer
    train_path = config.OUT_DIR / "text_features_mpnet.parquet"
    if not train_path.exists():
        print("  text: train parquet missing; run build_text_features.py"); return
    train_feat = pd.read_parquet(train_path)
    labels = load_train_labels()[[config.ID_COL, config.SCORE_COL]]
    tr = train_feat.merge(labels, on=config.ID_COL, how="inner")
    cols = [c for c in train_feat.columns if c.startswith("emb_") or c.startswith("ling_")]

    model = SentenceTransformer(MPNET)
    rows = []
    for pid in ids:
        txt = read_participant_text(pid)
        if txt is None: continue
        vec = chunk_pool_embed(txt, model)
        row = {f"emb_{j}": float(v) for j,v in enumerate(vec)}
        if ADD_LINGUISTIC: row.update(linguistic_features(txt))
        row[config.SCORE_COL] = scores[pid]
        rows.append(row)
    te = pd.DataFrame(rows)
    cols = [c for c in cols if c in te.columns]
    evaluate("text mpnet+Ridge", tr[cols].values.astype(float),
             tr[config.SCORE_COL].values.astype(float),
             te[cols].values.astype(float),
             te[config.SCORE_COL].values.astype(float), "ridge")


def run_baseline(ids, scores):
    # train: reuse existing baseline parquets
    frames = []
    for fn in ["train_features.parquet", "dev_features.parquet"]:
        p = config.OUT_DIR / fn
        if p.exists(): frames.append(pd.read_parquet(p))
    if not frames:
        print("  baseline: train parquets missing; run build_dataset.py"); return
    tr = pd.concat(frames, ignore_index=True)
    if config.SCORE_COL not in tr.columns:
        labels = load_train_labels()[[config.ID_COL, config.SCORE_COL]]
        tr = tr.merge(labels, on=config.ID_COL, how="inner")
    gcol = getattr(config, "GENDER_COL", "Gender")
    lab = {config.ID_COL, gcol, config.SCORE_COL, "PHQ8_Binary"}
    cols = [c for c in tr.columns if c not in lab and pd.api.types.is_numeric_dtype(tr[c])]

    # test: extract audio+visual features for the 47
    rows = []
    for pid in ids:
        try:
            feat = featmod.extract_participant(config.DATA_ROOT, pid)
        except Exception as e:
            print(f"  [{pid}] baseline extract failed: {e}"); continue
        feat[config.SCORE_COL] = scores[pid]
        rows.append(feat)
    te = pd.DataFrame(rows)
    cols = [c for c in cols if c in te.columns]
    evaluate("baseline a+v SVR", tr[cols].values.astype(float),
             tr[config.SCORE_COL].values.astype(float),
             te[cols].values.astype(float),
             te[config.SCORE_COL].values.astype(float), "svr")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    ids, scores = test_ids_scores()
    print("\nEnsuring test participant files (download if missing)...")
    have = sum(ensure_files(p) for p in ids)
    print(f"  available: {have}/{len(ids)}")

    if which in ("both", "text"):
        run_text(ids, scores)
    if which in ("both", "baseline"):
        run_baseline(ids, scores)
    print("\nDone. (Held-out test = the honest generalisation check for both modalities.)")


if __name__ == "__main__":
    main()