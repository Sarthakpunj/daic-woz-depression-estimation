"""
encoder_comparison.py — Item 1: test newer frozen encoders against mpnet.

Honest framing: ALL encoders are reported. If a newer encoder beats mpnet, the
headline improves with no methodological change (still frozen, same chunk-pool,
same nested CV). If none does, that is itself an honest ablation: performance
saturates at mpnet-class encoders.

Encoders tested (all frozen sentence embeddings):
    mpnet  - all-mpnet-base-v2        (2021 reference)
    bge    - BAAI/bge-large-en-v1.5
    e5     - intfloat/e5-large-v2      (REQUIRES "query: " prefix on inputs)
    gte    - thenlper/gte-large

Run:
    python3 encoder_comparison.py

Note: large models download on first use; run on a machine with disk + bandwidth.
Reports MAE / R2 / F1 per encoder x regressor. Nothing is hidden or cherry-picked.
"""

import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

CHUNK_WORDS = 200
PHQ8_CUTOFF = 10
DROP_TOKENS = ("scrubbed_entry", "<laughter>", "xxx", "<sync>", "<synch>")

ENCODERS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", ""),
    "bge":   ("BAAI/bge-large-en-v1.5", ""),
    "e5":    ("intfloat/e5-large-v2", "query: "),   # e5 needs this prefix
    "gte":   ("thenlper/gte-large", ""),
}


def read_transcript(pid):
    folder = config.DATA_ROOT / f"{pid}_P"
    path = folder / f"{pid}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "speaker" not in df.columns or "value" not in df.columns:
        return None
    df = df[df["speaker"].astype(str).str.strip().str.lower() == "participant"]
    parts = []
    for v in df["value"].astype(str):
        s = v.strip()
        if not s or any(t in s.lower() for t in DROP_TOKENS):
            continue
        parts.append(s)
    text = " ".join(parts).strip()
    return text if text else None


def chunk_pool(text, model, prefix):
    words = text.split()
    chunks = [prefix + " ".join(words[i:i+CHUNK_WORDS])
              for i in range(0, len(words), CHUNK_WORDS)]
    embs = model.encode(chunks, show_progress_bar=False)
    return np.asarray(embs).mean(axis=0)


def load_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn); d.columns=[c.strip() for c in d.columns]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    return dict(zip(df[config.ID_COL].astype(int), df[config.SCORE_COL].astype(float)))


def nested_cv(X, y, regressor):
    outer = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    at, ap = [], []
    for tr, te in outer.split(X):
        if regressor == "ridge":
            est, grid = Ridge(random_state=config.RANDOM_STATE), {"clf__alpha":[1.0,10.0,100.0]}
        elif regressor == "svr":
            est, grid = SVR(), {"clf__C":[1.0,10.0]}
        else:
            est, grid = RandomForestRegressor(random_state=config.RANDOM_STATE), {"clf__n_estimators":[200]}
        pipe = Pipeline([("imp",SimpleImputer(strategy="median")),
                         ("sc",StandardScaler()),("clf",est)])
        gs = GridSearchCV(pipe, grid, scoring="neg_mean_absolute_error",
                          cv=KFold(3,shuffle=True,random_state=config.RANDOM_STATE), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        at.extend(y[te]); ap.extend(gs.best_estimator_.predict(X[te]))
    at, ap = np.array(at), np.array(ap)
    return (mean_absolute_error(at,ap), r2_score(at,ap),
            f1_score(at>=PHQ8_CUTOFF, ap>=PHQ8_CUTOFF, zero_division=0))


def main():
    from sentence_transformers import SentenceTransformer
    labels = load_labels()
    ids = sorted(labels)
    texts = {pid: read_transcript(pid) for pid in ids}
    texts = {pid:t for pid,t in texts.items() if t}
    y = np.array([labels[pid] for pid in texts])
    print(f"{len(texts)} participants with transcripts\n")

    # mean predictor
    from sklearn.model_selection import KFold as KF
    mp_ap=[]; mp_at=[]
    for tr,te in KF(5,shuffle=True,random_state=config.RANDOM_STATE).split(y):
        mp_at.extend(y[te]); mp_ap.extend([y[tr].mean()]*len(te))
    print(f"Mean predictor: MAE={mean_absolute_error(mp_at,mp_ap):.3f}\n")

    results = []
    for name,(path,prefix) in ENCODERS.items():
        print(f"=== {name} ({path}) ===")
        try:
            model = SentenceTransformer(path)
        except Exception as e:
            print(f"  could not load ({e}); skipping\n"); continue
        X = np.vstack([chunk_pool(texts[pid], model, prefix) for pid in texts])
        for reg in ["ridge","svr","rf"]:
            mae,r2,f1 = nested_cv(X, y, reg)
            print(f"  {reg:5s}: MAE={mae:.3f}  R2={r2:+.3f}  F1={f1:.3f}")
            results.append({"encoder":name,"regressor":reg,"mae":mae,"r2":r2,"f1":f1})
        print()

    df = pd.DataFrame(results)
    out = config.OUT_DIR / "encoder_comparison.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    if len(df):
        best = df.loc[df["mae"].idxmin()]
        print(f"\nBest: {best['encoder']}-{best['regressor']} MAE={best['mae']:.3f} R2={best['r2']:+.3f}")
        mpnet_best = df[df.encoder=='mpnet']['mae'].min() if 'mpnet' in df.encoder.values else None
        if mpnet_best is not None:
            if best['mae'] < mpnet_best - 0.05:
                print(f"-> A newer encoder beats mpnet ({best['mae']:.3f} vs {mpnet_best:.3f}). Honest improvement.")
            else:
                print(f"-> No newer encoder clearly beats mpnet ({mpnet_best:.3f}). Performance saturates at mpnet-class (honest ablation).")


if __name__ == "__main__":
    main()