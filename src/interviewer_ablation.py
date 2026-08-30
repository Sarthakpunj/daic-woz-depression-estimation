"""
interviewer_ablation.py — tests the effect of INCLUDING the interviewer (Ellie).

Rationale: on DAIC-WOZ, the interviewer's turns have been documented (Burdisso;
Danylenko & Unold) to act as a leakage shortcut — models can exploit which
questions were asked rather than the participant's own language. This script
builds two text-feature sets from the SAME transcripts:

  (A) participant-only   (Ellie removed)   -- your main pipeline
  (B) participant+Ellie  (Ellie included)  -- the ablation

and runs BOTH through the same nested CV (mpnet embeddings, SVR).

Interpretation:
  - If (B) scores HIGHER than (A), that is NOT a better result — it is direct
    evidence of the interviewer shortcut, and justifies removing Ellie.
  - If (B) is similar or worse, it shows Ellie adds no useful context, which
    also justifies removal (simpler, no shortcut risk).

Run:
    python3 interviewer_ablation.py

Requires the same libraries as build_text_features.py (sentence-transformers).
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from pathlib import Path

import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, f1_score

MPNET = "sentence-transformers/all-mpnet-base-v2"
CHUNK_WORDS = 200
PHQ8_CUTOFF = 10
DROP_TOKENS = ("scrubbed_entry", "<laughter>", "xxx", "<sync>", "<synch>")


def read_transcript(pid, include_ellie):
    """Read a participant's transcript; return concatenated text.
       include_ellie=False -> participant turns only (main pipeline).
       include_ellie=True  -> all turns (ablation)."""
    folder = config.DATA_ROOT / f"{pid}_P"
    path = folder / f"{pid}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    # DAIC transcripts are TAB-separated despite the .csv extension
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "speaker" not in df.columns or "value" not in df.columns:
        return None
    if not include_ellie:
        df = df[df["speaker"].astype(str).str.strip().str.lower() == "participant"]
    parts = []
    for v in df["value"].astype(str):
        s = v.strip()
        if not s:
            continue
        low = s.lower()
        if any(tok in low for tok in DROP_TOKENS):
            continue
        parts.append(s)
    text = " ".join(parts).strip()
    return text if text else None


def chunk_pool(text, model):
    words = text.split()
    if not words:
        return None
    chunks = [" ".join(words[i:i+CHUNK_WORDS])
              for i in range(0, len(words), CHUNK_WORDS)]
    embs = model.encode(chunks, show_progress_bar=False)
    return np.asarray(embs).mean(axis=0)


def build(ids_scores, include_ellie, model):
    rows = []
    for pid, score in ids_scores.items():
        txt = read_transcript(pid, include_ellie)
        if txt is None:
            continue
        vec = chunk_pool(txt, model)
        if vec is None:
            continue
        row = {f"emb_{j}": float(v) for j, v in enumerate(vec)}
        row[config.ID_COL] = pid
        row[config.SCORE_COL] = score
        rows.append(row)
    return pd.DataFrame(rows)


def nested_cv(df):
    cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[cols].values.astype(float)
    y = df[config.SCORE_COL].values.astype(float)
    outer = KFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    at, ap = [], []
    for tr, te in outer.split(X):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", Ridge(random_state=config.RANDOM_STATE))])
        gs = GridSearchCV(pipe, {"clf__alpha": [1.0, 10.0, 100.0]},
                          scoring="neg_mean_absolute_error",
                          cv=KFold(3, shuffle=True, random_state=config.RANDOM_STATE),
                          n_jobs=-1)
        gs.fit(X[tr], y[tr])
        p = gs.best_estimator_.predict(X[te])
        at.extend(y[te]); ap.extend(p)
    at, ap = np.array(at), np.array(ap)
    return (mean_absolute_error(at, ap), r2_score(at, ap),
            f1_score(at >= PHQ8_CUTOFF, ap >= PHQ8_CUTOFF, zero_division=0))


def load_labels():
    rows = []
    for fn in [config.TRAIN_SPLIT, config.DEV_SPLIT]:
        d = pd.read_csv(config.DATA_ROOT / fn)
        d.columns = [c.strip() for c in d.columns]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    return dict(zip(df[config.ID_COL].astype(int),
                    df[config.SCORE_COL].astype(float)))


def main():
    from sentence_transformers import SentenceTransformer
    print("Loading labels...")
    ids_scores = load_labels()
    print(f"  {len(ids_scores)} participants")

    print("Loading mpnet...")
    model = SentenceTransformer(MPNET)

    print("\nBuilding PARTICIPANT-ONLY features (Ellie removed)...")
    df_part = build(ids_scores, include_ellie=False, model=model)
    print(f"  {df_part.shape[0]} participants embedded")

    print("Building PARTICIPANT+ELLIE features (Ellie included)...")
    df_both = build(ids_scores, include_ellie=True, model=model)
    print(f"  {df_both.shape[0]} participants embedded")

    print("\nRunning nested CV on each...")
    mae_p, r2_p, f1_p = nested_cv(df_part)
    mae_b, r2_b, f1_b = nested_cv(df_both)

    print("\n=== INTERVIEWER-INCLUSION ABLATION (mpnet + Ridge) ===")
    print(f"  participant-only  : MAE={mae_p:.3f}  R2={r2_p:+.3f}  F1={f1_p:.3f}")
    print(f"  participant+Ellie : MAE={mae_b:.3f}  R2={r2_b:+.3f}  F1={f1_b:.3f}")
    print("\nInterpretation:")
    if mae_b < mae_p - 0.1:
        print("  Including Ellie IMPROVES scores -> evidence of the interviewer")
        print("  shortcut. This JUSTIFIES removing Ellie (the gain is a leak,")
        print("  not genuine participant signal).")
    elif abs(mae_b - mae_p) <= 0.1:
        print("  Including Ellie makes little difference -> Ellie adds no useful")
        print("  context. Removing her is justified (simpler, no shortcut risk).")
    else:
        print("  Including Ellie WORSENS scores -> Ellie's turns are noise.")
        print("  Removing her is clearly justified.")


if __name__ == "__main__":
    main()